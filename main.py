"""
加密货币量化交易系统 v2.1
工业级架构 · 全量数据 · 政体自适应策略 · 智能执行 · 三层风控

v2.1 变更：
  - 新增 --backtest-v2: 政体自适应回测（完整 FeatureEngine + Regime 过滤）
  - 模拟/实盘交易从 ML 信号切换为 RegimeAdaptiveStrategy
  - 回测与实盘共享同一策略逻辑，保证策略一致性

用法:
    python main.py                          # 模拟交易（Regime 策略）
    python main.py --live                   # 实盘交易
    python main.py --backtest-v2            # 政体自适应回测（默认快照 backtest_v2_regime_snapshot.txt）
    python main.py --backtest-v2 --no-regime # 对照组（默认 backtest_v2_no_regime_snapshot.txt）
    python main.py --backtest               # 原版 ML（默认 backtest_v1_ml_snapshot.txt）
    python main.py --validate-strategy      # 政体策略 2 月验证回测
    python main.py --train                  # 仅训练 ML 模型
    python main.py --sync-data              # 仅同步数据
    python main.py --validate               # 数据质量检查
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from config.loader import load_config
from core.engine import TradingEngine
from data.client import BinanceClient
from data.storage import Storage
from data.historical import HistoryDownloader
from data.features import FeatureEngine
from alpha.ml_model import AlphaModel
from alpha.regime_strategy import (
    RegimeAdaptiveStrategy, RegimeStrategyConfig, add_regime_column,
    REGIME_BULL_TREND, REGIME_BULL_WEAK, REGIME_RANGE,
    REGIME_BEAR_WEAK, REGIME_BEAR_TREND,
)
from utils.logger import get_logger
from utils.metrics import generate_report, format_report

import numpy as np
import pandas as pd

logger = get_logger("main")


def _resolve_bt_snapshot_path(args: argparse.Namespace, mode: str) -> str:
    """
    未指定 --bt-snapshot 时，按回测类型使用不同默认文件名，避免互相覆盖。
    mode: v2_regime | v2_no_regime | v1_ml
    """
    if getattr(args, "bt_snapshot", None):
        return args.bt_snapshot
    if mode == "v2_regime":
        return "backtest_v2_regime_snapshot.txt"
    if mode == "v2_no_regime":
        return "backtest_v2_no_regime_snapshot.txt"
    if mode == "v1_ml":
        return "backtest_v1_ml_snapshot.txt"
    return "backtest_snapshot.txt"


# ═══════════════════════════════════════════════════════════════
# Regime 策略初始化（注入到 TradingEngine）
# ═══════════════════════════════════════════════════════════════

def _init_regime_on_engine(engine: TradingEngine):
    """
    给 TradingEngine 实例注入 RegimeAdaptiveStrategy。
    在 engine.warmup() 之前调用。
    """
    # 从 config 读取 regime 参数
    regime_cfg_raw = {}
    try:
        if hasattr(engine.config, 'strategy') and hasattr(engine.config.strategy, '_data'):
            regime_cfg_raw = engine.config.strategy._data.get("regime", {})
        elif hasattr(engine.config, 'strategy'):
            r = getattr(engine.config.strategy, 'regime', {})
            if hasattr(r, '_data'):
                regime_cfg_raw = r._data
            elif isinstance(r, dict):
                regime_cfg_raw = r
    except Exception:
        pass

    cfg = RegimeStrategyConfig(
        risk_per_trade=regime_cfg_raw.get("risk_per_trade", 0.03),
        risk_per_trade_weak=regime_cfg_raw.get("risk_per_trade_weak", 0.015),
        trailing_atr_mult=regime_cfg_raw.get("trailing_atr_mult", 2.5),
        take_profit_atr_mult=regime_cfg_raw.get("take_profit_atr_mult", 4.0),
        stop_atr_mult=regime_cfg_raw.get("stop_atr_mult", 2.0),
        max_holding_bars=regime_cfg_raw.get("max_holding_bars", 30),
        rsi_low=regime_cfg_raw.get("rsi_low", 35.0),
        rsi_high=regime_cfg_raw.get("rsi_high", 65.0),
        adx_min=regime_cfg_raw.get("adx_min", 18.0),
        commission_pct=regime_cfg_raw.get("commission_pct", 0.001),
        slippage_pct=regime_cfg_raw.get("slippage_pct", 0.001),
    )

    engine.regime_strategy = RegimeAdaptiveStrategy(cfg)
    engine.regime_cfg = cfg
    engine.regime_positions = {s: None for s in engine.symbols}

    logger.info(f"RegimeAdaptiveStrategy 已注入 | "
                f"risk={cfg.risk_per_trade} | trail={cfg.trailing_atr_mult}x")


def _process_symbol_regime(engine: TradingEngine, symbol: str,
                           prices: dict, nav: float):
    """
    用 RegimeAdaptiveStrategy 处理单标的信号（替代原版 _process_symbol）。

    数据链路和 backtest_runner.py 完全一致：
      FeatureEngine.compute_all() → add_regime_column() → should_enter/check_exit
    """
    df = engine.storage.get_klines(symbol, "1h")
    if len(df) < 500:
        logger.debug(f"{symbol} 数据不足 ({len(df)} 条)，跳过")
        return

    features = engine.feature_engine.compute_all(df)
    if features.empty:
        return
    features = add_regime_column(features)

    if len(features) < 2:
        return
    row = features.iloc[-1]
    prev_row = features.iloc[-2]

    close = row.get("close", 0)
    regime = row.get("regime", REGIME_RANGE)
    current_price = prices.get(symbol, close)

    logger.info(
        f"[{symbol}] regime={regime} | close={close:.2f} | "
        f"rsi={row.get('rsi_14', 0):.1f} | adx={row.get('adx_14', 0):.1f}")

    engine.storage.save_signal({
        "symbol": symbol,
        "timestamp": int(time.time() * 1000),
        "strategy": "regime_adaptive",
        "direction": regime,
        "strength": row.get("adx_14", 0) / 100.0,
        "confidence": 0.0,
        "model_version": "regime_v2",
    })

    position = engine.regime_positions.get(symbol)
    strategy = engine.regime_strategy
    cfg = engine.regime_cfg

    # ── 持仓中 → 检查出场 ──
    if position is not None:
        bar_count = position.get("bar_count", 0) + 1
        position["bar_count"] = bar_count
        position["highest_since_entry"] = max(
            position["highest_since_entry"], current_price)

        # 减仓
        if strategy.check_partial_exit(row, position):
            sell_qty = position["original_qty"] * cfg.partial_exit_pct
            if sell_qty > 0 and sell_qty < position["qty"]:
                result = engine.executor.execute(
                    symbol, "SELL", sell_qty, current_price,
                    strategy="regime_adaptive",
                    algo=engine.config.execution.algo)
                if result:
                    position["qty"] -= sell_qty
                    position["partial_done"] = True
                    logger.info(f"[{symbol}] 减仓 {sell_qty:.6f} @ {current_price:.2f}")

        # 出场
        should_exit, reason = strategy.check_exit(row, position, bar_count)
        if should_exit:
            qty_to_sell = position["qty"]
            ok, risk_reason = engine.risk_mgr.pre_trade_check(
                symbol, "SELL", qty_to_sell, current_price, nav, prices)

            if ok:
                result = engine.executor.execute(
                    symbol, "SELL", qty_to_sell, current_price,
                    strategy="regime_adaptive",
                    algo=engine.config.execution.algo)
                if result:
                    pnl_pct = current_price / position["entry_price"] - 1
                    icon = "🟢" if pnl_pct > 0 else "🔴"
                    logger.info(
                        f"  {icon} [{symbol}] 平仓 @ {current_price:.2f} | "
                        f"入={position['entry_price']:.2f} | "
                        f"PnL={pnl_pct:+.1%} | {bar_count}bars | {reason}")
                    engine.regime_positions[symbol] = None
                    engine.risk_mgr.record_order()
            else:
                logger.warning(f"[{symbol}] 平仓风控拒绝: {risk_reason}")
            return

    # ── 无持仓 → 检查入场 ──
    if position is None and strategy.should_enter(row, prev_row):
        natr = row.get("natr_20", 0)
        if pd.isna(natr) or natr <= 0:
            return

        qty, stop_loss = strategy.calc_position(nav, current_price, natr, regime)
        if qty <= 0:
            return

        ok, risk_reason = engine.risk_mgr.pre_trade_check(
            symbol, "BUY", qty, current_price, nav, prices)

        if ok:
            result = engine.executor.execute(
                symbol, "BUY", qty, current_price,
                strategy="regime_adaptive",
                algo=engine.config.execution.algo)
            if result:
                atr_abs = natr * current_price
                engine.regime_positions[symbol] = {
                    "qty": qty,
                    "original_qty": qty,
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "highest_since_entry": current_price,
                    "atr_at_entry": atr_abs,
                    "entry_time": time.strftime("%Y-%m-%d %H:%M"),
                    "regime": regime,
                    "partial_done": False,
                    "bar_count": 0,
                }
                logger.info(
                    f"  🔵 [{symbol}] 开仓 @ {current_price:.2f} | "
                    f"qty={qty:.6f} | {qty * current_price:.1f} USDT | "
                    f"止损={stop_loss:.2f} | {regime}")
                engine.risk_mgr.record_order()
        else:
            logger.info(f"[{symbol}] 开仓风控拒绝: {risk_reason}")


# ═══════════════════════════════════════════════════════════════
# CLI 命令
# ═══════════════════════════════════════════════════════════════

def cmd_run(args):
    """运行交易引擎（使用 Regime 策略）"""
    config = load_config(args.config)
    simulate = not args.live
    engine = TradingEngine(config, simulate=simulate)

    # 注入 Regime 策略
    _init_regime_on_engine(engine)

    # 替换 _process_symbol 为 regime 版本
    import types
    engine._process_symbol = types.MethodType(
        lambda self, sym, prices, nav: _process_symbol_regime(self, sym, prices, nav),
        engine)

    engine.warmup(skip_history=args.skip_data, skip_train=args.skip_train)

    if args.once:
        engine.run_cycle()
        engine.shutdown()
    else:
        engine.run(interval_seconds=args.interval)


def cmd_sync_data(args):
    """仅同步历史数据"""
    config = load_config(args.config)
    client = BinanceClient(config)
    storage = Storage(config.get_nested("data.database.path", "data/quant.db"))
    downloader = HistoryDownloader(config, client, storage)
    downloader.sync_all(max_workers=3)


def cmd_validate(args):
    """数据质量检查"""
    config = load_config(args.config)
    storage = Storage(config.get_nested("data.database.path", "data/quant.db"))
    client = BinanceClient(config)
    downloader = HistoryDownloader(config, client, storage)

    for symbol in config.get_symbols():
        for tf in config.get_timeframes():
            report = downloader.validate_data(symbol, tf["interval"])
            status_icon = "✅" if report.get("status") == "ok" else "⚠️"
            logger.info(f"{status_icon} {symbol}/{tf['interval']}: {report}")


def cmd_train(args):
    """仅训练 ML 模型"""
    config = load_config(args.config)
    storage = Storage(config.get_nested("data.database.path", "data/quant.db"))
    feature_engine = FeatureEngine(
        windows=config.get_nested("features.lookback_windows",
                                   [5, 10, 20, 60, 120, 240, 480]))
    alpha_model = AlphaModel(config, storage)

    for symbol in config.get_symbols():
        df = storage.get_klines(symbol, "1h")
        if len(df) < 1000:
            logger.warning(f"{symbol}: 数据不足 ({len(df)} 条)，跳过")
            continue

        logger.info(f"训练 {symbol}，{len(df)} 条K线...")
        features = feature_engine.compute_all(
            df, include_target=True, target_periods=[1, 4, 24])
        features = feature_engine.preprocess(features, method="zscore")

        report = alpha_model.train(features, target_col="target_dir_1")
        logger.info(f"完成: {report['model_id']}")
        for k, v in report["avg_metrics"].items():
            logger.info(f"  {k}: {v:.4f}")
        break


def cmd_backtest(args):
    """
    原版 ML 滚动窗口回测（保留兼容）

    如需使用 Regime 策略回测，请用 --backtest-v2
    """
    config = load_config(args.config)
    storage = Storage(config.get_nested("data.database.path", "data/quant.db"))
    db_path = config.get_nested("data.database.path", "data/quant.db")
    windows = config.get_nested("features.lookback_windows",
                                 [5, 10, 20, 60, 120, 240, 480])
    feature_engine = FeatureEngine(windows=windows)
    lookback = max(windows) + 20
    snapshot_path = _resolve_bt_snapshot_path(args, "v1_ml")
    snapshot_blocks: list[str] = []

    for symbol in config.get_symbols():
        df = storage.get_klines(symbol, "1h")
        if len(df) < 2000:
            logger.warning(f"{symbol}: 数据不足 ({len(df)} 条)，需要至少 2000 条")
            continue

        logger.info(f"回测 {symbol}，共 {len(df)} 条K线")

        train_size = 1500
        test_size = 200
        step = 200

        equity = [10000.0]
        trades_pnl = []
        position = 0.0
        entry_price = 0.0
        window_count = 0
        trade_count = 0

        for start in range(0, len(df) - train_size - test_size, step):
            window_count += 1
            train_df = df.iloc[start:start + train_size]
            test_start = start + train_size
            test_end = test_start + test_size

            train_feat = feature_engine.compute_all(
                train_df, include_target=True, target_periods=[1])
            if train_feat.empty:
                continue
            train_feat = feature_engine.preprocess(train_feat, method="zscore")

            alpha = AlphaModel(config, storage)
            try:
                alpha.train(train_feat, target_col="target_dir_1")
            except Exception as e:
                logger.debug(f"窗口 {window_count}: 训练失败 ({e})，跳过")
                continue

            context_start = max(0, test_start - lookback)
            context_df = df.iloc[context_start:test_end].copy()
            test_feat = feature_engine.compute_all(context_df)
            if test_feat.empty:
                continue
            test_feat = feature_engine.preprocess(test_feat, method="zscore")

            n_lookback = test_start - context_start
            if len(test_feat) > n_lookback:
                test_feat = test_feat.iloc[n_lookback:]
            else:
                continue

            try:
                preds = alpha.predict(test_feat)
            except Exception:
                continue

            if preds.empty:
                continue

            test_df = df.iloc[test_start:test_end]

            for i in range(len(preds)):
                sig = preds.iloc[i]["signal"]
                price = float(test_df.iloc[min(i, len(test_df) - 1)]["close"])

                if sig in ("BUY", "STRONG_BUY") and position == 0:
                    qty = equity[-1] * 0.95 / price
                    position = qty
                    entry_price = price * 1.001
                elif sig in ("SELL", "STRONG_SELL") and position > 0:
                    exit_price = price * 0.999
                    pnl = position * (exit_price - entry_price)
                    equity.append(equity[-1] + pnl)
                    trades_pnl.append(pnl / (position * entry_price))
                    position = 0
                    trade_count += 1
                else:
                    if position > 0 and i > 0:
                        prev_price = float(test_df.iloc[i - 1]["close"])
                        mark_pnl = position * (price - prev_price)
                        equity.append(equity[-1] + mark_pnl)
                    else:
                        equity.append(equity[-1])

            logger.info(f"窗口 {window_count}: "
                        f"训练={len(train_feat)}样本 测试={len(preds)}信号 "
                        f"累计交易={trade_count}")

        if position > 0:
            last_price = float(df.iloc[-1]["close"])
            exit_price = last_price * 0.999
            pnl = position * (exit_price - entry_price)
            equity.append(equity[-1] + pnl)
            trades_pnl.append(pnl / (position * entry_price))
            position = 0

        eq = np.array(equity)
        tp = np.array(trades_pnl) if trades_pnl else np.array([0.0])

        if len(eq) < 2:
            logger.warning(f"{symbol}: 没有产生任何交易")
            continue

        report = generate_report(eq, tp, frequency="1h")
        print(format_report(report))
        snapshot_blocks.append(
            f"{'=' * 60}\nSYMBOL: {symbol}\n{'=' * 60}\n" + format_report(report)
        )
        logger.info(f"{symbol} 回测完成: 共 {window_count} 个窗口, {trade_count} 笔交易")

    if snapshot_blocks:
        header = "\n".join(
            [
                "=" * 70,
                "BACKTEST SNAPSHOT v1 (ML rolling windows)",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Database: {db_path}",
                "=" * 70,
                "",
            ]
        )
        body = "\n\n".join(snapshot_blocks)
        with open(snapshot_path, "w", encoding="utf-8") as f:
            f.write(header + body + "\n")
        logger.info(f"快照已保存: {snapshot_path}")


def cmd_backtest_v2(args):
    """
    政体自适应回测 v2

    使用 FeatureEngine + RegimeAdaptiveStrategy，
    和 cmd_run 的模拟/实盘交易走完全相同的策略逻辑。
    """
    from backtest_runner import BacktestEngineV2, write_snapshot

    config = load_config(args.config)
    db_path = config.get_nested("data.database.path", "data/quant.db")

    if not Path(db_path).exists():
        logger.error(f"数据库不存在: {db_path}")
        logger.error("请先运行: python main.py --sync-data")
        return

    cfg = RegimeStrategyConfig()

    # 如果命令行指定了参数
    if hasattr(args, 'bt_risk') and args.bt_risk is not None:
        cfg.risk_per_trade = args.bt_risk
        cfg.risk_per_trade_weak = args.bt_risk / 2
    if hasattr(args, 'bt_trail') and args.bt_trail is not None:
        cfg.trailing_atr_mult = args.bt_trail

    engine = BacktestEngineV2(
        db_path=db_path,
        initial_capital=args.bt_capital,
        start_date=args.bt_start,
        end_date=args.bt_end,
        cfg=cfg,
        enable_regime=not args.no_regime,
    )

    report = engine.run()
    if report:
        mode = "v2_no_regime" if args.no_regime else "v2_regime"
        snapshot_path = _resolve_bt_snapshot_path(args, mode)
        write_snapshot(report, snapshot_path, db_path,
                       args.bt_start, args.bt_end)


def cmd_validate_strategy(args):
    """
    政体自适应策略验证回测（原版保留）

    用最近 2 个月作为测试期，和原版 EMA 交叉策略做对比。
    """
    from datetime import datetime, timezone

    config = load_config(args.config)
    storage = Storage(config.get_nested("data.database.path", "data/quant.db"))
    windows = config.get_nested("features.lookback_windows",
                                 [5, 10, 20, 60, 120, 240, 480])
    feature_engine = FeatureEngine(windows=windows)

    INITIAL_CAPITAL = 500.0
    TEST_START = "2026-02-12"
    TEST_END   = "2026-04-12"
    INTERVAL   = "4h"

    test_start_ms = int(datetime.strptime(TEST_START, "%Y-%m-%d")
                        .replace(tzinfo=timezone.utc).timestamp() * 1000)
    test_end_ms   = int(datetime.strptime(TEST_END, "%Y-%m-%d")
                        .replace(tzinfo=timezone.utc).timestamp() * 1000)

    strategy = RegimeAdaptiveStrategy()
    cfg = strategy.cfg
    symbols = config.get_symbols()

    for symbol in symbols:
        logger.info(f"{'=' * 60}")
        logger.info(f"  政体策略验证: {symbol}")
        logger.info(f"  测试期: {TEST_START} ~ {TEST_END} | 本金: {INITIAL_CAPITAL} USDT")
        logger.info(f"{'=' * 60}")

        df = storage.get_klines(symbol, INTERVAL)
        if len(df) < 1000:
            logger.warning(f"{symbol}: 4h 数据不足 ({len(df)} 条)")
            continue

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)

        features = feature_engine.compute_all(df)
        if features.empty:
            continue

        for col in ["open", "high", "low", "open_time"]:
            if col in df.columns:
                features[col] = df[col].values

        features = add_regime_column(features)

        test_mask = (features["open_time"] >= test_start_ms) & \
                    (features["open_time"] <= test_end_ms)
        test_indices = features[test_mask].index.tolist()

        if not test_indices:
            logger.warning(f"{symbol}: 测试期无数据")
            continue

        test_start_idx = test_indices[0]
        test_end_idx   = test_indices[-1]
        n_test_bars    = len(test_indices)
        test_start_price = features.loc[test_start_idx, "close"]
        test_end_price   = features.loc[test_end_idx, "close"]
        buy_hold_ret     = test_end_price / test_start_price - 1

        logger.info(f"测试期: {n_test_bars} bars | "
                    f"Buy&Hold {buy_hold_ret:+.2%}")

        # ── 新策略回测 ──
        cash = INITIAL_CAPITAL
        position = None
        trades = []
        equity_list = [INITIAL_CAPITAL]

        for idx in test_indices:
            if idx == 0:
                continue
            row      = features.iloc[idx]
            prev_row = features.iloc[idx - 1]
            close    = row["close"]
            high     = row.get("high", close)
            ts       = pd.Timestamp(row["open_time"], unit="ms", tz="UTC") \
                         .strftime("%Y-%m-%d %H:%M")

            equity = cash + (position["qty"] * close if position else 0)
            equity_list.append(equity)

            if position is not None:
                position["highest_since_entry"] = max(
                    position["highest_since_entry"], high)
                bar_count = idx - position["entry_bar"]

                if strategy.check_partial_exit(row, position):
                    sell_qty = position["qty"] * cfg.partial_exit_pct
                    sell_price = close * (1 - cfg.slippage_pct)
                    cash += sell_qty * sell_price * (1 - cfg.commission_pct)
                    position["qty"] -= sell_qty
                    position["partial_done"] = True

                should_exit, reason = strategy.check_exit(row, position, bar_count)
                if should_exit:
                    exit_price = close * (1 - cfg.slippage_pct)
                    proceeds = position["qty"] * exit_price * (1 - cfg.commission_pct)
                    pnl = proceeds - position["qty"] * position["entry_price"] * \
                          (1 + cfg.commission_pct)
                    pnl_pct = exit_price / position["entry_price"] - 1
                    cash += proceeds
                    trades.append({
                        "symbol": symbol, "entry_time": position["entry_time"],
                        "exit_time": ts, "entry_price": position["entry_price"],
                        "exit_price": exit_price, "pnl": round(pnl, 2),
                        "pnl_pct": pnl_pct, "holding_bars": bar_count,
                        "reason": reason, "regime": position["regime"],
                    })
                    position = None

            if position is None and strategy.should_enter(row, prev_row):
                regime = row.get("regime", "RANGE")
                natr   = row.get("natr_20", 0)
                if pd.isna(natr) or natr <= 0:
                    continue
                entry_price = close * (1 + cfg.slippage_pct)
                qty, stop_loss = strategy.calc_position(cash, entry_price, natr, regime)
                if qty > 0:
                    cost = qty * entry_price * (1 + cfg.commission_pct)
                    if cost <= cash:
                        cash -= cost
                        atr_abs = natr * entry_price
                        position = {
                            "qty": qty, "original_qty": qty,
                            "entry_price": entry_price, "stop_loss": stop_loss,
                            "highest_since_entry": high, "atr_at_entry": atr_abs,
                            "entry_bar": idx, "entry_time": ts,
                            "regime": regime, "partial_done": False,
                        }

        if position is not None:
            ep = features.loc[test_end_idx, "close"] * (1 - cfg.slippage_pct)
            pnl = position["qty"] * (ep - position["entry_price"]) * (1 - cfg.commission_pct)
            cash += position["qty"] * ep * (1 - cfg.commission_pct)
            trades.append({
                "symbol": symbol, "entry_time": position["entry_time"],
                "exit_time": "END", "entry_price": position["entry_price"],
                "exit_price": ep, "pnl": round(pnl, 2),
                "pnl_pct": ep / position["entry_price"] - 1,
                "holding_bars": test_end_idx - position["entry_bar"],
                "reason": "backtest_end", "regime": position["regime"],
            })

        # ── 对照组：原版 EMA 交叉 ──
        orig_cash = INITIAL_CAPITAL
        orig_pos = None
        orig_trades_pnl = []

        for idx in test_indices:
            if idx == 0:
                continue
            row = features.iloc[idx]
            prev_row = features.iloc[idx - 1]
            close = row["close"]
            high = row.get("high", close)
            natr = row.get("natr_20", 0)
            atr = natr * close if natr > 0 else 0
            ef = row.get("ema_20", close)
            es = row.get("ema_50", close)
            pef = prev_row.get("ema_20", close)
            pes = prev_row.get("ema_50", close)
            cn = 1 if ef > es else -1
            cp = 1 if pef > pes else -1

            if orig_pos:
                orig_pos["h"] = max(orig_pos["h"], high)
                ts_val = orig_pos["h"] - 2.5 * atr if atr > 0 else orig_pos["sl"]
                ts_val = max(ts_val, orig_pos["sl"])
                if close <= ts_val or (cn == -1 and cp == 1):
                    ep = close * 0.999
                    orig_cash += orig_pos["q"] * ep
                    orig_trades_pnl.append(orig_pos["q"] * (ep - orig_pos["e"]))
                    orig_pos = None

            if orig_pos is None and cn == 1 and cp != 1 and atr > 0:
                sd = 2.0 * atr
                q = (orig_cash * 0.02) / sd if sd > 0 else 0
                c = q * close * 1.001
                if q > 0 and c <= orig_cash * 0.6 and c >= 12:
                    orig_cash -= c
                    orig_pos = {"q": q, "e": close * 1.001,
                                "sl": close - sd, "h": high}

        if orig_pos:
            ep = features.loc[test_end_idx, "close"] * 0.999
            orig_cash += orig_pos["q"] * ep
            orig_trades_pnl.append(orig_pos["q"] * (ep - orig_pos["e"]))

        orig_return = (orig_cash - INITIAL_CAPITAL) / INITIAL_CAPITAL

        # ── 报告 ──
        final_equity = cash
        new_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
        eq = np.array(equity_list)
        peak = np.maximum.accumulate(eq)
        max_dd = float(((peak - eq) / peak.clip(min=1e-10)).max())
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / len(pnls) if pnls else 0
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else (
            float("inf") if wins else 0)

        print()
        print("=" * 65)
        print("              政体策略验证报告")
        print(f"         {symbol} | {TEST_START} ~ {TEST_END}")
        print("=" * 65)
        icon = "🟢" if new_return > 0 else "🔴"
        print(f"  {icon} 新策略收益:       {new_return:>+10.2%}  "
              f"({final_equity - INITIAL_CAPITAL:+.2f} USDT)")
        print(f"     原版策略收益:     {orig_return:>+10.2%}  "
              f"({orig_cash - INITIAL_CAPITAL:+.2f} USDT)")
        print(f"     Buy & Hold:       {buy_hold_ret:>+10.2%}")
        dvo = (new_return - orig_return) * 100
        dvb = (new_return - buy_hold_ret) * 100
        print(f"     vs 原版:          {dvo:>+10.2f}%  {'✅' if dvo > 0 else '❌'}")
        print(f"     vs Buy&Hold:      {dvb:>+10.2f}%  {'✅' if dvb > 0 else '❌'}")
        print(f"     最大回撤:         {max_dd:>10.2%}")
        print("-" * 65)
        print(f"  📊 交易统计:")
        print(f"     交易数 (新/原):   {len(trades):>4d} / {len(orig_trades_pnl)}")
        print(f"     胜率:             {win_rate:>10.1%}")
        print(f"     盈亏比:           {pf:>10.3f}")
        if pnls:
            print(f"     最大盈利:         {max(pnls):>+10.2f}")
            print(f"     最大亏损:         {min(pnls):>+10.2f}")

        if trades:
            print("-" * 65)
            print(f"  📋 交易明细:")
            for i, t in enumerate(trades, 1):
                ic = "🟢" if t["pnl"] > 0 else "🔴"
                print(f"   {ic}{i:2d} | {t['entry_time']} → {t['exit_time']} | "
                      f"入={t['entry_price']:.2f} 出={t['exit_price']:.2f} | "
                      f"PnL={t['pnl']:+.2f} ({t['pnl_pct']:+.1%}) | "
                      f"{t['holding_bars']}bars | {t['reason']} | {t['regime']}")

        print("=" * 65)

        monthly = {}
        for t in trades:
            m = t["exit_time"][:7]
            monthly[m] = monthly.get(m, 0) + t["pnl"]
        if monthly:
            print(f"\n  📅 月度 PnL:")
            for m, p in sorted(monthly.items()):
                mi = "🟢" if p > 0 else "🔴"
                print(f"     {mi} {m}: {p:+.2f} USDT")
        print()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="量化交易系统 v2.1")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--live", action="store_true", help="实盘模式")
    parser.add_argument("--once", action="store_true", help="只运行一轮")
    parser.add_argument("--interval", type=int, default=3600, help="循环间隔(秒)")
    parser.add_argument("--skip-data", action="store_true", help="跳过数据同步")
    parser.add_argument("--skip-train", action="store_true", help="跳过模型训练")
    parser.add_argument("--sync-data", action="store_true", help="仅同步数据")
    parser.add_argument("--validate", action="store_true", help="数据质量检查")
    parser.add_argument("--train", action="store_true", help="仅训练模型")
    parser.add_argument("--backtest", action="store_true", help="原版 ML 回测")
    parser.add_argument("--backtest-v2", action="store_true",
                        help="政体自适应回测 v2（推荐）")
    parser.add_argument("--validate-strategy", action="store_true",
                        help="政体策略验证回测 (2个月)")

    # backtest-v2 专用参数
    parser.add_argument("--no-regime", action="store_true",
                        help="关闭 regime 过滤（对照组）")
    parser.add_argument("--bt-capital", type=float, default=10000.0,
                        help="回测初始资金")
    parser.add_argument("--bt-start", type=str, default=None,
                        help="回测起始日期 YYYY-MM-DD")
    parser.add_argument("--bt-end", type=str, default=None,
                        help="回测结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--bt-snapshot",
        type=str,
        default=None,
        help="回测快照输出路径；省略时按类型自动命名："
        "v2+regime→backtest_v2_regime_snapshot.txt，"
        "v2+--no-regime→backtest_v2_no_regime_snapshot.txt，"
        "--backtest→backtest_v1_ml_snapshot.txt",
    )
    parser.add_argument("--bt-risk", type=float, default=None,
                        help="单笔风险比例")
    parser.add_argument("--bt-trail", type=float, default=None,
                        help="移动止损 ATR 倍数")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Crypto Quant Engine v2.1")
    logger.info("=" * 60)

    if args.sync_data:
        cmd_sync_data(args)
    elif args.validate:
        cmd_validate(args)
    elif args.train:
        cmd_train(args)
    elif args.backtest:
        cmd_backtest(args)
    elif args.backtest_v2:
        cmd_backtest_v2(args)
    elif args.validate_strategy:
        cmd_validate_strategy(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
