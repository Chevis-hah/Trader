"""
加密货币量化交易系统 v2.0
工业级架构 · 全量数据 · ML Alpha · 智能执行 · 三层风控

用法:
    python main.py                          # 模拟交易
    python main.py --live                   # 实盘交易
    python main.py --backtest               # 回测
    python main.py --validate-strategy      # 政体策略验证回测
    python main.py --train                  # 仅训练模型
    python main.py --sync-data              # 仅同步数据
    python main.py --validate               # 数据质量检查
    python main.py --config path/to/cfg.yaml # 指定配置文件
"""
import argparse
import sys
from pathlib import Path

from config.loader import load_config
from core.engine import TradingEngine
from data.client import BinanceClient
from data.storage import Storage
from data.historical import HistoryDownloader
from data.features import FeatureEngine
from alpha.ml_model import AlphaModel
from utils.logger import get_logger
from utils.metrics import generate_report, format_report

import numpy as np
import pandas as pd

logger = get_logger("main")

def cmd_run(args):
    """运行交易引擎"""
    config = load_config(args.config)
    simulate = not args.live
    engine = TradingEngine(config, simulate=simulate)
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
    """仅训练模型"""
    config = load_config(args.config)
    storage = Storage(config.get_nested("data.database.path", "data/quant.db"))
    feature_engine = FeatureEngine(
        windows=config.get_nested("features.lookback_windows", [5, 10, 20, 60, 120, 240, 480]))
    alpha_model = AlphaModel(config, storage)

    for symbol in config.get_symbols():
        df = storage.get_klines(symbol, "1h")
        if len(df) < 1000:
            logger.warning(f"{symbol}: 数据不足 ({len(df)} 条)，跳过")
            continue

        logger.info(f"训练 {symbol}，{len(df)} 条K线...")
        features = feature_engine.compute_all(df, include_target=True, target_periods=[1, 4, 24])
        features = feature_engine.preprocess(features, method="zscore")

        report = alpha_model.train(features, target_col="target_dir_1")
        logger.info(f"完成: {report['model_id']}")
        for k, v in report["avg_metrics"].items():
            logger.info(f"  {k}: {v:.4f}")
        break

def cmd_backtest(args):
    """
    滚动窗口回测

    修复要点：
    - 测试窗口需要包含足够的 lookback 数据才能计算特征
    - 从训练数据末尾取 lookback 条拼接到测试数据前面
    - 计算完特征后只取测试部分的预测结果
    """
    config = load_config(args.config)
    storage = Storage(config.get_nested("data.database.path", "data/quant.db"))
    windows = config.get_nested("features.lookback_windows", [5, 10, 20, 60, 120, 240, 480])
    feature_engine = FeatureEngine(windows=windows)

    # 特征计算需要的最小 lookback 条数
    lookback = max(windows) + 20  # 480 + 20 = 500

    for symbol in config.get_symbols():
        df = storage.get_klines(symbol, "1h")
        if len(df) < 2000:
            logger.warning(f"{symbol}: 数据不足 ({len(df)} 条)，需要至少 2000 条")
            continue

        logger.info(f"回测 {symbol}，共 {len(df)} 条K线")

        # 滚动窗口参数
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

            # ---- 训练 ----
            train_feat = feature_engine.compute_all(
                train_df, include_target=True, target_periods=[1])
            if train_feat.empty:
                logger.debug(f"窗口 {window_count}: 训练特征为空，跳过")
                continue

            train_feat = feature_engine.preprocess(train_feat, method="zscore")

            alpha = AlphaModel(config, storage)
            try:
                alpha.train(train_feat, target_col="target_dir_1")
            except Exception as e:
                logger.debug(f"窗口 {window_count}: 训练失败 ({e})，跳过")
                continue

            # ---- 测试 ----
            # 关键修复：取 lookback + test_size 条数据计算特征
            context_start = max(0, test_start - lookback)
            context_df = df.iloc[context_start:test_end].copy()

            test_feat = feature_engine.compute_all(context_df)
            if test_feat.empty:
                logger.debug(f"窗口 {window_count}: 测试特征为空，跳过")
                continue

            test_feat = feature_engine.preprocess(test_feat, method="zscore")

            # 只取测试部分（去掉 lookback 的行）
            n_lookback = test_start - context_start
            if len(test_feat) > n_lookback:
                test_feat = test_feat.iloc[n_lookback:]
            else:
                logger.debug(f"窗口 {window_count}: 测试特征不足，跳过")
                continue

            try:
                preds = alpha.predict(test_feat)
            except Exception as e:
                logger.debug(f"窗口 {window_count}: 预测失败 ({e})，跳过")
                continue

            if preds.empty:
                continue

            # ---- 模拟交易 ----
            test_df = df.iloc[test_start:test_end]

            for i in range(len(preds)):
                sig = preds.iloc[i]["signal"]

                if i < len(test_df):
                    price = float(test_df.iloc[i]["close"])
                else:
                    price = float(test_df.iloc[-1]["close"])

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

        # ---- 如果还有持仓，按最后价格平仓 ----
        if position > 0:
            last_price = float(df.iloc[-1]["close"])
            exit_price = last_price * 0.999
            pnl = position * (exit_price - entry_price)
            equity.append(equity[-1] + pnl)
            trades_pnl.append(pnl / (position * entry_price))
            position = 0

        # ---- 报告 ----
        eq = np.array(equity)
        tp = np.array(trades_pnl) if trades_pnl else np.array([0.0])

        if len(eq) < 2:
            logger.warning(f"{symbol}: 没有产生任何交易，无法生成报告")
            logger.info(f"  可能原因: 数据量不足或特征计算窗口太大")
            logger.info(f"  数据量={len(df)}，训练窗口={train_size}，"
                        f"测试窗口={test_size}，lookback={lookback}")
            continue

        report = generate_report(eq, tp, frequency="1h")
        print(format_report(report))
        logger.info(f"{symbol} 回测完成: 共 {window_count} 个窗口, {trade_count} 笔交易")


def cmd_validate_strategy(args):
    """
    政体自适应策略验证回测

    用最近 2 个月作为测试期，之前全部数据作为 lookback 上下文。
    同时运行原版 EMA 交叉策略作对照，量化「政体过滤」的改善。
    """
    from datetime import datetime, timezone
    from alpha.regime_strategy import (
        RegimeAdaptiveStrategy, RegimeStrategyConfig, add_regime_column,
        REGIME_BULL_TREND, REGIME_BULL_WEAK,
        REGIME_BEAR_TREND, REGIME_BEAR_WEAK,
    )

    config = load_config(args.config)
    storage = Storage(config.get_nested("data.database.path", "data/quant.db"))
    windows = config.get_nested("features.lookback_windows", [5, 10, 20, 60, 120, 240, 480])
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
        logger.info(f"{'='*60}")
        logger.info(f"  政体策略验证: {symbol}")
        logger.info(f"  测试期: {TEST_START} ~ {TEST_END} | 本金: {INITIAL_CAPITAL} USDT")
        logger.info(f"{'='*60}")

        # ---- 加载全部 4h 数据 ----
        df = storage.get_klines(symbol, INTERVAL)
        if len(df) < 1000:
            logger.warning(f"{symbol}: 4h 数据不足 ({len(df)} 条)，需要至少 1000 条")
            continue

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)

        start_ts = pd.Timestamp(df["open_time"].iloc[0], unit="ms", tz="UTC")
        end_ts   = pd.Timestamp(df["open_time"].iloc[-1], unit="ms", tz="UTC")
        logger.info(f"数据: {len(df)} 条 4h K 线 ({start_ts.date()} ~ {end_ts.date()})")

        # ---- 计算特征（复用 FeatureEngine） ----
        features = feature_engine.compute_all(df)
        if features.empty:
            logger.warning(f"{symbol}: 特征计算失败")
            continue

        # 补回 FeatureEngine 没保留的列
        for col in ["open", "high", "low", "open_time"]:
            if col in df.columns:
                features[col] = df[col].values

        # 添加政体标签
        features = add_regime_column(features)

        # ---- 定位测试期 ----
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
        buy_hold_ret     = (test_end_price / test_start_price - 1)

        logger.info(f"测试期: {n_test_bars} bars | "
                    f"起始价 {test_start_price:.2f} → 结束价 {test_end_price:.2f} | "
                    f"Buy&Hold {buy_hold_ret:+.2%}")

        # 政体分布
        regime_dist = features.loc[test_start_idx:test_end_idx, "regime"].value_counts()
        for r, cnt in regime_dist.items():
            logger.info(f"  政体 {r:15s}: {cnt:4d} bars ({cnt/n_test_bars*100:.1f}%)")

        # ============================================================
        # 新策略回测
        # ============================================================
        cash = INITIAL_CAPITAL
        position = None
        trades = []
        equity_list = [INITIAL_CAPITAL]
        peak_equity = INITIAL_CAPITAL

        for idx in test_indices:
            if idx == 0:
                continue

            row      = features.iloc[idx]
            prev_row = features.iloc[idx - 1]
            close    = row["close"]
            high     = row.get("high", close)
            low      = row.get("low", close)
            ts       = pd.Timestamp(row["open_time"], unit="ms", tz="UTC") \
                         .strftime("%Y-%m-%d %H:%M")

            equity = cash + (position["qty"] * close if position else 0)
            peak_equity = max(peak_equity, equity)
            equity_list.append(equity)

            # ---- 持仓中 ----
            if position is not None:
                position["highest_since_entry"] = max(
                    position["highest_since_entry"], high)
                bar_count = idx - position["entry_bar"]

                # 减仓
                if strategy.check_partial_exit(row, position):
                    sell_qty = position["qty"] * cfg.partial_exit_pct
                    sell_price = close * (1 - cfg.slippage_pct)
                    cash += sell_qty * sell_price * (1 - cfg.commission_pct)
                    position["qty"] -= sell_qty
                    position["partial_done"] = True
                    logger.info(f"  📉 {ts} | 减仓 {cfg.partial_exit_pct:.0%} "
                                f"@ {sell_price:.2f}")

                # 出场
                should_exit, reason = strategy.check_exit(row, position, bar_count)
                if should_exit:
                    exit_price = close * (1 - cfg.slippage_pct)
                    proceeds = position["qty"] * exit_price * (1 - cfg.commission_pct)
                    pnl = proceeds - position["qty"] * position["entry_price"] * \
                          (1 + cfg.commission_pct)
                    pnl_pct = exit_price / position["entry_price"] - 1
                    cash += proceeds

                    icon = "🟢" if pnl > 0 else "🔴"
                    logger.info(
                        f"  {icon} {ts} | 平仓 @ {exit_price:.2f} | "
                        f"入={position['entry_price']:.2f} | "
                        f"PnL={pnl:+.2f} ({pnl_pct:+.1%}) | "
                        f"{bar_count}bars | {reason}")

                    trades.append({
                        "symbol": symbol, "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl": round(pnl, 2), "pnl_pct": pnl_pct,
                        "holding_bars": bar_count, "reason": reason,
                        "regime": position["regime"],
                    })
                    position = None

            # ---- 入场 ----
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
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "highest_since_entry": high,
                            "atr_at_entry": atr_abs,
                            "entry_bar": idx, "entry_time": ts,
                            "regime": regime, "partial_done": False,
                        }
                        logger.info(
                            f"  🔵 {ts} | 开仓 @ {entry_price:.2f} | "
                            f"qty={qty:.6f} | {qty*entry_price:.2f} USDT | "
                            f"止损={stop_loss:.2f} | {regime}")

        # 收尾
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

        # ============================================================
        # 对照组：原版 EMA 交叉
        # ============================================================
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
                do_exit = close <= ts_val or (cn == -1 and cp == 1)
                if do_exit:
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

        # ============================================================
        # 报告
        # ============================================================
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
        in_market = sum(t["holding_bars"] for t in trades)

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
        print(f"     在场时间:         {in_market:>4d}/{n_test_bars} bars "
              f"({in_market/n_test_bars*100:.0f}%)")
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

        if len(trades) == 0:
            logger.info("测试期无交易 — 政体过滤器阻止了熊市做多，不亏就是赢")

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


def main():
    parser = argparse.ArgumentParser(description="量化交易系统 v2.0")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--live", action="store_true", help="实盘模式")
    parser.add_argument("--once", action="store_true", help="只运行一轮")
    parser.add_argument("--interval", type=int, default=3600, help="循环间隔(秒)")
    parser.add_argument("--skip-data", action="store_true", help="跳过数据同步")
    parser.add_argument("--skip-train", action="store_true", help="跳过模型训练")
    parser.add_argument("--sync-data", action="store_true", help="仅同步数据")
    parser.add_argument("--validate", action="store_true", help="数据质量检查")
    parser.add_argument("--train", action="store_true", help="仅训练模型")
    parser.add_argument("--backtest", action="store_true", help="回测")
    parser.add_argument("--validate-strategy", action="store_true",
                        help="政体策略验证回测 (2个月)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Crypto Quant Engine v2.0")
    logger.info("=" * 60)

    if args.sync_data:
        cmd_sync_data(args)
    elif args.validate:
        cmd_validate(args)
    elif args.train:
        cmd_train(args)
    elif args.backtest:
        cmd_backtest(args)
    elif args.validate_strategy:
        cmd_validate_strategy(args)
    else:
        cmd_run(args)

if __name__ == "__main__":
    main()
