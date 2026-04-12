"""
main.py 补丁 — 添加 cmd_validate_strategy

使用方法：
  1. 将 cmd_validate_strategy() 函数复制到 main.py 中（放在 cmd_backtest 后面）
  2. 在 argparse 部分加一行:
       parser.add_argument("--validate-strategy", action="store_true", help="政体策略验证回测")
  3. 在 main() 的 if/elif 链中加一个分支:
       elif args.validate_strategy:
           cmd_validate_strategy(args)
  4. 在 main.py 顶部加一行 import:
       from alpha.regime_strategy import RegimeAdaptiveStrategy, RegimeStrategyConfig, add_regime_column

运行:
  python main.py --validate-strategy
  python main.py --validate-strategy --config path/to/cfg.yaml
"""

# ============================================================
# 以下为需要添加到 main.py 的完整函数
# ============================================================

# --- 在 main.py 顶部添加 import ---
# from alpha.regime_strategy import (
#     RegimeAdaptiveStrategy, RegimeStrategyConfig, add_regime_column,
#     REGIME_BULL_TREND, REGIME_BULL_WEAK, REGIME_BEAR_TREND, REGIME_BEAR_WEAK,
# )

# --- 以下函数复制到 main.py（放在 cmd_backtest 之后） ---

def cmd_validate_strategy(args):
    """
    政体自适应策略验证回测

    用最近 2 个月 (2026-02-12 ~ 2026-04-12) 作为测试期，
    之前的全部数据作为特征计算的上下文（lookback）。
    同时运行原版 EMA 交叉策略作对照，量化「政体过滤」带来的改善。
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

    # 使用 500 USDT 验证
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

        # 类型转换
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)

        start_ts = pd.Timestamp(df["open_time"].iloc[0], unit="ms", tz="UTC")
        end_ts   = pd.Timestamp(df["open_time"].iloc[-1], unit="ms", tz="UTC")
        logger.info(f"数据: {len(df)} 条 4h K 线 ({start_ts.date()} ~ {end_ts.date()})")

        # ---- 计算特征 ----
        features = feature_engine.compute_all(df)
        if features.empty:
            logger.warning(f"{symbol}: 特征计算失败")
            continue

        # 从原始 df 补回 open/high/low（FeatureEngine 保留了 close 但没保留这些）
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
        position = None  # dict: qty, original_qty, entry_price, stop_loss, ...
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
            ts       = pd.Timestamp(row["open_time"], unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")

            # 当前权益
            equity = cash + (position["qty"] * close if position else 0)
            peak_equity = max(peak_equity, equity)
            equity_list.append(equity)

            # ---- 持仓中：检查出场 ----
            if position is not None:
                position["highest_since_entry"] = max(position["highest_since_entry"], high)
                bar_count = idx - position["entry_bar"]

                # 减仓检查
                if strategy.check_partial_exit(row, position):
                    sell_qty = position["qty"] * cfg.partial_exit_pct
                    sell_price = close * (1 - cfg.slippage_pct)
                    cash += sell_qty * sell_price * (1 - cfg.commission_pct)
                    position["qty"] -= sell_qty
                    position["partial_done"] = True
                    logger.info(f"  📉 {ts} | 减仓 {cfg.partial_exit_pct:.0%} "
                                f"@ {sell_price:.2f}")

                # 全部出场检查
                should_exit, reason = strategy.check_exit(row, position, bar_count)
                if should_exit:
                    exit_price = close * (1 - cfg.slippage_pct)
                    commission = position["qty"] * exit_price * cfg.commission_pct
                    proceeds = position["qty"] * exit_price - commission

                    # PnL 计算（含已减仓部分的近似）
                    total_cost = position["original_qty"] * position["entry_price"] * \
                                 (1 + cfg.commission_pct)
                    total_proceeds = proceeds
                    if position.get("partial_done"):
                        partial_qty = position["original_qty"] * cfg.partial_exit_pct
                        # 减仓那部分已经计入 cash 了，这里只算剩余部分
                        total_cost = position["qty"] * position["entry_price"] * \
                                     (1 + cfg.commission_pct)

                    pnl = proceeds - position["qty"] * position["entry_price"] * \
                          (1 + cfg.commission_pct)
                    pnl_pct = (exit_price / position["entry_price"] - 1)

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

            # ---- 无持仓：检查入场 ----
            if position is None and strategy.should_enter(row, prev_row):
                regime  = row.get("regime", "RANGE")
                natr    = row.get("natr_20", 0)

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
                            "qty": qty,
                            "original_qty": qty,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "highest_since_entry": high,
                            "atr_at_entry": atr_abs,
                            "entry_bar": idx,
                            "entry_time": ts,
                            "regime": regime,
                            "partial_done": False,
                        }
                        pos_val = qty * entry_price
                        logger.info(
                            f"  🔵 {ts} | 开仓 @ {entry_price:.2f} | "
                            f"qty={qty:.6f} | {pos_val:.2f} USDT "
                            f"({pos_val/equity*100:.1f}%) | "
                            f"止损={stop_loss:.2f} | {regime}")

        # 回测结束平仓
        if position is not None:
            last_close = features.loc[test_end_idx, "close"]
            exit_price = last_close * (1 - cfg.slippage_pct)
            pnl = position["qty"] * (exit_price - position["entry_price"]) * \
                  (1 - cfg.commission_pct)
            cash += position["qty"] * exit_price * (1 - cfg.commission_pct)
            bar_count = test_end_idx - position["entry_bar"]
            trades.append({
                "symbol": symbol, "entry_time": position["entry_time"],
                "exit_time": "END",
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "pnl_pct": (exit_price / position["entry_price"] - 1),
                "holding_bars": bar_count, "reason": "backtest_end",
                "regime": position["regime"],
            })
            position = None

        # ============================================================
        # 对照组：原版 EMA 交叉（无政体过滤）
        # ============================================================
        orig_cash = INITIAL_CAPITAL
        orig_pos  = None
        orig_trades = []

        for idx in test_indices:
            if idx == 0:
                continue
            row = features.iloc[idx]
            prev_row = features.iloc[idx - 1]
            close = row["close"]
            high  = row.get("high", close)
            natr  = row.get("natr_20", 0)
            atr   = natr * close if natr > 0 else 0

            # 用 ema_20 vs ema_50 模拟原版交叉逻辑
            ema_fast = row.get("ema_20", close)
            ema_slow = row.get("ema_50", close)
            prev_ema_fast = prev_row.get("ema_20", close)
            prev_ema_slow = prev_row.get("ema_50", close)

            cross_now  = 1 if ema_fast > ema_slow else -1
            cross_prev = 1 if prev_ema_fast > prev_ema_slow else -1

            if orig_pos:
                orig_pos["highest"] = max(orig_pos["highest"], high)
                trailing = orig_pos["highest"] - 2.5 * atr if atr > 0 else orig_pos["stop"]
                trailing = max(trailing, orig_pos["stop"])

                exit_now = False
                exit_reason = ""
                if close <= trailing:
                    exit_now, exit_reason = True, "trailing_stop"
                elif cross_now == -1 and cross_prev == 1:
                    exit_now, exit_reason = True, "ma_cross"

                if exit_now:
                    ep = close * 0.999
                    pnl = orig_pos["qty"] * (ep - orig_pos["entry"])
                    orig_cash += orig_pos["qty"] * ep
                    orig_trades.append(pnl)
                    orig_pos = None

            if orig_pos is None:
                golden_cross = cross_now == 1 and cross_prev != 1
                if golden_cross and atr > 0:
                    stop_d = 2.0 * atr
                    risk = orig_cash * 0.02
                    qty = risk / stop_d if stop_d > 0 else 0
                    cost = qty * close * 1.001
                    if qty > 0 and cost <= orig_cash * 0.6 and cost >= 12:
                        orig_cash -= cost
                        orig_pos = {
                            "qty": qty, "entry": close * 1.001,
                            "stop": close - stop_d, "highest": high,
                        }

        if orig_pos:
            ep = features.loc[test_end_idx, "close"] * 0.999
            orig_cash += orig_pos["qty"] * ep
            orig_trades.append(orig_pos["qty"] * (ep - orig_pos["entry"]))

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
        print(f"  {icon} 新策略收益:       {new_return:>+10.2%}  ({final_equity - INITIAL_CAPITAL:+.2f} USDT)")
        print(f"     原版策略收益:     {orig_return:>+10.2%}  ({orig_cash - INITIAL_CAPITAL:+.2f} USDT)")
        print(f"     Buy & Hold:       {buy_hold_ret:>+10.2%}")
        diff_vs_orig = (new_return - orig_return) * 100
        diff_vs_bh   = (new_return - buy_hold_ret) * 100
        print(f"     vs 原版:          {diff_vs_orig:>+10.2f}%  {'✅' if diff_vs_orig > 0 else '❌'}")
        print(f"     vs Buy&Hold:      {diff_vs_bh:>+10.2f}%  {'✅' if diff_vs_bh > 0 else '❌'}")
        print(f"     最大回撤:         {max_dd:>10.2%}")
        print("-" * 65)
        print(f"  📊 交易统计:")
        print(f"     交易数 (新/原):   {len(trades):>4d} / {len(orig_trades)}")
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
            logger.info("测试期无交易 — 政体过滤器阻止了熊市做多")
            logger.info("这在下跌行情中是正确的策略行为：不亏就是赢")

        # 月度 PnL
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
