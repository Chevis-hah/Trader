"""
main.py 补丁 — 集成 RegimeAdaptiveStrategy 到模拟/实盘交易

═══════════════════════════════════════════════════════════════
  目标：回测 / 模拟 / 实盘 使用 *完全相同* 的策略逻辑
═══════════════════════════════════════════════════════════════

使用方法：
  1. 将下面的代码合并到 main.py 中
  2. 核心变更：
     - TradingEngine.__init__() 新增 self.regime_strategy 初始化
     - _process_symbol() 用 RegimeAdaptiveStrategy 替代纯 ML 信号
     - 新增 --backtest-v2 命令行入口
  3. 策略逻辑链路：
     FeatureEngine.compute_all()
       → add_regime_column()
       → RegimeAdaptiveStrategy.should_enter() / check_exit()
     回测和实盘走的是同一条路径。

下面给出需要修改的三个区域的完整代码。
"""


# ═══════════════════════════════════════════════════════════════
# 区域 1：在 main.py 顶部 import 区域新增
# ═══════════════════════════════════════════════════════════════

# --- 在现有 import 块末尾添加 ---
# from alpha.regime_strategy import (
#     RegimeAdaptiveStrategy, RegimeStrategyConfig,
#     add_regime_column, classify_regime,
#     REGIME_BULL_TREND, REGIME_BULL_WEAK, REGIME_RANGE,
#     REGIME_BEAR_WEAK, REGIME_BEAR_TREND,
# )


# ═══════════════════════════════════════════════════════════════
# 区域 2：TradingEngine.__init__() 中新增策略初始化
#
# 在 self.alpha_model = ... 之后添加以下代码：
# ═══════════════════════════════════════════════════════════════

def _init_regime_strategy(self):
    """
    在 TradingEngine.__init__() 中调用。
    初始化 RegimeAdaptiveStrategy，参数从 settings.yaml 读取。

    在 __init__ 末尾加一行:
        self._init_regime_strategy()
    """
    # 从 config 读取策略参数（如果 settings.yaml 有 regime 段则用，否则默认值）
    regime_cfg_raw = {}
    if hasattr(self.config, 'strategy') and hasattr(self.config.strategy, '_data'):
        regime_cfg_raw = self.config.strategy._data.get("regime", {})
    elif hasattr(self.config, 'strategy'):
        regime_cfg_raw = getattr(self.config.strategy, 'regime', {})
        if hasattr(regime_cfg_raw, '_data'):
            regime_cfg_raw = regime_cfg_raw._data

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
    self.regime_strategy = RegimeAdaptiveStrategy(cfg)
    self.regime_cfg = cfg

    # 持仓状态（per-symbol）
    self.regime_positions: dict[str, dict | None] = {s: None for s in self.symbols}

    logger.info(f"RegimeAdaptiveStrategy 已初始化 | "
                f"risk={cfg.risk_per_trade} | trail={cfg.trailing_atr_mult}x | "
                f"tp={cfg.take_profit_atr_mult}x")


# ═══════════════════════════════════════════════════════════════
# 区域 3：替换 _process_symbol()
#
# 这是核心变更。原版用 ML 模型预测信号，新版用
# RegimeAdaptiveStrategy 的 should_enter/check_exit。
# 两者使用相同的 FeatureEngine 计算特征。
# ═══════════════════════════════════════════════════════════════

def _process_symbol_regime(self, symbol: str, prices: dict, nav: float):
    """
    处理单个标的的信号生成和执行（Regime 版）

    替换原版 _process_symbol()，使用和回测完全一致的策略逻辑。

    数据流：
      1. storage.get_klines(symbol, "1h")
      2. FeatureEngine.compute_all(df)
      3. add_regime_column(features)
      4. RegimeAdaptiveStrategy.should_enter() / check_exit()
      5. executor.execute() 下单
    """
    # ── 数据准备（和回测中的 _prepare_features 完全一致）──
    df = self.storage.get_klines(symbol, "1h")
    if len(df) < 500:
        logger.debug(f"{symbol} 数据不足 ({len(df)} 条)，跳过")
        return

    features = self.feature_engine.compute_all(df)
    if features.empty:
        return

    # 添加 regime 标签
    features = add_regime_column(features)

    # 取最新的两行
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

    # 记录信号到数据库
    self.storage.save_signal({
        "symbol": symbol,
        "timestamp": int(time.time() * 1000),
        "strategy": "regime_adaptive",
        "direction": regime,
        "strength": row.get("adx_14", 0) / 100.0,
        "confidence": 0.0,
        "model_version": "regime_v2",
    })

    position = self.regime_positions.get(symbol)
    strategy = self.regime_strategy
    cfg = self.regime_cfg

    # ════════════════════════════════════════════════════════
    # 持仓中 → 检查出场
    # ════════════════════════════════════════════════════════
    if position is not None:
        bar_count = position.get("bar_count", 0) + 1
        position["bar_count"] = bar_count
        position["highest_since_entry"] = max(
            position["highest_since_entry"], current_price)

        # 减仓
        if strategy.check_partial_exit(row, position):
            sell_qty = position["original_qty"] * cfg.partial_exit_pct
            if sell_qty > 0 and sell_qty < position["qty"]:
                result = self.executor.execute(
                    symbol, "SELL", sell_qty, current_price,
                    strategy="regime_adaptive",
                    algo=self.config.execution.algo)
                if result:
                    position["qty"] -= sell_qty
                    position["partial_done"] = True
                    logger.info(f"[{symbol}] 减仓 {sell_qty:.6f} @ {current_price:.2f}")

        # 出场
        should_exit, reason = strategy.check_exit(row, position, bar_count)
        if should_exit:
            qty_to_sell = position["qty"]
            ok, risk_reason = self.risk_mgr.pre_trade_check(
                symbol, "SELL", qty_to_sell, current_price, nav, prices)

            if ok:
                result = self.executor.execute(
                    symbol, "SELL", qty_to_sell, current_price,
                    strategy="regime_adaptive",
                    algo=self.config.execution.algo)
                if result:
                    pnl_pct = current_price / position["entry_price"] - 1
                    icon = "🟢" if pnl_pct > 0 else "🔴"
                    logger.info(
                        f"  {icon} [{symbol}] 平仓 @ {current_price:.2f} | "
                        f"入={position['entry_price']:.2f} | "
                        f"PnL={pnl_pct:+.1%} | {bar_count}bars | {reason}")
                    self.regime_positions[symbol] = None
                    self.risk_mgr.record_order()
            else:
                logger.warning(f"[{symbol}] 平仓风控拒绝: {risk_reason}")
            return

    # ════════════════════════════════════════════════════════
    # 无持仓 → 检查入场
    # ════════════════════════════════════════════════════════
    if position is None and strategy.should_enter(row, prev_row):
        natr = row.get("natr_20", 0)
        if pd.isna(natr) or natr <= 0:
            return

        qty, stop_loss = strategy.calc_position(nav, current_price, natr, regime)
        if qty <= 0:
            return

        ok, risk_reason = self.risk_mgr.pre_trade_check(
            symbol, "BUY", qty, current_price, nav, prices)

        if ok:
            result = self.executor.execute(
                symbol, "BUY", qty, current_price,
                strategy="regime_adaptive",
                algo=self.config.execution.algo)
            if result:
                atr_abs = natr * current_price
                self.regime_positions[symbol] = {
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
                self.risk_mgr.record_order()
        else:
            logger.info(f"[{symbol}] 开仓风控拒绝: {risk_reason}")


# ═══════════════════════════════════════════════════════════════
# 区域 4：新增 cmd_backtest_v2 命令
# ═══════════════════════════════════════════════════════════════

def cmd_backtest_v2(args):
    """
    运行 v2 回测（Regime-Adaptive）

    在 main() 的 argparse 中添加:
        parser.add_argument("--backtest-v2", action="store_true",
                            help="政体自适应回测 v2")

    在 main() 的 if/elif 链中添加:
        elif args.backtest_v2:
            cmd_backtest_v2(args)
    """
    from backtest_runner import BacktestEngineV2, write_snapshot
    from alpha.regime_strategy import RegimeStrategyConfig

    # 加载配置以获取数据库路径
    from config.loader import Config
    config = Config(args.config)
    db_path = config._data.get("database", {}).get("path", "data/quant.db")

    if not Path(db_path).exists():
        logger.error(f"数据库不存在: {db_path}")
        logger.error("请先运行: python main.py --sync-data")
        return

    cfg = RegimeStrategyConfig()

    engine = BacktestEngineV2(
        db_path=db_path,
        initial_capital=10000.0,
        start_date=None,
        end_date=None,
        cfg=cfg,
        enable_regime=True,
    )

    report = engine.run()
    if report:
        write_snapshot(report, "backtest_snapshot.txt", db_path, None, None)


# ═══════════════════════════════════════════════════════════════
# 区域 5：settings.yaml 新增配置段
#
# 在 settings.yaml 的 strategy 段下添加：
# ═══════════════════════════════════════════════════════════════

SETTINGS_YAML_ADDITION = """
# ---- 政体自适应策略参数 ----
strategy:
  regime:
    # 入场过滤
    rsi_low: 35.0             # RSI 下限
    rsi_high: 65.0            # RSI 上限
    adx_min: 18.0             # 最小 ADX
    adx_min_weak: 22.0        # BULL_WEAK 下的最小 ADX
    dist_ema_max_atr: 2.0     # 价格距 EMA20 最大距离(ATR倍数)
    vol_ratio_min: 0.8        # 最小相对成交量

    # 出场
    trailing_atr_mult: 2.5    # 移动止损 ATR 倍数
    take_profit_atr_mult: 4.0 # 止盈 ATR 倍数
    partial_exit_atr_mult: 1.5
    partial_exit_pct: 0.3
    max_holding_bars: 30      # 最大持仓 bar 数(1h)
    time_stop_min_profit: 0.005

    # 仓位
    risk_per_trade: 0.03      # 每笔风险占权益比
    risk_per_trade_weak: 0.015
    stop_atr_mult: 2.0
    max_position_pct: 0.50

    # 成本
    commission_pct: 0.001     # 手续费 0.1%
    slippage_pct: 0.001       # 滑点 0.1%
"""


# ═══════════════════════════════════════════════════════════════
# 集成步骤汇总
# ═══════════════════════════════════════════════════════════════

INTEGRATION_GUIDE = """
═══════════════════════════════════════════════════════════════
  集成指南 — 将 Regime 策略接入 main.py
═══════════════════════════════════════════════════════════════

Step 1: 添加 import（main.py 顶部）
  from alpha.regime_strategy import (
      RegimeAdaptiveStrategy, RegimeStrategyConfig,
      add_regime_column, classify_regime,
      REGIME_BULL_TREND, REGIME_BULL_WEAK, REGIME_RANGE,
      REGIME_BEAR_WEAK, REGIME_BEAR_TREND,
  )

Step 2: TradingEngine.__init__() 末尾添加
  self._init_regime_strategy = _init_regime_strategy.__get__(self)
  self._init_regime_strategy()

  或者直接把 _init_regime_strategy 的代码粘贴进 __init__()

Step 3: 替换信号处理
  方案 A（推荐）：在 _run_cycle() 中把
    self._process_symbol(symbol, prices, nav)
  改为
    self._process_symbol_regime(symbol, prices, nav)

  方案 B（保守）：保留原版 _process_symbol()，
  新增 _process_symbol_regime() 并用配置项切换

Step 4: argparse 新增
  parser.add_argument("--backtest-v2", action="store_true",
                      help="政体自适应回测 v2")

  main() if/elif 链新增：
  elif args.backtest_v2:
      cmd_backtest_v2(args)

Step 5: settings.yaml 添加 regime 配置段（见上面的 SETTINGS_YAML_ADDITION）

Step 6: 验证策略一致性
  # 先跑回测
  python backtest_runner.py --start 2025-01-01 --snapshot bt_regime.txt

  # 再跑一轮模拟（对比信号是否一致）
  python main.py --once

═══════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(INTEGRATION_GUIDE)
