# core/ - 核心引擎模块

## 职责

交易引擎主循环、事件总线。**v2.3** 活跃版本：REST + 规则策略（趋势策略已按 v2.3 回滚过滤链；配置默认无 1h/regime）。详情见根目录 `SYNC_UPDATE_LOG.md`。

## 模块组成

### engine.py - TradingEngine（v2.3）

主交易循环，现金制记账 (`self.cash`, `_apply_buy_cash()`, `_apply_sell_cash()`)。

**启动流程** (`warmup()`):
1. `BinanceClient` 连通性检查
2. `HistoryDownloader.sync_all()` 批量拉取历史数据
3. `build_strategy()` 从 registry 构建策略实例

**主循环** (`run_cycle()`):
1. `_update_data()` - REST 增量获取 klines → Storage
2. `_get_prices()` - REST ticker 价格 (DB 兜底)
3. `PositionTracker.update_all_extremes()` - 更新持仓极值
4. `RiskManager.check_portfolio()` + `check_positions()` - 风控检查
5. 逐标的: `_prepare_symbol_features()` → `strategy.should_enter()` / `check_exit()`
6. `OrderExecutor.execute()` → BinanceClient 下单 / 模拟成交
7. `Storage.save_order()` / `save_signal()` - 持久化

**每标的策略状态**: `self.strategy_states[symbol]` 和 `self.positions[symbol]` 独立维护

**与旧版顶层 engine.py 的区别**:
- 旧版: WebSocket 实时数据 + ML (AlphaModel) 信号
- v3: REST 轮询 + 规则策略 (strategy_registry)
- v3 不使用 ws_manager.py

### events.py - 事件系统

**EventType 枚举**: 22 种事件类型，覆盖 5 大类:
- Data: KLINE_CLOSED, KLINE_PARTIAL, ORDERBOOK_UPDATE, TICKER_UPDATE
- Signal: SIGNAL_GENERATED, SIGNAL_CONFIRMED, SIGNAL_REJECTED
- Execution: ORDER_CREATED, ORDER_FILLED, ORDER_CANCELLED, ORDER_REJECTED
- Position: POSITION_OPENED, POSITION_CLOSED, POSITION_MODIFIED
- Risk/System: RISK_ALERT, CIRCUIT_BREAKER, ENGINE_STARTED, etc.

**Event**: dataclass (type, data dict, timestamp, source)

**EventBus**: 线程安全发布/订阅
- `subscribe(event_type, handler)`: 注册处理器
- `publish(event)`: 异步处理 (daemon 线程 Queue 消费)
- `publish_sync(event)`: 同步处理

## 依赖关系

- **输入**: `data/client.py` (行情), `data/storage.py` (读写), `alpha/` (策略), `risk/manager.py` (风控)
- **输出**: `execution/executor.py` (下单), `utils/logger.py` (日志)
- **配置**: `config/settings.yaml`

## 关键约束

- v3 引擎不使用 WebSocket，仅 REST 轮询
- 现金制记账: 每笔交易直接加减 cash，非净值制
- 主循环间隔由配置 `system.cycle_interval` 控制
- EventBus 的 publish 是异步的 (daemon 线程)，不要在 handler 中做阻塞操作
