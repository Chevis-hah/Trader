# execution/ - 执行与持仓模块

## 职责

订单执行 (模拟/实盘)、滑点模型、持仓跟踪与 PnL 计算。

> v2.3 默认滑点等在 `config/settings.yaml` 的 `execution` 节更新；摘要见根目录 `SYNC_UPDATE_LOG.md`。

## 模块组成

### executor.py - OrderExecutor

统一执行接口，支持 simulate 和 live 两种模式。

**滑点模型** (`SlippageModel`):
- fixed: 固定滑点
- linear: 线性滑点 (与订单规模成正比)
- sqrt: 平方根滑点模型

**执行算法**:
- market: 市价单
- TWAP: 时间加权平均价格 (分片执行)
- iceberg: 冰山单 (随机化部分展示)

**精度处理**: 从 symbol 配置读取 tick_size / lot_size，处理 min_qty / min_notional 检查

**模式分支**:
- 实盘: 调用 `BinanceClient.create_order()`
- 模拟: 应用滑点模型模拟成交

**副作用**: 更新 PositionTracker、写入 Storage、发布 ORDER_FILLED 事件

### position.py - PositionTracker + Position

**Position dataclass**:
- avg_entry_price, cost_basis, quantity
- highest_since_entry, lowest_since_entry (用于跟踪止损)
- MFE (最大有利波动), MAE (最大不利波动)

**PositionTracker**:
- `open_position()`: 开仓，支持加仓
- `close_position()`: 平仓，支持部分平仓，FIFO PnL 计算
- `update_extremes()`: 根据最新价更新极值
- `get_total_exposure()`: 总敞口
- `get_total_unrealized_pnl()`: 总浮动盈亏
- `get_portfolio_summary()`: 组合概览

## 依赖关系

- **输入**: `data/client.py` (实盘下单), `alpha/` (信号 → 订单参数)
- **输出**: `risk/manager.py` (持仓信息用于风控), `data/storage.py` (订单记录)
- **被调用**: `core/engine.py` 主循环驱动

## 关键约束

- 部分平仓使用 FIFO 原则计算 PnL
- 滑点模型仅影响模拟模式，实盘使用真实成交价
- 精度处理必须严格遵守 Binance 的 tick_size / lot_size 规则
- MFE/MAE 在每次 update_extremes() 时更新，用于策略评估
