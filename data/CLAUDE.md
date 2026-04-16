# data/ - 数据层模块

## 职责

Binance API 交互、历史数据下载、技术特征计算、SQLite 持久化存储；**v2.3** 起增加横截面路线所需的 **universe** 构建（见 `universe.py`）。

> 仓库级 v2.3 同步摘要见根目录 `SYNC_UPDATE_LOG.md`（2026-04-16）。

## 模块组成

### universe.py（v2.3）

- Top N 标的滚动 universe 构造，供 `alpha/cross_sectional_momentum.py` 与 `cross_sectional_backtest.py` 使用。
- 与 `config/settings.yaml` 中 `universe.symbols` / 间隔配置协同；v2.3 默认配置已移除 `1h`。

### client.py - BinanceClient

Binance REST API 封装，HMAC-SHA256 签名。

**关键特性**:
- WSL2 代理自动解析: `exchange.proxy.url` → `wsl_clash_port` (自动获取 Windows host IP) → `HTTPS_PROXY` 环境变量
- 令牌桶限流: 1200 req/min，指数退避重试
- HTTP 451 区域限制处理
- 方法: ping, get_klines, get_orderbook, get_balances, create_order, cancel_order, get_order, get_ticker_price

**WSL 代理解析链** (重要，开发环境依赖):
```
config.exchange.proxy.url
  → wsl_clash_port (ip route 获取 Windows host IP + 端口)
    → HTTPS_PROXY 环境变量
      → 无代理
```

### historical.py - HistoryDownloader

历史 K 线批量下载与数据质量维护。

- `sync_all(max_workers)`: ThreadPoolExecutor 并行下载多标的/多周期
- 前向填充早期数据缺口
- 增量更新: 从最新已存 K 线开始续拉
- `check_gaps()` / `fill_gaps()`: 检测并修复缺失柱 (时间差 > 1.5x 预期间隔)
- `validate_data()`: 数据质量报告 (null 数、零量柱、高低价异常、负价格、缺口数)

### features.py - FeatureEngine

技术特征引擎，产出 100+ 特征。

**特征类别**:
- **多窗口 (5/10/20/60/120/240/480)**: 动量、波动率 (realized/Parkinson/Garman-Klass/NATR)、成交量、均值回归
- **趋势**: 多周期 SMA/EMA、MACD、ADX、Ichimoku
- **振荡器**: RSI(7/14/21)、Stochastic、Williams %R、CCI、MFI、Keltner
- **微观结构**: Amihud 非流动性、Kyle's lambda、日内/日间波动率比
- **状态**: 波动率百分位、趋势强度 (SMA50-SMA200)、Hurst 代理
- **时间**: 小时/星期 cyclical 编码
- **目标**: fwd_ret_N, target_dir_N (二分类), target_3class_N (三分类)

- `preprocess()`: zscore / rank / minmax + std clip
- `merge_multi_timeframe()`: 多周期特征融合 (占位)

### storage.py - Storage

SQLite 持久化，WAL 模式。

**9 张表**:
- klines (PK: symbol+interval+open_time, WITHOUT ROWID)
- orderbook_snapshots, orders, positions, factor_values
- signals, daily_performance, model_metadata, system_state

- 批量 `upsert_klines()` via executemany
- `get_klines()` 支持时间范围 + limit
- 缓存: `cache_size`, `mmap_size` 调优

## 依赖关系

- **被依赖**: `core/engine.py` (读取 klines + 写入 orders/signals), `alpha/` (FeatureEngine), `execution/` (写 orders), `backtest_runner.py` (读 klines)
- **外部依赖**: Binance REST API (需 API key + 网络代理)

## 与顶层旧版文件的区别

| 特性 | data/client.py (当前) | 顶层 client.py (旧版) |
|------|----------------------|---------------------|
| WSL 代理解析 | 有 (自动) | 无 |
| HTTP 451 处理 | 有 | 无 |

| 特性 | data/storage.py (当前) | 顶层 storage.py (旧版) |
|------|----------------------|---------------------|
| orderbook 方法 | 无 | save/get/cleanup_orderbook |
| get_earliest_kline_time | 有 | 无 |

## 关键约束

- klines 表使用 WITHOUT ROWID 优化，主键为 (symbol, interval, open_time)
- 限流: 严格遵守 1200 req/min，否则触发 IP 封禁
- 代理: 开发环境 (WSL2) 需要正确配置 Clash 代理
- 特征计算: 100+ 特征可能产生内存压力，注意大时间范围场景
