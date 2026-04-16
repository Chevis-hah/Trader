# config/ - 配置模块

## 职责

YAML 配置加载、环境变量解析、配置校验。

> **v2.3**：`config/settings.yaml` 默认已去除 `1h` 间隔与 `regime` 策略块；滑点 bps 与执行段对齐实测。见根目录 `CHANGELOG_v23.md` 与 `SYNC_UPDATE_LOG.md`。

## 模块组成

### loader.py - Config + load_config

**Config 类**: 包装 dict，支持:
- 点号访问: `config.exchange.api_key`
- 嵌套查找: `config.get_nested("risk.position.stop_loss_pct")`
- 标的配置: `config.get_symbol_config("BTCUSDT")`

**load_config(path)**:
1. 加载 YAML 文件
2. 解析 `${ENV_VAR:default}` 模式 (从环境变量替换)
3. 校验必须配置节: system, exchange, universe, data, risk
4. 返回 Config 实例

**ConfigError**: 配置缺失/无效时抛出

### 配置文件

| 文件 | 版本 | 标的 | 特性 |
|------|------|------|------|
| settings.yaml (config/) | **v2.3** | BTC/ETH/BNB/SOL | strategy 参数节；**无默认 1h**；**无 regime 节**；滑点与执行与 v2.3 迁移一致 |
| symbols_extended.yaml | 参考 | +AVAX/LINK | 扩展标的配置模板 |

**注意**: 顶层 settings.yaml 已删除，统一使用 config/settings.yaml

### v2.1 配置结构 (config/settings.yaml)

```yaml
system:          # paper_capital, cycle_interval, mode
exchange:        # api_key/secret, proxy, wsl_clash_port, wsl_host
universe:        # symbols 列表 (BTC/ETH/BNB/SOL)
data:            # primary_interval(4h), higher_interval(1d), history_days
strategy:        # per-strategy 参数 (common + 策略特有)
  common:        # 通用参数 (slippage_pct=0.0021)
  triple_ema:    # regime_gate_enabled, regime_adx_min, trail_atr_base, trail_profit_bonus
  macd_momentum: # regime_gate_enabled, regime_adx_min, cooldown_bars_after_stop
  regime:        # ...
risk:            # position/order/portfolio 阈值, circuit_breaker
portfolio:       # method (risk_parity), rebalance_interval
execution:       # slippage_model, algorithms, base_bps=5, impact_coefficient=0.2
logging:         # level, file
```

### v2.1 关键参数变更

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| slippage_pct | 0.001 | 0.0021 | 滑点校正 (BTC ~21bps) |
| regime_gate_enabled | false | true | ADX 门控入场 |
| regime_adx_min | - | 22~25 | ADX 门槛 |
| trail_atr_base | - | 2.5~2.8 | 自适应追踪基础倍数 |
| trail_profit_bonus | - | 0.3 | 盈利放宽系数 |
| cooldown_bars_after_stop | - | 8 | 止损后冷却延长 |

## 依赖关系

- **被依赖**: 所有模块通过 `load_config()` 获取配置
- **无外部依赖**: 仅使用标准库 + pyyaml

## 关键约束

- API key/secret 通过 `${ENV_VAR:default}` 模式注入，不要硬编码
- 修改配置结构后需同步更新 `load_config()` 中的校验逻辑
- WSL 代理配置 (`wsl_clash_port`, `wsl_host`) 仅开发环境需要
- 新增策略参数必须在 `strategy` 节添加对应配置
- 使用 `scripts/apply_settings_patch.py` 批量校正参数
