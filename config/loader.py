"""
配置加载器
- 从 YAML 加载配置
- 支持环境变量替换 ${VAR_NAME}
- 配置校验
- 热更新支持
"""
import os
import re
import copy
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import yaml


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env_vars(value: Any) -> Any:
    """递归替换字符串中的 ${ENV_VAR} 或 ${ENV_VAR:default}"""
    if isinstance(value, str):
        def replacer(m):
            var_name, default = m.group(1), m.group(2)
            return os.environ.get(var_name, default if default is not None else m.group(0))
        return _ENV_VAR_PATTERN.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


class ConfigError(Exception):
    pass


class Config:
    """
    全局配置对象
    支持点号访问: config.exchange.api_key
    """

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            return super().__getattribute__(key)
        try:
            val = self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")
        if isinstance(val, dict):
            return Config(val)
        return val

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getattr__(key)
        except (AttributeError, KeyError):
            return default

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def get_nested(self, dotted_key: str, default: Any = None) -> Any:
        """通过点号路径获取: config.get_nested('risk.position.stop_loss_pct')"""
        keys = dotted_key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def get_symbol_config(self, symbol: str) -> Optional[dict]:
        """获取指定交易对的配置"""
        for s in self._data.get("universe", {}).get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return None

    def get_symbols(self) -> list[str]:
        return [s["symbol"] for s in self._data.get("universe", {}).get("symbols", [])]

    def get_timeframes(self) -> list[dict]:
        return self._data.get("data", {}).get("history", {}).get("timeframes", [])


def load_config(path: str | Path = None) -> Config:
    """
    加载配置文件
    默认路径: config/settings.yaml
    """
    if path is None:
        path = Path(__file__).parent / "settings.yaml"
    path = Path(path)

    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("配置文件格式错误")

    resolved = _resolve_env_vars(raw)

    # 基本校验
    _validate(resolved)

    return Config(resolved)


def _validate(cfg: dict):
    """配置校验"""
    required_sections = ["system", "exchange", "universe", "data", "risk"]
    for sec in required_sections:
        if sec not in cfg:
            raise ConfigError(f"缺少必需配置节: {sec}")

    symbols = cfg.get("universe", {}).get("symbols", [])
    if not symbols:
        raise ConfigError("至少需要配置一个交易对 (universe.symbols)")

    for s in symbols:
        for field in ["symbol", "min_qty", "qty_precision", "price_precision"]:
            if field not in s:
                raise ConfigError(f"交易对配置缺少字段 {field}: {s}")
