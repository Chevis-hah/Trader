"""
结构化日志系统
- JSON 格式日志用于生产环境
- 可读格式用于开发环境
- 自动按天分割文件
"""
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class JsonFormatter(logging.Formatter):
    """JSON 结构化日志格式"""
    def format(self, record):
        log_entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        return json.dumps(log_entry, ensure_ascii=False)


class ReadableFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m", "INFO": "\033[32m",
        "WARNING": "\033[33m", "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        return (f"{color}[{ts}] [{record.name:>12s}] "
                f"[{record.levelname:>7s}]{self.RESET} {record.getMessage()}")


_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, log_dir: Optional[str | Path] = "logs",
               level: str = "INFO", json_format: bool = False) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(f"quant.{name}")
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    if not logger.handlers:
        # Console
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(ReadableFormatter() if not json_format else JsonFormatter())
        logger.addHandler(sh)

        # File
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now().strftime("%Y%m%d")
            fh = logging.FileHandler(
                log_dir / f"{name}_{today}.log", encoding="utf-8")
            fh.setFormatter(JsonFormatter())
            logger.addHandler(fh)

    _loggers[name] = logger
    return logger
