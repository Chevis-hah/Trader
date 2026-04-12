"""
Crypto Quant Engine v3

本版默认主线：
- 规则趋势策略优先，不再默认走 ML
- 回测/模拟/实盘共用 strategy_registry
- 默认策略为 triple_ema，可切换为 macd_momentum / regime
"""
from __future__ import annotations

import argparse
from pathlib import Path

from alpha.strategy_registry import available_strategies, build_strategy, resolve_strategy_name
from backtest_runner import BacktestEngine, write_snapshot
from config.loader import load_config
from core.engine import TradingEngine
from data.client import BinanceClient
from data.historical import HistoryDownloader
from data.storage import Storage
from utils.logger import get_logger

logger = get_logger("main")


def _resolve_snapshot_path(args) -> str:
    if args.bt_snapshot:
        return args.bt_snapshot
    return f"backtest_{args.strategy}_snapshot.txt"


def cmd_run(args):
    config = load_config(args.config)
    simulate = not args.live
    engine = TradingEngine(config, simulate=simulate, strategy_name=args.strategy)
    engine.warmup(skip_history=args.skip_data, skip_train=True)

    if args.once:
        engine.run_cycle()
        engine.shutdown()
    else:
        engine.run(interval_seconds=args.interval)


def cmd_sync_data(args):
    config = load_config(args.config)
    client = BinanceClient(config)
    storage = Storage(config.get_nested("data.database.path", "data/quant.db"))
    downloader = HistoryDownloader(config, client, storage)
    downloader.sync_all(max_workers=3)


def cmd_validate(args):
    config = load_config(args.config)
    storage = Storage(config.get_nested("data.database.path", "data/quant.db"))
    client = BinanceClient(config)
    downloader = HistoryDownloader(config, client, storage)

    strategy = build_strategy(config=config, explicit_name=args.strategy)
    intervals = {strategy.primary_interval}
    if strategy.higher_interval:
        intervals.add(strategy.higher_interval)

    for symbol in config.get_symbols():
        for interval in sorted(intervals):
            report = downloader.validate_data(symbol, interval)
            logger.info(f"{symbol}/{interval}: {report}")


def cmd_backtest(args):
    config = load_config(args.config)
    db_path = config.get_nested("data.database.path", "data/quant.db")
    if not Path(db_path).exists():
        logger.error(f"数据库不存在: {db_path}")
        logger.error("请先运行: python main.py --sync-data")
        return

    engine = BacktestEngine(
        db_path=db_path,
        strategy_name=args.strategy,
        initial_capital=args.bt_capital,
        start_date=args.bt_start,
        end_date=args.bt_end,
        config=config,
    )
    report = engine.run()
    if report:
        snapshot_path = _resolve_snapshot_path(args)
        write_snapshot(report, snapshot_path, db_path, args.bt_start, args.bt_end)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crypto Quant Engine v3")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--strategy", type=str, default="triple_ema", choices=available_strategies(), help="策略名称")

    parser.add_argument("--live", action="store_true", help="实盘模式")
    parser.add_argument("--once", action="store_true", help="只运行一轮")
    parser.add_argument("--interval", type=int, default=3600, help="循环间隔(秒)")
    parser.add_argument("--skip-data", action="store_true", help="跳过数据同步")

    parser.add_argument("--sync-data", action="store_true", help="仅同步历史数据")
    parser.add_argument("--validate", action="store_true", help="数据质量检查")
    parser.add_argument("--backtest", action="store_true", help="统一规则策略回测")
    parser.add_argument("--backtest-v2", action="store_true", help="兼容旧命令，等价于 --backtest")

    parser.add_argument("--bt-capital", type=float, default=10000.0, help="回测初始资金")
    parser.add_argument("--bt-start", type=str, default=None, help="回测起始日期 YYYY-MM-DD")
    parser.add_argument("--bt-end", type=str, default=None, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--bt-snapshot", type=str, default=None, help="回测快照输出路径")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Crypto Quant Engine v3")
    logger.info("=" * 60)
    logger.info(f"strategy={args.strategy}")

    if args.sync_data:
        cmd_sync_data(args)
    elif args.validate:
        cmd_validate(args)
    elif args.backtest or args.backtest_v2:
        cmd_backtest(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
