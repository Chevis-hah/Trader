"""
加密货币量化交易系统 v2.0
工业级架构 · 全量数据 · ML Alpha · 智能执行 · 三层风控

用法:
    python main.py                          # 模拟交易
    python main.py --live                   # 实盘交易
    python main.py --backtest               # 回测
    python main.py --train                  # 仅训练模型
    python main.py --sync-data              # 仅同步数据
    python main.py --validate               # 数据质量检查
    python main.py --network-probe          # 本机检测代理与 Binance REST 连通（需在 WSL/本机执行）
    python main.py --config path/to/cfg.yaml # 指定配置文件
"""
import argparse
import sys
from pathlib import Path

from config.loader import load_config
from core.engine import TradingEngine
from data.client import BinanceAPIError, BinanceClient
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


def cmd_network_probe(args):
    """
    代理与 Binance REST 一键检测。Agent 无法替你执行此闭环（无你的 WSL/Clash/局域网）。
    成功：打印 serverTime 并以退出码 0 结束；失败：退出码 1。
    """
    import requests
    from data.client import network_probe_diagnostics

    config = load_config(args.config)
    logger.info("网络探测（仅反映当前机器；请在出现连接问题时于本机/WSL 运行本命令）")
    diag = network_probe_diagnostics(config)
    for key in sorted(diag.keys()):
        logger.info(f"  {key}: {diag[key]}")

    client = BinanceClient(config)
    logger.info(f"  session.proxies: {dict(client._session.proxies)}")

    try:
        client._request("GET", "/api/v3/ping")
        t = client.get_server_time()
        logger.info(f"  Binance REST 正常 | serverTime(ms)={t}")
        sys.exit(0)
    except BinanceAPIError as e:
        logger.error(f"  Binance API 响应: {e}")
        if e.status_code == 451:
            logger.error(
                "  HTTP 451：代理已连通，但 Binance 认定当前出口 IP 位于受限地区（合规策略），"
                "与 Allow LAN / wsl_host 无关。请在 Clash 中更换到 Binance 支持的地区的干净节点；"
                "若浏览器访问 binance.com 同样提示地区限制，可佐证为出口地区问题。"
            )
        sys.exit(1)
    except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError) as e:
        logger.error(f"  代理或 TCP 连接失败: {e}")
        logger.error(
            "  建议: Clash 开启 Allow LAN；核对 wsl_clash_port；"
            "必要时设置 proxy.wsl_host 为 Windows 在 vEthernet(WSL) 上的 IP。"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"  Binance REST 不可用: {e}")
        sys.exit(1)


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
    """回测"""
    config = load_config(args.config)
    storage = Storage(config.get_nested("data.database.path", "data/quant.db"))
    feature_engine = FeatureEngine(
        windows=config.get_nested("features.lookback_windows", [5, 10, 20, 60, 120, 240, 480]))

    for symbol in config.get_symbols():
        df = storage.get_klines(symbol, "1h")
        if len(df) < 2000:
            logger.warning(f"{symbol}: 数据不足 ({len(df)} 条)")
            continue

        logger.info(f"回测 {symbol}，共 {len(df)} 条K线")

        # 滚动窗口回测
        train_size = 1500
        test_size = 200
        step = 200

        equity = [10000.0]
        trades_pnl = []
        position = 0.0
        entry_price = 0.0

        for start in range(0, len(df) - train_size - test_size, step):
            train_df = df.iloc[start:start + train_size]
            test_df = df.iloc[start + train_size:start + train_size + test_size]

            # 训练
            train_feat = feature_engine.compute_all(train_df, include_target=True, target_periods=[1])
            train_feat = feature_engine.preprocess(train_feat, method="zscore")

            alpha = AlphaModel(config, storage)
            try:
                alpha.train(train_feat, target_col="target_dir_1")
            except Exception as e:
                logger.debug(f"训练跳过: {e}")
                continue

            # 测试
            test_feat = feature_engine.compute_all(test_df)
            test_feat = feature_engine.preprocess(test_feat, method="zscore")

            try:
                preds = alpha.predict(test_feat)
            except Exception:
                continue

            for i in range(len(preds)):
                sig = preds.iloc[i]["signal"]
                price = test_df.iloc[i]["close"] if i < len(test_df) else test_df.iloc[-1]["close"]

                if sig in ("BUY", "STRONG_BUY") and position == 0:
                    qty = equity[-1] * 0.95 / price
                    position = qty
                    entry_price = price * 1.0003  # 滑点

                elif sig in ("SELL", "STRONG_SELL") and position > 0:
                    exit_price = price * 0.9997
                    pnl = position * (exit_price - entry_price)
                    equity.append(equity[-1] + pnl)
                    trades_pnl.append(pnl / (position * entry_price))
                    position = 0
                else:
                    equity.append(equity[-1] + position * (price - (test_df.iloc[i-1]["close"] if i > 0 else price)) if position > 0 else equity[-1])

        # 报告
        eq = np.array(equity)
        tp = np.array(trades_pnl) if trades_pnl else np.array([0.0])
        report = generate_report(eq, tp, frequency="1h")
        print(format_report(report))


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
    parser.add_argument(
        "--network-probe",
        action="store_true",
        help="检测代理与 Binance REST（需在本人电脑/WSL 上运行）",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Crypto Quant Engine v2.0")
    logger.info("=" * 60)

    if args.network_probe:
        cmd_network_probe(args)
    elif args.sync_data:
        cmd_sync_data(args)
    elif args.validate:
        cmd_validate(args)
    elif args.train:
        cmd_train(args)
    elif args.backtest:
        cmd_backtest(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
