"""
P4: ML 信号过滤器

将已有的 241 特征 + GradientBoosting 模型作为入场过滤条件叠加到规则策略上。
不替换规则策略的信号，而是在规则信号成立时，用 ML 做二次确认。

核心逻辑:
  if rule_strategy.should_enter(row):
      if ml_filter.confirm(row) >= threshold:
          -> 允许入场
      else:
          -> 过滤掉此信号

用法:
  python -c "
  from alpha.ml_signal_filter import MLSignalFilter
  f = MLSignalFilter('data/quant.db')
  f.train(start='2024-01-01', end='2024-12-31')
  f.save('models/ml_filter.pkl')
  "

集成方式:
  在 backtest_runner.py 中:
    ml_filter = MLSignalFilter.load('models/ml_filter.pkl')
    # 在入场检查时:
    if strategy.should_enter(row, prev_row, state):
        if ml_filter.confirm(row) >= 0.55:
            # 入场
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logger import get_logger

logger = get_logger("ml_filter")

# 用于 ML 的核心特征子集 (从 241 个中精选)
ML_FEATURES = [
    # 动量
    "ret_5", "ret_10", "ret_20", "ret_60",
    "momentum_5", "momentum_20",
    # 波动率
    "natr_20", "natr_60",
    "realized_vol_20", "realized_vol_60",
    # 趋势
    "adx_14",
    "ema_20", "ema_50",
    "macd_line", "macd_signal", "macd_hist",
    "rsi_14",
    # 量价
    "rel_volume_20",
    "obv_slope_20",
    # 均值回归
    "bb_position_20",
    "zscore_20",
    # 微观结构
    "amihud_20",
    # 市场状态
    "vol_regime_60",
    "trend_strength_20",
]


class MLSignalFilter:
    """
    ML 信号过滤器

    训练: 用历史数据训练一个分类器，预测 "未来 N bar 是否盈利"
    推理: 对当前 bar 的特征给出概率分数 [0, 1]
    """

    def __init__(self, db_path: str = None, feature_cols: list = None):
        self.db_path = db_path
        self.feature_cols = feature_cols or ML_FEATURES
        self.model = None
        self.scaler = None
        self.threshold = 0.55
        self._feature_importance = None

    def train(
        self,
        start: str = None,
        end: str = None,
        target_bars: int = 6,        # 预测未来 6 个 4h bar (1天) 的收益
        target_threshold: float = 0.005,  # 0.5% 以上算正样本
        n_splits: int = 5,
        purge_gap: int = 6,
    ):
        """
        训练 ML 过滤模型

        使用 Purged 时序 CV 防止数据泄漏
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, roc_auc_score

        if not self.db_path:
            raise ValueError("需要 db_path")

        from data.storage import Storage
        from data.features import FeatureEngine

        storage = Storage(self.db_path)
        feat_engine = FeatureEngine()

        all_features = []
        for sym in ["BTCUSDT", "ETHUSDT"]:
            klines = storage.get_klines(sym, "4h", limit=100000)
            if klines.empty:
                continue
            features = feat_engine.compute_all(klines)
            if features.empty:
                continue
            if "open_time" not in features.columns:
                features = features.copy()
                features["open_time"] = klines["open_time"].to_numpy()

            if start:
                ts = int(pd.Timestamp(start).timestamp() * 1000)
                features = features[features["open_time"] >= ts]
            if end:
                ts = int(pd.Timestamp(end).timestamp() * 1000)
                features = features[features["open_time"] <= ts]

            # 生成标签: 未来 N bar 收益是否 > threshold
            features["forward_return"] = (
                features["close"].shift(-target_bars) / features["close"] - 1
            )
            features["target"] = (features["forward_return"] > target_threshold).astype(int)
            features["symbol"] = sym
            all_features.append(features)

        if not all_features:
            logger.error("无训练数据")
            return

        df = pd.concat(all_features, ignore_index=True)
        df = df.dropna(subset=["target"])

        # 选择可用特征
        available = [c for c in self.feature_cols if c in df.columns]
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.warning(f"缺少特征: {missing}")
        self.feature_cols = available

        X = df[available].copy()
        y = df["target"].copy()

        # 清洗
        X = X.replace([np.inf, -np.inf], np.nan)
        valid_mask = X.notna().all(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(f"训练集: {len(X)} 样本, {len(available)} 特征, "
                    f"正样本率: {y.mean():.2%}")

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Purged 时序 CV
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=purge_gap)

        scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=30,
                max_features=0.7,
                random_state=42 + fold,
            )
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            try:
                auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc = 0.5
            scores.append(auc)
            logger.info(f"  Fold {fold}: AUC={auc:.4f}")

        avg_auc = np.mean(scores)
        logger.info(f"平均 AUC: {avg_auc:.4f}")

        # 最终模型 (全量训练)
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=30,
            max_features=0.7,
            random_state=42,
        )
        self.model.fit(X_scaled, y)

        # 特征重要性
        self._feature_importance = dict(zip(available, self.model.feature_importances_))
        top_features = sorted(self._feature_importance.items(), key=lambda x: -x[1])[:10]
        logger.info("Top 10 特征:")
        for fname, imp in top_features:
            logger.info(f"  {fname}: {imp:.4f}")

        return {"avg_auc": avg_auc, "n_samples": len(X), "n_features": len(available)}

    def confirm(self, row: pd.Series) -> float:
        """
        对单条 bar 给出 ML 确认分数

        Returns:
            float [0, 1]: 概率分数，>= threshold 意味着 ML 确认做多
        """
        if self.model is None:
            return 1.0  # 模型未训练，默认放行

        features = []
        for col in self.feature_cols:
            val = row.get(col, np.nan)
            features.append(float(val) if pd.notna(val) else np.nan)

        x = np.array(features).reshape(1, -1)
        if np.any(np.isnan(x)):
            return 0.5  # 特征缺失，中性分数

        try:
            x_scaled = self.scaler.transform(x)
            prob = self.model.predict_proba(x_scaled)[0, 1]
            return float(prob)
        except Exception as e:
            logger.warning(f"ML 推理失败: {e}")
            return 0.5

    def save(self, path: str):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "threshold": self.threshold,
                "feature_importance": self._feature_importance,
            }, f)
        logger.info(f"模型已保存: {path}")

    @classmethod
    def load(cls, path: str) -> MLSignalFilter:
        """加载模型"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_cols = data["feature_cols"]
        obj.threshold = data.get("threshold", 0.55)
        obj._feature_importance = data.get("feature_importance")
        logger.info(f"模型已加载: {path}")
        return obj


def patch_strategy_with_ml(strategy, ml_filter: MLSignalFilter, threshold: float = 0.55):
    """
    给策略添加 ML 过滤

    用法:
        ml = MLSignalFilter.load("models/ml_filter.pkl")
        patch_strategy_with_ml(strategy, ml, threshold=0.55)
    """
    original_should_enter = strategy.should_enter

    def filtered_should_enter(row, prev_row, state=None):
        if not original_should_enter(row, prev_row, state):
            return False

        score = ml_filter.confirm(row)
        if score >= threshold:
            logger.debug(f"ML 确认入场 (score={score:.3f})")
            return True
        else:
            logger.debug(f"ML 过滤掉信号 (score={score:.3f} < {threshold})")
            return False

    strategy.should_enter = filtered_should_enter
    strategy._ml_filter = ml_filter
    logger.info(f"已为 {strategy.name} 启用 ML 过滤 (threshold={threshold})")
    return strategy
