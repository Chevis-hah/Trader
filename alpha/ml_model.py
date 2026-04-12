"""
机器学习 Alpha 模型
- 支持 GradientBoosting / RandomForest / Ensemble
- 时序交叉验证 with Purge & Embargo
- 特征重要性分析
- 自动重训练
- 模型持久化
"""
import json
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    VotingClassifier,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from config.loader import Config
from data.storage import Storage
from utils.logger import get_logger

logger = get_logger("ml_model")


class PurgedTimeSeriesSplit:
    """
    时序交叉验证 with Purge Gap & Embargo
    避免训练集和验证集之间的信息泄漏
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 6,
                 embargo: int = 3):
        self.n_splits = n_splits
        self.purge_gap = purge_gap   # 训练集末尾删除的样本数
        self.embargo = embargo       # 验证集开头跳过的样本数

    def split(self, X: pd.DataFrame):
        n = len(X)
        base_split = TimeSeriesSplit(n_splits=self.n_splits)

        for train_idx, test_idx in base_split.split(X):
            # Purge: 从训练集末尾移除
            if self.purge_gap > 0:
                train_idx = train_idx[:-self.purge_gap]

            # Embargo: 从测试集开头移除
            if self.embargo > 0 and len(test_idx) > self.embargo:
                test_idx = test_idx[self.embargo:]

            yield train_idx, test_idx


class AlphaModel:
    """
    机器学习 Alpha 信号生成器
    """

    def __init__(self, config: Config, storage: Storage):
        self.config = config
        self.storage = storage
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.feature_importances: dict[str, float] = {}
        self.model_id: str = ""
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

        ml_cfg = config.alpha.ml
        self._model_type = ml_cfg.model_type
        self._target = ml_cfg.target
        self._train_cfg = ml_cfg.train
        self._params = ml_cfg.params
        self._signal_thresholds = ml_cfg.signal_threshold
        self._feature_selection = ml_cfg.feature_selection

        logger.info(f"Alpha 模型初始化 | 类型={self._model_type} | 目标={self._target}")

    # ==============================================================
    # 模型构建
    # ==============================================================
    def _build_model(self):
        if self._model_type == "gradient_boosting":
            params = self._params.gradient_boosting.to_dict() if hasattr(
                self._params.gradient_boosting, "to_dict") else self._params._data["gradient_boosting"]
            return GradientBoostingClassifier(**params, random_state=42)

        elif self._model_type == "random_forest":
            params = self._params.random_forest.to_dict() if hasattr(
                self._params.random_forest, "to_dict") else self._params._data["random_forest"]
            return RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        elif self._model_type == "ensemble":
            gb_params = self._params._data["gradient_boosting"]
            rf_params = self._params._data["random_forest"]
            gb = GradientBoostingClassifier(**gb_params, random_state=42)
            rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
            return VotingClassifier(
                estimators=[("gb", gb), ("rf", rf)],
                voting="soft", weights=[0.6, 0.4])
        else:
            raise ValueError(f"未知模型类型: {self._model_type}")

    # ==============================================================
    # 训练
    # ==============================================================
    def train(self, features: pd.DataFrame, target_col: str = None) -> dict:
        """
        训练模型
        features: FeatureEngine.compute_all() 的输出（含目标列）
        target_col: 目标列名（默认 target_dir_1）
        """
        target_col = target_col or "target_dir_1"

        if features.empty:
            raise ValueError(f"特征表为空，数据不足无法训练（需要至少 500 条有效样本）")
        if target_col not in features.columns:
            raise ValueError(f"目标列 {target_col} 不在特征表中")

        # 准备数据
        exclude = [c for c in features.columns
                   if c.startswith("fwd_") or c.startswith("target_") or c == "close"]
        feature_cols = [c for c in features.columns if c not in exclude]

        df = features[feature_cols + [target_col]].dropna()
        if len(df) < 500:
            raise ValueError(f"样本不足: {len(df)}，至少需要 500 条")

        X = df[feature_cols]
        y = df[target_col].astype(int)

        # 特征选择
        X, selected_features = self._select_features(X, y)
        self.feature_names = selected_features

        logger.info(f"训练数据: {X.shape[0]} 样本, {X.shape[1]} 特征, "
                    f"正样本比例={y.mean():.3f}")

        # 时序交叉验证
        train_cfg = self._train_cfg
        purge = train_cfg.purge_gap_hours if hasattr(train_cfg, "purge_gap_hours") else train_cfg._data.get("purge_gap_hours", 6)
        embargo = train_cfg.embargo_hours if hasattr(train_cfg, "embargo_hours") else train_cfg._data.get("embargo_hours", 3)
        n_splits = train_cfg.n_splits if hasattr(train_cfg, "n_splits") else train_cfg._data.get("n_splits", 5)

        cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_gap=purge, embargo=embargo)

        cv_metrics = []
        ic_values = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns, index=X_train.index)
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                columns=X_val.columns, index=X_val.index)

            # 训练
            model = self._build_model()
            model.fit(X_train_scaled, y_train)

            # 评估
            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]

            metrics = {
                "fold": fold,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
            }

            try:
                metrics["log_loss"] = log_loss(y_val, y_proba)
            except Exception:
                metrics["log_loss"] = None

            # IC (信息系数)
            if "fwd_ret_1" in features.columns:
                fwd = features.loc[X_val.index, "fwd_ret_1"].dropna()
                common = fwd.index.intersection(pd.Index(range(len(y_proba))))
                if len(fwd) == len(y_proba):
                    ic, _ = spearmanr(y_proba, fwd.values)
                    metrics["ic"] = ic if not np.isnan(ic) else 0
                    ic_values.append(ic if not np.isnan(ic) else 0)

            cv_metrics.append(metrics)
            logger.info(
                f"  Fold {fold}: acc={metrics['accuracy']:.3f} "
                f"prec={metrics['precision']:.3f} f1={metrics['f1']:.3f}")

        # 用全部数据训练最终模型
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        self.model = self._build_model()
        self.model.fit(X_scaled, y)

        # 特征重要性
        self._compute_importance(X.columns.tolist())

        # 生成 model_id
        self.model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 汇总
        avg_metrics = {
            k: np.mean([m[k] for m in cv_metrics if m.get(k) is not None])
            for k in ["accuracy", "precision", "recall", "f1", "log_loss"]
        }
        if ic_values:
            avg_metrics["ic_mean"] = np.mean(ic_values)
            avg_metrics["ic_std"] = np.std(ic_values)
            avg_metrics["icir"] = np.mean(ic_values) / (np.std(ic_values) + 1e-10)

        logger.info(f"训练完成 | {self.model_id} | 平均指标: {avg_metrics}")

        # 保存
        self.save()

        return {
            "model_id": self.model_id,
            "cv_metrics": cv_metrics,
            "avg_metrics": avg_metrics,
            "n_features": len(self.feature_names),
            "top_features": dict(sorted(
                self.feature_importances.items(),
                key=lambda x: x[1], reverse=True)[:20]),
        }

    # ==============================================================
    # 预测
    # ==============================================================
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        返回 DataFrame 包含 probability, signal, strength 列
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")

        # 对齐特征
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            logger.warning(f"缺少特征: {missing}，使用 0 填充")
            for col in missing:
                features[col] = 0

        X = features[self.feature_names].copy()
        X = X.fillna(0)

        # 标准化
        X_scaled = pd.DataFrame(
            self.scaler.transform(X), columns=X.columns, index=X.index)

        # 预测概率
        proba = self.model.predict_proba(X_scaled)[:, 1]

        # 转化为信号
        thresholds = self._signal_thresholds
        buy_th = thresholds.buy if hasattr(thresholds, "buy") else thresholds._data.get("buy", 0.55)
        sell_th = thresholds.sell if hasattr(thresholds, "sell") else thresholds._data.get("sell", 0.55)
        strong_buy = thresholds.strong_buy if hasattr(thresholds, "strong_buy") else thresholds._data.get("strong_buy", 0.65)
        strong_sell = thresholds.strong_sell if hasattr(thresholds, "strong_sell") else thresholds._data.get("strong_sell", 0.65)

        result = pd.DataFrame(index=X.index)
        result["probability"] = proba
        result["signal"] = "HOLD"
        result["strength"] = 0.0

        # BUY
        buy_mask = proba >= buy_th
        result.loc[buy_mask, "signal"] = "BUY"
        result.loc[buy_mask, "strength"] = (proba[buy_mask] - 0.5) * 2  # 归一到 0~1

        # SELL
        sell_mask = proba <= (1 - sell_th)
        result.loc[sell_mask, "signal"] = "SELL"
        result.loc[sell_mask, "strength"] = ((1 - proba[sell_mask]) - 0.5) * 2

        # 强信号
        result.loc[proba >= strong_buy, "signal"] = "STRONG_BUY"
        result.loc[proba <= (1 - strong_sell), "signal"] = "STRONG_SELL"

        return result

    # ==============================================================
    # 特征选择
    # ==============================================================
    def _select_features(self, X: pd.DataFrame,
                         y: pd.Series) -> tuple[pd.DataFrame, list[str]]:
        fs_cfg = self._feature_selection
        max_feat = fs_cfg.max_features if hasattr(fs_cfg, "max_features") else fs_cfg._data.get("max_features", 50)
        method = fs_cfg.method if hasattr(fs_cfg, "method") else fs_cfg._data.get("method", "importance")

        if method == "importance":
            # 用小模型做初筛
            quick_model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42)
            X_filled = X.fillna(0)
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_filled), columns=X.columns)
            quick_model.fit(X_scaled, y)

            importances = pd.Series(
                quick_model.feature_importances_, index=X.columns)
            selected = importances.nlargest(max_feat).index.tolist()

        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif
            X_filled = X.fillna(0)
            mi = mutual_info_classif(X_filled, y, random_state=42)
            mi_series = pd.Series(mi, index=X.columns)
            selected = mi_series.nlargest(max_feat).index.tolist()

        else:
            selected = X.columns.tolist()[:max_feat]

        logger.info(f"特征选择: {len(X.columns)} -> {len(selected)} 个")
        return X[selected], selected

    def _compute_importance(self, feature_names: list[str]):
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances = dict(
                zip(feature_names, self.model.feature_importances_))
        elif hasattr(self.model, "estimators_"):
            # VotingClassifier
            importances = np.zeros(len(feature_names))
            for name, est in self.model.named_estimators_.items():
                if hasattr(est, "feature_importances_"):
                    importances += est.feature_importances_
            importances /= len(self.model.estimators_)
            self.feature_importances = dict(zip(feature_names, importances))

    # ==============================================================
    # 持久化
    # ==============================================================
    def save(self):
        path = self.model_dir / f"{self.model_id}.pkl"
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "model_id": self.model_id,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"模型已保存: {path}")

    def load(self, model_id: str = None, path: str = None):
        if path is None:
            if model_id:
                path = self.model_dir / f"{model_id}.pkl"
            else:
                # 加载最新的
                models = sorted(self.model_dir.glob("model_*.pkl"))
                if not models:
                    raise FileNotFoundError("没有找到已训练的模型")
                path = models[-1]

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.model = state["model"]
        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self.feature_importances = state["feature_importances"]
        self.model_id = state["model_id"]
        logger.info(f"模型已加载: {self.model_id} ({len(self.feature_names)} 特征)")
