"""
LightGBM Alpha 模型 — 基于因子扫描结果构建

特点:
1. 使用因子扫描确认的 Top 12 因子 (去冗余)
2. LightGBM 替代 sklearn GBM (速度快 10x+)
3. Purged + Embargo 时序交叉验证
4. 支持 predict_proba 接入信号过滤
5. 自动降级到 sklearn 如果 lightgbm 未安装

核心因子 (factor_scan_report.json):
  均值回归: ma_dev_240, zscore_240, bb_position_240, close_vs_sma_100
  震荡指标: rsi_14, mfi_14, keltner_position
  短期反转: ret_5, mom_accel_5
  微观结构: amihud_60, taker_buy_ratio_120, price_vol_corr_120
"""
from __future__ import annotations

import json
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.logger import get_logger

logger = get_logger("ml_lightgbm")

# 因子扫描确认的核心因子 (去冗余后)
CORE_FEATURES = [
    # 均值回归 (IC: -0.36 ~ -0.43, ICIR: -1.5 ~ -2.0)
    "ma_dev_240", "zscore_240", "bb_position_240", "close_vs_sma_100",
    # 震荡指标 (IC: -0.13 ~ -0.20)
    "rsi_14", "mfi_14", "keltner_position",
    # 短期反转 (IC: -0.13, t: -7.8)
    "ret_5", "mom_accel_5",
    # 微观结构 + 量价 (IC: 0.11 ~ 0.28)
    "amihud_60", "taker_buy_ratio_120", "price_vol_corr_120",
    # 波动率 (IC: 0.39)
    "natr_20", "natr_60",
    # 趋势确认
    "adx_14",
]

# 扩展因子 (可选，如果数据可用)
EXTENDED_FEATURES = [
    "ma_dev_120", "ma_dev_480", "vwap_dev_240",
    "zscore_120", "bb_width_240",
    "rsi_21", "cci_20", "williams_r", "stoch_d",
    "ret_10", "ret_20",
    "rel_volume_20", "rel_volume_60",
    "obv_change_120",
    "natr_120", "natr_240",
]


def _try_import_lgbm():
    """
    尝试导入 LightGBM，失败则降级到 sklearn。

    除了 ImportError (未安装) 外, 还需捕获 OSError —— WSL/Debian 上 `pip install lightgbm`
    成功但系统缺少 `libgomp.so.1` 时, import 会抛 OSError。本地 agent 反馈修复 (2026-04-19)。
    """
    try:
        import lightgbm as lgb
        return lgb, True
    except ImportError:
        logger.warning("LightGBM 未安装，降级到 sklearn GradientBoosting")
        return None, False
    except OSError as e:
        logger.warning(
            f"LightGBM 原生库加载失败 ({e}), 降级到 sklearn。"
            f"若需使用 LightGBM, 在 Debian/WSL 上运行: bash scripts/install_debian_test_deps.sh"
        )
        return None, False


class LightGBMAlphaModel:
    """
    LightGBM Alpha 模型

    用法:
        model = LightGBMAlphaModel()
        report = model.train(features_df, target_col="forward_return_6")
        predictions = model.predict(new_features)
        model.save("models/lgbm_v1.pkl")
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 6,
                 embargo: int = 3, use_core_only: bool = False):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo = embargo
        self.use_core_only = use_core_only

        self.model = None
        self.feature_names: list[str] = []
        self.model_id: str = ""
        self.train_report: dict = {}

        self._lgb, self._use_lgb = _try_import_lgbm()

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        """选择可用因子"""
        candidates = CORE_FEATURES if self.use_core_only else CORE_FEATURES + EXTENDED_FEATURES
        available = [f for f in candidates if f in df.columns]
        if len(available) < 5:
            logger.warning(f"仅找到 {len(available)} 个核心因子，尝试使用所有数值列")
            exclude = {"open_time", "close_time", "open", "high", "low", "close",
                       "volume", "quote_volume", "trades_count", "symbol", "interval",
                       "taker_buy_base", "taker_buy_quote", "is_closed",
                       "label_binary", "label_ternary", "forward_return"}
            available = [c for c in df.columns
                         if c not in exclude and df[c].dtype in ["float64", "float32", "int64"]]
        logger.info(f"选择 {len(available)} 个因子")
        return available

    def _make_target(self, df: pd.DataFrame, target_col: str = None,
                     forward_bars: int = 6) -> pd.Series:
        """生成目标变量: 二分类 (涨/跌)"""
        if target_col and target_col in df.columns:
            target = df[target_col]
        else:
            close = pd.to_numeric(df["close"], errors="coerce")
            fwd = close.shift(-forward_bars) / close - 1
            target = (fwd > 0).astype(int)
        return target

    def _purged_cv_split(self, n: int):
        """Purged + Embargo 时序交叉验证"""
        fold_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end + self.purge_gap
            val_end = val_start + fold_size

            if val_end > n:
                break

            train_idx = list(range(0, train_end))
            val_idx = list(range(val_start + self.embargo, min(val_end, n)))

            if len(train_idx) < 100 or len(val_idx) < 20:
                continue
            yield train_idx, val_idx

    def train(self, df: pd.DataFrame, target_col: str = None,
              forward_bars: int = 6) -> dict:
        """
        训练模型

        Args:
            df: 含因子列的 DataFrame
            target_col: 目标列名 (None 则自动生成)
            forward_bars: 前瞻 bar 数
        """
        t0 = time.time()
        self.feature_names = self._select_features(df)
        target = self._make_target(df, target_col, forward_bars)

        # 对齐
        valid_mask = target.notna() & df[self.feature_names].notna().all(axis=1)
        X = df.loc[valid_mask, self.feature_names].values.astype(np.float32)
        y = target[valid_mask].values.astype(int)

        logger.info(f"训练样本: {len(X)}, 因子数: {len(self.feature_names)}")

        # 交叉验证
        cv_scores = []
        cv_ics = []
        best_model = None
        best_score = -np.inf

        for fold_i, (train_idx, val_idx) in enumerate(self._purged_cv_split(len(X))):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = self._build_model()
            model.fit(X_train, y_train)

            # 评估
            proba = model.predict_proba(X_val)[:, 1]
            from sklearn.metrics import accuracy_score, roc_auc_score
            acc = accuracy_score(y_val, (proba > 0.5).astype(int))

            try:
                auc = roc_auc_score(y_val, proba)
            except Exception:
                auc = 0.5

            # IC
            ic, _ = spearmanr(proba, y_val)
            ic = float(ic) if not np.isnan(ic) else 0.0

            cv_scores.append({"fold": fold_i, "accuracy": acc, "auc": auc, "ic": ic})
            cv_ics.append(ic)

            if auc > best_score:
                best_score = auc
                best_model = model

            logger.info(f"  Fold {fold_i}: AUC={auc:.4f} ACC={acc:.4f} IC={ic:.4f}")

        # 最终模型: 用全量数据训练
        self.model = self._build_model()
        self.model.fit(X, y)

        # 特征重要性
        if hasattr(self.model, "feature_importances_"):
            importances = dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            importances = {}

        top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:15])

        self.model_id = f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elapsed = time.time() - t0

        self.train_report = {
            "model_id": self.model_id,
            "n_samples": len(X),
            "n_features": len(self.feature_names),
            "cv_folds": len(cv_scores),
            "avg_auc": np.mean([s["auc"] for s in cv_scores]),
            "avg_ic": np.mean(cv_ics),
            "std_ic": np.std(cv_ics),
            "icir": np.mean(cv_ics) / (np.std(cv_ics) + 1e-10),
            "cv_scores": cv_scores,
            "top_features": top_features,
            "elapsed_seconds": round(elapsed, 1),
            "engine": "lightgbm" if self._use_lgb else "sklearn",
        }

        logger.info(f"训练完成: {self.model_id} | AUC={self.train_report['avg_auc']:.4f} "
                     f"| IC={self.train_report['avg_ic']:.4f} | {elapsed:.1f}s")
        return self.train_report

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")

        available = [f for f in self.feature_names if f in df.columns]
        if len(available) < len(self.feature_names):
            missing = set(self.feature_names) - set(available)
            logger.warning(f"缺少因子: {missing}")

        X = df[available].fillna(0).values.astype(np.float32)

        # 补齐缺失列
        if len(available) < len(self.feature_names):
            X_full = np.zeros((len(X), len(self.feature_names)), dtype=np.float32)
            for i, fname in enumerate(self.feature_names):
                if fname in available:
                    X_full[:, i] = df[fname].fillna(0).values
            X = X_full

        proba = self.model.predict_proba(X)[:, 1]
        preds = pd.DataFrame({
            "probability": proba,
            "signal": pd.cut(proba, bins=[0, 0.4, 0.45, 0.55, 0.6, 1.0],
                             labels=["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]),
            "strength": np.abs(proba - 0.5) * 2,
        }, index=df.index)
        return preds

    def _build_model(self):
        """构建模型实例"""
        if self._use_lgb:
            return self._lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_samples=50,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=-1,
                verbose=-1,
                importance_type="gain",
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=50,
                max_features=0.7,
            )

    def save(self, path: str = "models/lgbm_alpha.pkl"):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "model_id": self.model_id,
            "train_report": self.train_report,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"模型已保存: {path}")

    def load(self, path: str = "models/lgbm_alpha.pkl"):
        """加载模型"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.model_id = data["model_id"]
        self.train_report = data.get("train_report", {})
        logger.info(f"模型已加载: {self.model_id} ({len(self.feature_names)} features)")