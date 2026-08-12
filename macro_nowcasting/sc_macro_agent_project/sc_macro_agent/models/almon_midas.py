"""
Exponential Almon Lag MIDAS 模型。

按申报书公式 (1)(2) 实现真正的 Almon 权重函数：
    y_t = β0 + Σ_i β_i · B(L^(1/m); θ_i) · x_{i,t}
    B(k; θ) = exp(θ1·k + θ2·k²) / Σ_j exp(θ1·j + θ2·j²)

与现有 RidgeMIDAS/ElasticMIDAS 的区别：
- 不依赖月度→季度聚合，直接使用原始月度滞后值
- 滞后权重由双参数 Almon 函数决定，而非自由回归系数
- 参数极少（每指标仅 3 个参数），小样本友好
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from .base import BaseForecastModel
from ..exceptions import ModelTrainingError


def _compute_almon_weights(theta1: float, theta2: float, n_lags: int) -> np.ndarray:
    """在 log 空间数值稳定地计算 Almon 归一化权重。

    w_k = exp(θ1·k + θ2·k²) / Σ_j exp(θ1·j + θ2·j²)

    Args:
        theta1: 一次项系数
        theta2: 二次项系数
        n_lags: 滞后阶数 K

    Returns:
        长度为 n_lags 的权重向量，求和 = 1.0
    """
    k = np.arange(n_lags, dtype=float)
    z = theta1 * k + theta2 * (k ** 2)
    z_max = z.max()
    # log-sum-exp 稳定化：减去最大值后再 exp
    w = np.exp(z - z_max)
    w_sum = w.sum()
    if w_sum < 1e-15:
        # 数值下溢：回退为均匀权重
        return np.ones(n_lags, dtype=float) / n_lags
    return w / w_sum


def _parse_midas_columns(
    feature_names: List[str],
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], List[str]]:
    """解析 midas__ 前缀列，提取指标组和滞后阶数。

    Returns:
        groups: {group_key: [k0, k1, ...]} — 每组对应的滞后阶数列表
        col_indices: {group_key: [col_idx_0, col_idx_1, ...]} — 每组在 X 中的列索引
        sorted_names: 所有 midas 列名（保持顺序）
    """
    midas_pattern = re.compile(r"^midas__(.+)__L(\d+)$")

    groups: Dict[str, List[int]] = {}
    col_indices: Dict[str, List[int]] = {}
    sorted_names: List[str] = []

    for idx, name in enumerate(feature_names):
        m = midas_pattern.match(name)
        if not m:
            continue
        group_key = m.group(1)  # e.g. "四川省__PMI"
        lag_k = int(m.group(2))
        if group_key not in groups:
            groups[group_key] = []
            col_indices[group_key] = []
        groups[group_key].append(lag_k)
        col_indices[group_key].append(idx)
        sorted_names.append(name)

    # 每组内按 k 排序
    for g in groups:
        order = np.argsort(groups[g])
        groups[g] = [groups[g][i] for i in order]
        col_indices[g] = [col_indices[g][i] for i in order]

    return groups, col_indices, sorted_names


class AlmonMIDASModel(BaseForecastModel):
    """Exponential Almon Lag MIDAS 模型。

    每个指标组（如 "四川省__PMI"）一套参数 (θ1, θ2, β)，
    另加一个全局截距 β0。

    权重函数 B(k; θ) = exp(θ1·k + θ2·k²) / Σ_j exp(θ1·j + θ2·j²)
    对应申报书公式 (2)。
    """

    model_name = "almon_midas"
    uses_midas_lags = True  # 需要 midas__ 前缀列用于 Almon 权重计算

    # 多点重启的 θ 初值组合
    _INIT_THETA_PAIRS: List[Tuple[float, float]] = [
        (0.0, -0.1),
        (-0.5, -0.05),
        (0.5, -0.3),
        (0.0, 0.0),
        (-1.0, -0.5),
    ]

    def __init__(
        self,
        theta_l2: float = 0.1,
        theta1_bounds: Tuple[float, float] = (-2.0, 2.0),
        theta2_bounds: Tuple[float, float] = (-1.5, 0.5),
    ) -> None:
        super().__init__()
        self.theta_l2 = theta_l2
        self.theta1_bounds = theta1_bounds
        self.theta2_bounds = theta2_bounds
        self.scaler = StandardScaler()
        self._groups: Dict[str, List[int]] = {}         # group_key -> [k0, k1, ...]
        self._col_indices: Dict[str, List[int]] = {}    # group_key -> [col_idx, ...]
        self._group_keys: List[str] = []                 # ordered group keys
        self._group_n_lags: Dict[str, int] = {}          # group_key -> K

        # 拟合结果
        self._beta0: float = 0.0
        self._theta1: Dict[str, float] = {}
        self._theta2: Dict[str, float] = {}
        self._beta: Dict[str, float] = {}
        self._weights: Dict[str, np.ndarray] = {}        # group_key -> normalized weight vector
        self._sse: float = 0.0
        self._n_samples: int = 0

    # ------------------------------------------------------------------
    # 核心计算
    # ------------------------------------------------------------------
    def _group_weighted_sum(
        self,
        Xs: np.ndarray,
        theta1: float,
        theta2: float,
        col_idxs: List[int],
        n_lags: int,
    ) -> np.ndarray:
        """计算一个指标组的加权和 Σ_k w_k · x_{t,k}。

        Args:
            Xs: 标准化后的特征矩阵 (n_samples, n_features)
            theta1, theta2: Almon 参数
            col_idxs: 该组在 Xs 中的列索引列表（按 k 排序）
            n_lags: 滞后阶数

        Returns:
            长度为 n_samples 的加权和向量
        """
        if not col_idxs:
            return np.zeros(Xs.shape[0], dtype=float)
        weights = _compute_almon_weights(theta1, theta2, n_lags)
        # col_idxs 可能比 n_lags 短（有些 lag 缺失），取实际存在的
        actual_k = min(len(col_idxs), n_lags)
        result = np.zeros(Xs.shape[0], dtype=float)
        for j in range(actual_k):
            result += weights[j] * Xs[:, col_idxs[j]]
        return result

    def _predict_internal(
        self,
        Xs: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """给定参数向量，计算预测值。

        params = [β0, θ1_1, θ2_1, β_1, ..., θ1_G, θ2_G, β_G]
        """
        pred = np.full(Xs.shape[0], params[0], dtype=float)  # β0
        for g_idx, g_key in enumerate(self._group_keys):
            offset = 1 + 3 * g_idx
            theta1 = params[offset]
            theta2 = params[offset + 1]
            beta_g = params[offset + 2]
            n_lags = self._group_n_lags[g_key]
            weighted = self._group_weighted_sum(
                Xs, theta1, theta2, self._col_indices[g_key], n_lags
            )
            pred += beta_g * weighted
        return pred

    def _sse_objective(self, params: np.ndarray, Xs: np.ndarray, y: np.ndarray) -> float:
        """目标函数：SSE/n + theta_l2 * Σ (θ1² + θ2²)。

        SSE 除以 n 使惩罚强度不随样本量漂移。
        """
        pred = self._predict_internal(Xs, params)
        residuals = y - pred
        mse = float(np.sum(residuals ** 2)) / max(len(y), 1)
        # L2 penalty on all theta parameters
        theta_penalty = 0.0
        if self.theta_l2 > 0:
            for g_idx in range(len(self._group_keys)):
                offset = 1 + 3 * g_idx
                theta_penalty += params[offset] ** 2 + params[offset + 1] ** 2
        return mse + self.theta_l2 * theta_penalty

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AlmonMIDASModel":
        self.feature_names = list(X.columns)
        y_arr = np.asarray(y, dtype=float)

        # 1. 解析 midas 列
        groups, col_indices, _ = _parse_midas_columns(self.feature_names)
        if not groups:
            raise ModelTrainingError(
                "未找到任何 midas__ 前缀列，AlmonMIDASModel 需要月度滞后特征。"
                " 请确认 FeatureConfig.build_midas_lags=True 且关键词匹配到指标。"
            )

        self._groups = groups
        self._col_indices = col_indices
        self._group_keys = sorted(groups.keys())
        for g_key in self._group_keys:
            self._group_n_lags[g_key] = len(groups[g_key])

        # 2. 标准化 midas 列
        midas_col_indices_all = sorted(set(
            idx for idxs in col_indices.values() for idx in idxs
        ))
        X_midas = X.iloc[:, midas_col_indices_all].to_numpy(dtype=float)

        if len(X_midas) < 2:
            raise ModelTrainingError("AlmonMIDASModel 需要至少 2 个训练样本")

        Xs_midas = self.scaler.fit_transform(X_midas)

        # 构建完整 Xs（仅 midas 列）
        Xs = np.zeros((len(X), len(self.feature_names)), dtype=float)
        for i, orig_idx in enumerate(midas_col_indices_all):
            Xs[:, orig_idx] = Xs_midas[:, i]

        # 3. 多点重启优化
        G = len(self._group_keys)
        n_params = 1 + 3 * G  # β0 + per-group (θ1, θ2, β)
        self._n_samples = len(y_arr)

        # 参数边界：从 config 读取
        bounds = [(None, None)]  # β0 无界
        for _ in range(G):
            bounds.extend([
                self.theta1_bounds,
                self.theta2_bounds,
                (None, None),  # β 无界
            ])

        best_params: Optional[np.ndarray] = None
        best_sse: float = float("inf")

        for theta1_init, theta2_init in self._INIT_THETA_PAIRS:
            # 构建初始参数
            x0 = np.zeros(n_params, dtype=float)
            x0[0] = float(np.mean(y_arr))  # β0 从 y 均值开始
            for g_idx in range(G):
                offset = 1 + 3 * g_idx
                x0[offset] = theta1_init
                x0[offset + 1] = theta2_init
                x0[offset + 2] = 0.1  # β 从小正数开始

            try:
                result = minimize(
                    fun=self._sse_objective,
                    x0=x0,
                    args=(Xs, y_arr),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 5000, "ftol": 1e-12},
                )
                if result.success or result.fun < best_sse:
                    if result.fun < best_sse:
                        best_sse = float(result.fun)
                        best_params = result.x.copy()
            except Exception:
                continue

        if best_params is None:
            raise ModelTrainingError(
                "AlmonMIDASModel 优化失败：所有初始点均未收敛。"
                " 请检查 midas 特征列是否存在全零或常数问题。"
            )

        # 4. 保存拟合结果
        self._beta0 = float(best_params[0])
        self._sse = best_sse

        for g_idx, g_key in enumerate(self._group_keys):
            offset = 1 + 3 * g_idx
            self._theta1[g_key] = float(best_params[offset])
            self._theta2[g_key] = float(best_params[offset + 1])
            self._beta[g_key] = float(best_params[offset + 2])
            self._weights[g_key] = _compute_almon_weights(
                self._theta1[g_key],
                self._theta2[g_key],
                self._group_n_lags[g_key],
            )

        # 5. 计算训练集上的预测和残差
        final_pred = self._predict_internal(Xs, best_params)
        self.train_predictions_ = final_pred
        self.train_residuals_ = y_arr - final_pred
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ModelTrainingError("模型尚未拟合，请先调用 fit()")

        # 提取并标准化 midas 列
        midas_col_indices_all = sorted(set(
            idx for idxs in self._col_indices.values() for idx in idxs
        ))

        # 检查列对齐
        expected = set(self.feature_names)
        actual = set(X.columns)
        if not expected.issubset(actual):
            missing = expected - actual
            raise ModelTrainingError(
                f"预测数据缺少 midas 列: {missing}"
            )

        X_aligned = X[self.feature_names]
        X_midas = X_aligned.iloc[:, midas_col_indices_all].to_numpy(dtype=float)
        Xs_midas = self.scaler.transform(X_midas)

        Xs = np.zeros((len(X), len(self.feature_names)), dtype=float)
        for i, orig_idx in enumerate(midas_col_indices_all):
            Xs[:, orig_idx] = Xs_midas[:, i]

        # 重建参数向量
        G = len(self._group_keys)
        params = np.zeros(1 + 3 * G, dtype=float)
        params[0] = self._beta0
        for g_idx, g_key in enumerate(self._group_keys):
            offset = 1 + 3 * g_idx
            params[offset] = self._theta1[g_key]
            params[offset + 1] = self._theta2[g_key]
            params[offset + 2] = self._beta[g_key]

        return self._predict_internal(Xs, params)

    # ------------------------------------------------------------------
    # 诊断与可视化
    # ------------------------------------------------------------------
    def get_weight_curves(self) -> List[Dict[str, Any]]:
        """获取每个指标组的 Almon 权重曲线。

        Returns:
            [{"indicator": str, "theta1": float, "theta2": float,
              "beta": float, "weights": List[float],
              "at_boundary": bool, "weight_entropy": float}]
            weight_entropy: 1.0 = 完全均匀，0.0 = 完全集中于单一滞后
        """
        curves: List[Dict[str, Any]] = []
        for g_key in self._group_keys:
            t1 = self._theta1.get(g_key, 0.0)
            t2 = self._theta2.get(g_key, 0.0)
            w = self._weights.get(g_key, np.array([]))

            # at_boundary check (1e-3 tolerance)
            at_bound = False
            if abs(t1 - self.theta1_bounds[0]) < 1e-3 or abs(t1 - self.theta1_bounds[1]) < 1e-3:
                at_bound = True
            if abs(t2 - self.theta2_bounds[0]) < 1e-3 or abs(t2 - self.theta2_bounds[1]) < 1e-3:
                at_bound = True

            # weight_entropy: normalized entropy H / H_max
            w_arr = np.asarray(w, dtype=float)
            if len(w_arr) > 1 and w_arr.sum() > 1e-15:
                w_norm = w_arr / w_arr.sum()
                eps = 1e-15
                entropy = -np.sum(w_norm * np.log(w_norm + eps))
                max_entropy = np.log(len(w_norm))
                weight_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
            else:
                weight_entropy = 0.0

            curves.append({
                "indicator": g_key,
                "theta1": t1,
                "theta2": t2,
                "beta": self._beta.get(g_key, 0.0),
                "weights": w_arr.tolist(),
                "at_boundary": at_bound,
                "weight_entropy": weight_entropy,
            })
        return curves

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_fitted": self.is_fitted,
            "n_groups": len(self._group_keys),
            "n_params": 1 + 3 * len(self._group_keys),
            "sse": self._sse,
            "beta0": self._beta0,
            "groups": [
                {
                    "indicator": g_key,
                    "theta1": self._theta1.get(g_key, 0.0),
                    "theta2": self._theta2.get(g_key, 0.0),
                    "beta": self._beta.get(g_key, 0.0),
                    "n_lags": self._group_n_lags.get(g_key, 0),
                    "weights": self._weights.get(g_key, np.array([])).tolist(),
                }
                for g_key in self._group_keys
            ],
        }

    def get_feature_importance(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """基于权重 × β 的贡献排序。"""
        items: List[Dict[str, Any]] = []
        for g_key in self._group_keys:
            w = self._weights.get(g_key, np.array([]))
            beta = self._beta.get(g_key, 0.0)
            for k_idx, wk in enumerate(w):
                col_name = f"midas__{g_key}__L{k_idx}"
                items.append({
                    "feature": col_name,
                    "coefficient": float(beta * wk),
                    "abs_coefficient": float(abs(beta * wk)),
                    "normalized_importance": 0.0,
                    "group": g_key,
                    "lag": k_idx,
                    "weight": float(wk),
                    "group_beta": float(beta),
                })
        # 按绝对贡献排序
        items.sort(key=lambda x: x["abs_coefficient"], reverse=True)
        total = sum(it["abs_coefficient"] for it in items) or 1.0
        for it in items:
            it["normalized_importance"] = it["abs_coefficient"] / total
        return items[:top_n]
