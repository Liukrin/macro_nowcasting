"""
DFM / PCA 因子提取器。

严格说这里更像“工程化 DFM-style 因子抽取器”：
- 先把月度长表 pivot 成 wide
- 再做标准化 + PCA
- 最后把月度因子重新聚合到季度

这样对中期报告和项目叙事已经足够，而且实现稳定
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..logging_utils import get_logger
from ..utils import quarter_end, safe_last, safe_mean, safe_std, safe_trend


@dataclass
class DFMArtifacts:
    monthly_factor_frame: pd.DataFrame
    quarterly_factor_frame: pd.DataFrame
    loadings_frame: pd.DataFrame
    explained_variance_ratio: List[float]


class DFMModel:
    def __init__(self, n_factors: int = 3, standardize: bool = True) -> None:
        self.n_factors = n_factors
        self.standardize = standardize
        self.logger = get_logger("sc_macro_agent.models.dfm")
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_factors)
        self.input_columns: List[str] = []
        self.loadings_: Optional[pd.DataFrame] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.is_fitted = False

    def build_monthly_wide(self, monthly_local_df: pd.DataFrame, monthly_national_df: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for source_name, df in [("local", monthly_local_df), ("national", monthly_national_df)]:
            if df.empty:
                continue
            cur = df.copy()
            cur["date"] = pd.to_datetime(cur["date"], errors="coerce")
            cur["col_name"] = cur["region"].astype(str) + "__" + cur["indicator_name"].astype(str)
            wide = (
                cur.pivot_table(index="date", columns="col_name", values="indicator_value", aggfunc="last")
                .sort_index()
                .reset_index()
            )
            frames.append(wide)

        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="date", how="outer")

        merged = merged.sort_values("date").reset_index(drop=True)
        return merged

    def fit_transform(self, monthly_local_df: pd.DataFrame, monthly_national_df: pd.DataFrame) -> DFMArtifacts:
        wide = self.build_monthly_wide(monthly_local_df, monthly_national_df)
        if wide.empty:
            return DFMArtifacts(
                monthly_factor_frame=pd.DataFrame(),
                quarterly_factor_frame=pd.DataFrame(),
                loadings_frame=pd.DataFrame(),
                explained_variance_ratio=[],
            )

        x_cols = [c for c in wide.columns if c != "date"]
        X = wide[x_cols].copy()

        # 小心：真实数据有的列只有一期
        X = X.ffill().bfill()
        for col in X.columns:
            if X[col].isna().all():
                X[col] = 0.0
            else:
                X[col] = X[col].fillna(X[col].median())

        self.input_columns = list(X.columns)
        values = X.to_numpy(dtype=float)
        if self.standardize:
            values = self.scaler.fit_transform(values)

        n_components = min(self.n_factors, values.shape[0], values.shape[1])
        if n_components <= 0:
            return DFMArtifacts(
                monthly_factor_frame=pd.DataFrame(),
                quarterly_factor_frame=pd.DataFrame(),
                loadings_frame=pd.DataFrame(),
                explained_variance_ratio=[],
            )

        self.pca = PCA(n_components=n_components)
        factors = self.pca.fit_transform(values)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_

        monthly_factor_frame = wide[["date"]].copy()
        for i in range(n_components):
            monthly_factor_frame[f"dfm_factor_{i+1}"] = factors[:, i]
        monthly_factor_frame["quarter_end"] = quarter_end(monthly_factor_frame["date"])

        rows = []
        factor_cols = [c for c in monthly_factor_frame.columns if c.startswith("dfm_factor_")]
        for q_end, grp in monthly_factor_frame.groupby("quarter_end"):
            grp = grp.sort_values("date")
            row = {"quarter_end": q_end}
            for col in factor_cols:
                row[f"{col}_mean"] = safe_mean(grp[col])
                row[f"{col}_last"] = safe_last(grp[col])
                row[f"{col}_std"] = safe_std(grp[col])
                row[f"{col}_trend"] = safe_trend(grp[col])
            rows.append(row)
        quarterly_factor_frame = pd.DataFrame(rows).sort_values("quarter_end").reset_index(drop=True)

        loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.input_columns,
            columns=[f"factor_{i+1}" for i in range(n_components)],
        )
        self.loadings_ = loadings
        self.is_fitted = True

        return DFMArtifacts(
            monthly_factor_frame=monthly_factor_frame,
            quarterly_factor_frame=quarterly_factor_frame,
            loadings_frame=loadings.reset_index().rename(columns={"index": "feature"}),
            explained_variance_ratio=self.explained_variance_ratio_.tolist(),
        )

    def top_loadings(self, top_n: int = 8) -> List[Dict[str, Any]]:
        if self.loadings_ is None or self.loadings_.empty:
            return []
        results: List[Dict[str, Any]] = []
        for factor in self.loadings_.columns:
            ser = self.loadings_[factor].abs().sort_values(ascending=False).head(top_n)
            results.append({
                "factor_name": factor,
                "top_loadings": [
                    {
                        "feature": idx,
                        "loading": float(self.loadings_.loc[idx, factor]),
                        "abs_loading": float(val),
                    }
                    for idx, val in ser.items()
                ],
            })
        return results

    def summary(self) -> Dict[str, Any]:
        if self.explained_variance_ratio_ is None:
            return {"is_fitted": False, "n_factors": 0}
        return {
            "is_fitted": self.is_fitted,
            "n_factors": int(len(self.explained_variance_ratio_)),
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(np.cumsum(self.explained_variance_ratio_)[-1]),
            "top_loadings": self.top_loadings(),
        }
