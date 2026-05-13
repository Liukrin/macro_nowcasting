"""
特征工程模块

设计目标：
1. 输入是标准化长表
2. 输出是季度级宽表训练面板
3. 既支持直接消费已有 quarterly_panel，也支持从月度长表重新聚。
4. 在小样本场景下主动限制特征数量，避免“列比样本还多很多”
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from ..exceptions import FeatureBuildError
from .feature_registry import FeatureRegistry
from ..logging_utils import get_logger
from ..utils import (
    cap_feature_count,
    ensure_datetime,
    quarter_end,
    safe_change,
    safe_corr,
    safe_last,
    safe_max,
    safe_mean,
    safe_min,
    safe_pct_change,
    safe_std,
    safe_trend,
    sorted_unique,
    winsorize_series,
)


@dataclass
class FeatureArtifacts:
    training_panel: pd.DataFrame
    feature_registry: FeatureRegistry
    feature_columns: List[str]
    target_column: str
    notes: List[str]
    monthly_factor_frame: Optional[pd.DataFrame] = None
    quarterly_factor_frame: Optional[pd.DataFrame] = None


class FeatureEngineer:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.logger = get_logger("sc_macro_agent.feature_engineering")

    @staticmethod
    def _sort_long(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "date" in out.columns:
            out["date"] = ensure_datetime(out["date"])
        return out.sort_values([c for c in ["region", "indicator_name", "date"] if c in out.columns]).reset_index(drop=True)

    @staticmethod
    def _infer_family(indicator_name: str, metadata: Optional[pd.DataFrame] = None) -> str:
        indicator_name = str(indicator_name)
        if metadata is not None and not metadata.empty and "original_name" in metadata.columns:
            hit = metadata[metadata["original_name"] == indicator_name]
            if not hit.empty and "category" in hit.columns:
                val = str(hit.iloc[0]["category"]).strip()
                if val:
                    return val
        if "PMI" in indicator_name:
            return "PMI"
        if "工业" in indicator_name:
            return "工业"
        if "投资" in indicator_name:
            return "投资"
        if "消费" in indicator_name or "零售" in indicator_name:
            return "消费"
        if "出口" in indicator_name or "进口" in indicator_name:
            return "外贸"
        if "CPI" in indicator_name or "PPI" in indicator_name:
            return "价格"
        if "M2" in indicator_name or "融资" in indicator_name:
            return "金融"
        return "其他"

    def _pivot_quarterly_target(self, quarterly_target_df: pd.DataFrame) -> pd.DataFrame:
        if quarterly_target_df.empty:
            raise FeatureBuildError("季度目标数据为空")
        df = quarterly_target_df.copy()
        df["date"] = ensure_datetime(df["date"])
        target_df = df[df["indicator_name"] == self.config.target_indicator].copy()
        if target_df.empty:
            # 宽容一点：只要名字里有 GDP 就先拿来
            target_df = df[df["indicator_name"].astype(str).str.contains("GDP", na=False)].copy()
        if target_df.empty:
            raise FeatureBuildError(f"未找到目标指标: {self.config.target_indicator}")

        out = target_df[["date", "region", "indicator_value"]].rename(columns={"indicator_value": "target_value"})
        out["quarter_end"] = out["date"].dt.to_period("Q").dt.to_timestamp("Q")
        out = out.drop_duplicates(subset=["quarter_end", "region"]).sort_values(["quarter_end", "region"]).reset_index(drop=True)
        return out

    def _pivot_existing_quarterly_panel(
        self,
        quarterly_panel_df: pd.DataFrame,
        registry: FeatureRegistry,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if quarterly_panel_df.empty:
            return pd.DataFrame()

        df = quarterly_panel_df.copy()
        df["date"] = ensure_datetime(df["date"])
        df["quarter_end"] = df["date"].dt.to_period("Q").dt.to_timestamp("Q")
        wide = (
            df.pivot_table(
                index="quarter_end",
                columns=["region", "indicator_name"],
                values="indicator_value",
                aggfunc="last",
            )
            .sort_index()
        )
        if wide.empty:
            return pd.DataFrame()

        wide.columns = [f"{region}__{indicator}" for region, indicator in wide.columns]
        wide = wide.reset_index()

        for col in wide.columns:
            if col == "quarter_end":
                continue
            region, indicator = col.split("__", 1)
            family = self._infer_family(indicator, metadata_df)
            transform = indicator.split("_")[-1] if "_" in indicator else "raw"
            registry.register(
                name=col,
                source_table="quarterly_panel",
                source_indicator=indicator,
                region=region,
                family=family,
                transform=transform,
                frequency="quarterly",
                note="directly from provided quarterly panel",
            )
        return wide

    def _aggregate_indicator_quarter(
        self,
        grp: pd.DataFrame,
        indicator_col: str = "indicator_value",
    ) -> Dict[str, float]:
        values = grp[indicator_col].astype(float).tolist()
        values_non_na = pd.Series(values).dropna()
        first_val = float(values_non_na.iloc[0]) if not values_non_na.empty else 0.0
        last_val = float(values_non_na.iloc[-1]) if not values_non_na.empty else 0.0

        feats = {}
        if self.config.use_mean_agg:
            feats["mean"] = safe_mean(values)
        if self.config.use_last_agg:
            feats["last"] = safe_last(values)
        if self.config.use_std_agg:
            feats["std"] = safe_std(values)
        if self.config.use_min_agg:
            feats["min"] = safe_min(values)
        if self.config.use_max_agg:
            feats["max"] = safe_max(values)
        if self.config.use_trend_agg:
            feats["trend"] = safe_trend(values)
        if self.config.use_qoq_delta_agg:
            feats["delta"] = safe_change(first_val, last_val)
        if self.config.use_range_agg:
            feats["range"] = safe_max(values) - safe_min(values)
        if self.config.use_availability_flags:
            feats["available_months"] = int(values_non_na.shape[0])
        return feats

    def _build_monthly_aggregated_panel(
        self,
        monthly_df: pd.DataFrame,
        source_table_name: str,
        registry: FeatureRegistry,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if monthly_df.empty:
            return pd.DataFrame()

        df = self._sort_long(monthly_df)
        df["quarter_end"] = quarter_end(df["date"])

        rows: List[Dict[str, Any]] = []
        for (region, indicator, q_end), grp in df.groupby(["region", "indicator_name", "quarter_end"]):
            region = str(region)
            indicator = str(indicator)
            grp = grp.sort_values("date")
            feature_values = self._aggregate_indicator_quarter(grp)
            row = {"quarter_end": q_end}
            family = self._infer_family(indicator, metadata_df)
            for suffix, value in feature_values.items():
                col = f"{region}__{indicator}_{suffix}"
                row[col] = value
                registry.register(
                    name=col,
                    source_table=source_table_name,
                    source_indicator=indicator,
                    region=region,
                    family=family,
                    transform=suffix,
                    frequency="quarterly",
                    note="aggregated from monthly long table",
                )
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        agg = pd.DataFrame(rows)
        agg = agg.groupby("quarter_end", as_index=False).first()
        return agg.sort_values("quarter_end").reset_index(drop=True)

    def _merge_frames_on_quarter(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        valid = [f.copy() for f in frames if f is not None and not f.empty]
        if not valid:
            return pd.DataFrame()
        merged = valid[0].copy()
        for frame in valid[1:]:
            merged = merged.merge(frame, on="quarter_end", how="outer")
        merged = merged.sort_values("quarter_end").reset_index(drop=True)
        return merged

    def _add_target_features(self, df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
        out = df.copy()
        if self.config.add_target_lags:
            for lag in self.config.target_lags:
                col = f"target_lag_{lag}"
                out[col] = out["target_value"].shift(lag)
                registry.register(col, "target", self.config.target_indicator, "四川省", "目标", f"lag_{lag}", "quarterly")

        if self.config.add_target_rolling:
            for window in self.config.target_roll_windows:
                mean_col = f"target_roll_mean_{window}"
                std_col = f"target_roll_std_{window}"
                out[mean_col] = out["target_value"].shift(1).rolling(window).mean()
                out[std_col] = out["target_value"].shift(1).rolling(window).std()
                registry.register(mean_col, "target", self.config.target_indicator, "四川省", "目标", f"roll_mean_{window}", "quarterly")
                registry.register(std_col, "target", self.config.target_indicator, "四川省", "目标", f"roll_std_{window}", "quarterly")
        return out

    def _add_feature_lags(self, df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
        if not self.config.add_feature_lags:
            return df
        out = df.copy()
        base_features = [
            c for c in out.columns
            if c not in {"quarter_end", "target_value"} and pd.api.types.is_numeric_dtype(out[c])
        ]
        lagged_columns = {}
        for col in base_features:
            meta = registry.get(col)
            for lag in self.config.feature_lags:
                lag_col = f"{col}__lag{lag}"
                lagged_columns[lag_col] = out[col].shift(lag)
                registry.register(
                    name=lag_col,
                    source_table=meta.source_table if meta else "derived",
                    source_indicator=meta.source_indicator if meta else col,
                    region=meta.region if meta else None,
                    family=meta.family if meta else "派生",
                    transform=f"{meta.transform if meta else 'raw'}_lag{lag}",
                    frequency="quarterly",
                    note=f"quarter lag={lag}",
                )
        if lagged_columns:
            out = pd.concat([out, pd.DataFrame(lagged_columns, index=out.index)], axis=1)
        return out

    def _add_family_features(
        self,
        df: pd.DataFrame,
        registry: FeatureRegistry,
    ) -> pd.DataFrame:
        if not self.config.include_family_features:
            return df
        out = df.copy()
        feature_frame = registry.to_frame()
        if feature_frame.empty:
            return out

        # 选出每个 family 里最核心的前 N 个原始特征，做均值和波动聚合
        for family, sub in feature_frame.groupby("family"):
            family = str(family)
            candidates = [name for name in sub["name"].tolist() if name in out.columns]
            if not candidates:
                continue
            candidates = candidates[: self.config.family_top_n]
            mean_col = f"family__{family}__mean"
            std_col = f"family__{family}__std"
            out[mean_col] = out[candidates].mean(axis=1)
            out[std_col] = out[candidates].std(axis=1)
            registry.register(mean_col, "family", None, None, family, "mean", "quarterly", note="family aggregate mean")
            registry.register(std_col, "family", None, None, family, "std", "quarterly", note="family aggregate std")
        return out

    def _add_interactions(self, df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
        if not self.config.add_interactions:
            return df
        out = df.copy()
        numeric_cols = [
            c for c in out.select_dtypes(include=[np.number]).columns
            if c != "target_value"
        ]
        if len(numeric_cols) < 2:
            return out

        corr_scores: List[Tuple[str, str, float]] = []
        target = out["target_value"]
        for a, b in combinations(numeric_cols[: min(16, len(numeric_cols))], 2):
            score = abs(safe_corr(out[a].tolist(), target.tolist())) + abs(safe_corr(out[b].tolist(), target.tolist()))
            corr_scores.append((a, b, score))
        corr_scores.sort(key=lambda x: x[2], reverse=True)

        for a, b, _ in corr_scores[: self.config.interaction_top_pairs]:
            inter_col = f"interaction__{a}__x__{b}"
            diff_col = f"interaction__{a}__minus__{b}"
            out[inter_col] = out[a] * out[b]
            out[diff_col] = out[a] - out[b]
            registry.register(inter_col, "interaction", None, None, "交互", "product", "quarterly")
            registry.register(diff_col, "interaction", None, None, "交互", "difference", "quarterly")
        return out

    def _clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.select_dtypes(include=[np.number]).columns:
            if col == "target_value":
                continue
            ser = out[col]
            if self.config.clip_outliers:
                ser = winsorize_series(ser, self.config.winsorize_quantile)
            if self.config.fill_method == "ffill_bfill":
                ser = ser.ffill().bfill()
            elif self.config.fill_method == "zero":
                ser = ser.fillna(0.0)
            else:
                ser = ser.fillna(ser.median())
            out[col] = ser

        # 过滤缺失过高的列
        feature_cols = [c for c in out.columns if c not in {"quarter_end", "target_value"}]
        keep_cols = ["quarter_end", "target_value"]
        for col in feature_cols:
            miss_ratio = float(out[col].isna().mean())
            non_na_count = int(out[col].notna().sum())
            if miss_ratio <= self.config.max_feature_missing_ratio and non_na_count >= self.config.min_non_na_observations:
                keep_cols.append(col)
        out = out[keep_cols].copy()
        out = out.dropna(subset=["target_value"]).reset_index(drop=True)
        return out

    def build_training_panel(
        self,
        quarterly_target_df: pd.DataFrame,
        monthly_local_df: pd.DataFrame,
        monthly_national_df: pd.DataFrame,
        quarterly_panel_df: Optional[pd.DataFrame] = None,
        metadata_df: Optional[pd.DataFrame] = None,
        quarterly_factor_frame: Optional[pd.DataFrame] = None,
    ) -> FeatureArtifacts:
        registry = FeatureRegistry()
        notes: List[str] = []

        target = self._pivot_quarterly_target(quarterly_target_df)
        # 真实数据只保留四川省目标
        target = target[target["region"].astype(str).str.contains("四川", na=False)].copy()
        target = target[["quarter_end", "target_value"]].drop_duplicates("quarter_end")
        if target.empty:
            raise FeatureBuildError("未能构建四川省目标序列")

        frames: List[pd.DataFrame] = [target]

        if self.config.use_quarterly_panel_if_available and quarterly_panel_df is not None and not quarterly_panel_df.empty:
            qp = self._pivot_existing_quarterly_panel(quarterly_panel_df, registry, metadata_df)
            if not qp.empty:
                frames.append(qp)
                notes.append("used_existing_quarterly_panel")

        if self.config.build_monthly_aggregations:
            local_agg = self._build_monthly_aggregated_panel(monthly_local_df, "monthly_local", registry, metadata_df)
            national_agg = self._build_monthly_aggregated_panel(monthly_national_df, "monthly_national", registry, metadata_df)
            if not local_agg.empty:
                frames.append(local_agg)
            if not national_agg.empty:
                frames.append(national_agg)
            notes.append("rebuilt_monthly_aggregations")

        if quarterly_factor_frame is not None and not quarterly_factor_frame.empty:
            frames.append(quarterly_factor_frame.copy())
            notes.append("attached_dfm_quarterly_factors")

        panel = self._merge_frames_on_quarter(frames)
        if panel.empty:
            raise FeatureBuildError("训练面板构建失败，合并后为空")

        panel = panel.sort_values("quarter_end").reset_index(drop=True)
        panel = self._add_target_features(panel, registry)
        panel = self._add_feature_lags(panel, registry)
        panel = self._add_family_features(panel, registry)
        panel = self._add_interactions(panel, registry)
        panel = self._clean_and_impute(panel)

        # 限制特征数：不超过样本数的 1/2，避免 p>>n 导致模型退化
        n_rows = len(panel)
        protected = ["quarter_end", "target_value"]
        feature_cols = [c for c in panel.columns if c not in protected]
        feature_cap = min(self.config.small_sample_feature_cap, max(10, n_rows // 2))
        if len(feature_cols) > feature_cap:
            panel = cap_feature_count(panel, protected=protected, max_features=feature_cap)
            notes.append(f"feature_cap_applied: {len(feature_cols)} -> {feature_cap}")

        feature_cols = [c for c in panel.columns if c not in protected]
        if not feature_cols:
            raise FeatureBuildError("没有可用于训练的特征列")

        return FeatureArtifacts(
            training_panel=panel,
            feature_registry=registry,
            feature_columns=feature_cols,
            target_column="target_value",
            notes=notes,
            monthly_factor_frame=None,
            quarterly_factor_frame=quarterly_factor_frame,
        )
