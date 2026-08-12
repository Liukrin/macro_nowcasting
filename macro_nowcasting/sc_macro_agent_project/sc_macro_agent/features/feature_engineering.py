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
from typing import Any, Dict, List, Optional, Tuple

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
    safe_std,
    safe_trend,
    winsorize_series,
)


def select_features_by_policy(
    panel: pd.DataFrame,
    registry: "FeatureRegistry",
    config: "FeatureConfig",
    metadata_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """\n    规则驱动的特征白名单选择器。\n\n    完全不依赖数据分布（方差/相关性），只根据元数据和指标名称决定入选特征。\n    传入任何子集数据，返回的特征列表都一样（消除信息泄漏）。\n    """
    protected = {"quarter_end", "target_value"}
    all_cols = [c for c in panel.columns if c not in protected and pd.api.types.is_numeric_dtype(panel[c])]

    # ---- MIDAS 列整组保留（不通过白名单机制筛选）----
    midas_cols = [c for c in all_cols if c.startswith("midas__")]
    # 按组前缀聚合（去掉 __L{k} 后缀即为组）
    midas_groups: Dict[str, List[str]] = {}
    for c in midas_cols:
        # midas__{region}__{indicator}__L{k}
        group_key = c.rsplit("__L", 1)[0]
        midas_groups.setdefault(group_key, []).append(c)
    # 将所有 midas 列加入预选（后续 cap 时整组保留或丢弃）
    preselected_midas: List[str] = []
    for group_key, cols in midas_groups.items():
        preselected_midas.extend(cols)
    # 从 all_cols 中移除 midas 列，避免后续逻辑再处理
    all_cols = [c for c in all_cols if not c.startswith("midas__")]

    # 1. 用 metadata 判断每个基础指标的属性
    is_cum: Dict[str, bool] = {}
    if metadata_df is not None and not metadata_df.empty:
        for _, row in metadata_df.iterrows():
            name = str(row.get("standard_name", row.get("original_name", "")))
            is_cum[name] = str(row.get("is_cumulative", "")).strip().lower() == "true"

    def _region(col: str) -> str:
        return col.split("__", 1)[0] if "__" in col else "unknown"

    def _is_target_lag(col: str) -> bool:
        return col.startswith("target_lag_")

    def _is_cumulative_feature(col: str) -> bool:
        """Check if a column derives from a cumulative (level) indicator."""
        body = col.split("__", 1)[-1] if "__" in col else col
        for lag in range(1, 10):
            if body.endswith(f"__lag{lag}"):
                body = body[: -len(f"__lag{lag}")]
                break
        # Strip pandas merge disambiguation suffixes (_x, _y)
        if body.endswith("_x") or body.endswith("_y"):
            body = body[:-2]
        # Check metadata first
        for base_name, is_c in is_cum.items():
            if base_name in body:
                if is_c:
                    return True
                if "同比" in body or "增速" in body:
                    return False
        return "累计值" in body

    def _matches_indicator(col: str, indicator_keywords: List[str]) -> bool:
        """Check if a column matches any of the indicator keyword patterns."""
        for kw in indicator_keywords:
            if kw in col:
                return True
        return False

    def _has_agg_method(col: str, methods: List[str]) -> bool:
        """Check if column uses one of the allowed aggregation methods."""
        body = col.split("__", 1)[-1] if "__" in col else col
        # Strip __lagN suffix
        for lag in range(1, 10):
            if body.endswith(f"__lag{lag}"):
                body = body[: -len(f"__lag{lag}")]
                break
        # Strip pandas merge disambiguation suffixes (_x, _y)
        if body.endswith("_x") or body.endswith("_y"):
            body = body[:-2]
        for m in methods:
            if body.endswith(f"_{m}"):
                return True
        return False

    # 2. Build whitelist using fuzzy keyword matching on indicator names
    selected: List[str] = []

    # --- Tier B: Sichuan local indicators (must have) ---
    # New data uses "累计同比" (YTD cumulative YoY) naming
    sc_keywords = [
        "规模以上工业增加值_累计同比",
        "固定资产投资",
        "社会消费品零售总额_累计同比",
        "房地产开发投资_累计同比",
    ]
    for col in all_cols:
        if "四川省" not in col:
            continue
        if _is_cumulative_feature(col):
            continue  # exclude level features (累计值 without 同比)
        if not _has_agg_method(col, config.policy_agg_methods):
            continue
        if _matches_indicator(col, sc_keywords):
            selected.append(col)

    # --- Tier C: National leading indicators ---
    # ytd_yoy = YTD cumulative YoY, mom_yoy = monthly YoY
    nat_keywords = [
        "工业增加值_当月同比_mom_yoy",
        "固定资产投资（不含农户）_累计同比_ytd_yoy",
    ]
    for col in all_cols:
        if "全国" not in col:
            continue
        if _is_cumulative_feature(col):
            continue
        if not _has_agg_method(col, config.policy_agg_methods):
            continue
        if _matches_indicator(col, nat_keywords):
            selected.append(col)

    # --- PMI indicators (classic leading indicators, non-YoY) ---
    # pmi_data.csv column names: PMI, 生产, 新订单, ...
    # After pivot + lag: 全国__PMI_last, 全国__生产_mean__lag1, etc.
    pmi_patterns = ["__PMI_", "__生产_", "__新订单_"]
    for col in all_cols:
        if "全国" not in col:
            continue
        if _is_cumulative_feature(col):
            continue
        if not _has_agg_method(col, config.policy_agg_methods):
            continue
        # Check if column contains a PMI indicator marker
        for pp in pmi_patterns:
            if pp in col:
                selected.append(col)
                break

    # --- Tier E: Target lags ---
    # target_lag_1 is mandatory — it's the anchor for delta modeling
    if "target_lag_1" in all_cols and "target_lag_1" not in selected:
        selected.append("target_lag_1")
    for col in sorted(all_cols):
        if _is_target_lag(col) and col not in selected:
            selected.append(col)

    # 3. Add preselected MIDAS columns (always included, separate budget)
    selected = list(dict.fromkeys(selected))
    midas_to_include: List[str] = []
    for _group_key, cols in midas_groups.items():
        midas_to_include.extend(cols)
    # Cap only the non-MIDAS features
    if len(selected) > config.policy_max_features:
        target_lag_cols = [c for c in selected if _is_target_lag(c)]
        sichuan = [c for c in selected if _region(c) == "四川省"]
        pmi = [c for c in selected if "PMI" in c and c not in target_lag_cols and c not in sichuan]
        national = [c for c in selected if _region(c) == "全国" and c not in pmi and c not in target_lag_cols and c not in sichuan]
        other = [c for c in selected if c not in target_lag_cols and c not in sichuan and c not in pmi and c not in national]
        budget = config.policy_max_features
        result: List[str] = []
        result.extend(target_lag_cols[:budget - len(result)])
        result.extend(sichuan[:budget - len(result)])
        result.extend(pmi[:budget - len(result)])
        result.extend(national[:budget - len(result)])
        result.extend(other[:budget - len(result)])
        selected = result
    all_selected = selected + midas_to_include

    # 4. Report missing Sichuan indicators
    found_sc = [c for c in all_selected if "四川省" in c]
    for kw in ["规模以上工业增加值", "固定资产投资", "社会消费品零售总额", "房地产开发投资"]:
        if not any(kw in c for c in found_sc):
            print(f"  [WARNING] Sichuan indicator '{kw}' not found in panel")

    return all_selected


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

    def _build_midas_lag_panel(
        self,
        monthly_df: pd.DataFrame,
        source_table_name: str,
        registry: FeatureRegistry,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """从月度长表构建 MIDAS 滞后面板。

        对每个 (region, indicator) 按季度生成 midas_n_lags 个滞后列，保留原始
        月度值而非聚合，供 AlmonMIDASModel 使用。

        Returns:
            (midas_panel, notes) — midas_panel 以 quarter_end 为索引，
            每列名为 midas__{region}__{indicator}__L{k}
        """
        notes: List[str] = []
        if monthly_df.empty or not self.config.build_midas_lags:
            return pd.DataFrame(), notes

        df = self._sort_long(monthly_df)
        df["quarter_end"] = quarter_end(df["date"])
        df["month_rank_in_quarter"] = df.groupby(
            ["region", "indicator_name", "quarter_end"]
        ).cumcount(ascending=False)  # 0 = 最新月

        # 子串匹配挑选指标
        all_indicators = df["indicator_name"].unique().tolist()
        matched: List[str] = []
        for ind in all_indicators:
            ind_str = str(ind)
            for kw in self.config.midas_indicator_keywords:
                if kw in ind_str:
                    matched.append(ind_str)
                    break
        if not matched:
            notes.append("midas_no_matched_indicators")
            return pd.DataFrame(), notes

        # 上限
        if len(matched) > self.config.midas_max_indicators:
            matched = matched[: self.config.midas_max_indicators]
            notes.append(f"midas_capped_to_{self.config.midas_max_indicators}")

        df_matched = df[df["indicator_name"].isin(matched)].copy()

        # 对每个 (region, indicator) 按时间排序，为每个季度生成滞后列
        # L0 = 该季度最后一个月, L1 = 倒数第二月, ..., L5 = 上上季度第一个月
        rows: List[Dict[str, Any]] = []
        for (region, indicator), grp in df_matched.groupby(
            ["region", "indicator_name"]
        ):
            region = str(region)
            indicator = str(indicator)
            grp = grp.sort_values("date").reset_index(drop=True)
            # 构建日期→值的查找表
            dates = grp["date"].tolist()
            values = grp["indicator_value"].tolist()
            # 为每个季度生成一行
            quarters_seen: set = set()
            for idx in range(len(dates)):
                # Normalize to quarter start (midnight) to match _pivot_quarterly_target
                q_end = pd.Timestamp(dates[idx]).to_period("Q").to_timestamp("Q")
                if q_end in quarters_seen:
                    continue
                quarters_seen.add(q_end)
                row: Dict[str, Any] = {"quarter_end": q_end}
                # 从当前日期位置往回走 midas_n_lags 步
                for k in range(self.config.midas_n_lags):
                    col_name = f"midas__{region}__{indicator}__L{k}"
                    src_idx = idx - k
                    if src_idx >= 0 and dates[src_idx] <= q_end:
                        row[col_name] = float(values[src_idx])
                    else:
                        row[col_name] = np.nan
                rows.append(row)

        if not rows:
            notes.append("midas_empty_after_pivot")
            return pd.DataFrame(), notes

        midas_panel = pd.DataFrame(rows)
        midas_panel = midas_panel.groupby("quarter_end", as_index=False).first()
        midas_panel = midas_panel.sort_values("quarter_end").reset_index(drop=True)

        # 注册特征
        for col in midas_panel.columns:
            if col == "quarter_end":
                continue
            # 解析 midas__{region}__{indicator}__L{k}
            parts = col.split("__", 3)
            if len(parts) >= 4 and parts[0] == "midas":
                region = parts[1]
                indicator = parts[2]
                family = self._infer_family(indicator, metadata_df)
                registry.register(
                    name=col,
                    source_table=source_table_name,
                    source_indicator=indicator,
                    region=region,
                    family=family,
                    transform="midas_lag",
                    frequency="monthly",
                    note=f"月度滞后 L{parts[3][1:] if parts[3].startswith('L') else parts[3]}",
                )

        notes.append(f"midas_lags_built: {len(matched)} indicators, "
                     f"{len([c for c in midas_panel.columns if c != 'quarter_end'])} columns")
        return midas_panel, notes

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
            and not c.startswith("target_lag_")  # 禁止对已滞后列再次施加 lag
            and not c.startswith("midas__")      # 不衍生 midas 滞后特征
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
            candidates = [
                name for name in sub["name"].tolist()
                if name in out.columns and not name.startswith("midas__")
            ]
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
            and not c.startswith("midas__")
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

    def _clean_and_impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        clean_notes: List[str] = []
        out = df.copy()
        # Drop rows without target FIRST, before any fill.
        out = out.dropna(subset=["target_value"]).reset_index(drop=True)

        # ---- MIDAS 列特殊处理：组内前向填充，仍缺失则填列中位数 ----
        midas_cols = [c for c in out.columns if c.startswith("midas__")]
        if midas_cols:
            midas_filled_count = 0
            midas_total_na = int(out[midas_cols].isna().sum().sum())
            for col in midas_cols:
                ser = out[col]
                na_before = int(ser.isna().sum())
                if self.config.clip_outliers:
                    ser = winsorize_series(ser, self.config.winsorize_quantile)
                # 组内前向填充（按时间顺序）
                if na_before > 0:
                    ser = ser.ffill().bfill()
                    # 仍缺失则填中位数
                    still_na = int(ser.isna().sum())
                    if still_na > 0 and not ser.dropna().empty:
                        ser = ser.fillna(ser.median())
                        midas_filled_count += still_na
                out[col] = ser
            if midas_total_na > 0:
                fill_pct = midas_filled_count / max(midas_total_na, 1) * 100
                clean_notes.append(f"midas_impute: {midas_total_na} NA, {midas_filled_count} median-filled ({fill_pct:.1f}%)")

        for col in out.select_dtypes(include=[np.number]).columns:
            if col == "target_value" or col.startswith("midas__"):
                continue
            ser = out[col]
            if self.config.clip_outliers:
                ser = winsorize_series(ser, self.config.winsorize_quantile)
            if self.config.fill_method == "ffill_bfill":
                non_na_mask = ser.notna()
                if non_na_mask.any():
                    last_valid_idx = int(non_na_mask[non_na_mask].index[-1])
                    ser.loc[:last_valid_idx] = ser.loc[:last_valid_idx].ffill().bfill()
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
        return out, clean_notes

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

        # MIDAS 月度滞后面板（并行通道，不替换聚合面板）
        midas_local, midas_local_notes = self._build_midas_lag_panel(
            monthly_local_df, "monthly_local", registry, metadata_df)
        if not midas_local.empty:
            frames.append(midas_local)
        notes.extend(midas_local_notes)

        midas_national, midas_national_notes = self._build_midas_lag_panel(
            monthly_national_df, "monthly_national", registry, metadata_df)
        if not midas_national.empty:
            frames.append(midas_national)
        notes.extend(midas_national_notes)

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
        panel, clean_notes = self._clean_and_impute(panel)
        notes.extend(clean_notes)

        # 特征选择
        protected = ["quarter_end", "target_value"]
        n_rows = len(panel)

        if self.config.use_policy_selection:
            # 规则驱动的白名单 —— 不依赖数据分布，消除泄漏
            feature_cols = select_features_by_policy(panel, registry, self.config, metadata_df)
            feature_cols = [c for c in feature_cols if c in panel.columns]
            panel = panel[protected + feature_cols].copy()
            notes.append(f"policy_selection: {len(feature_cols)} features")
        else:
            # 旧逻辑：方差排序（保留但默认关闭）
            feature_cols = [c for c in panel.columns if c not in protected]
            feature_cap = min(self.config.small_sample_feature_cap, max(15, n_rows * 3 // 2))
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
