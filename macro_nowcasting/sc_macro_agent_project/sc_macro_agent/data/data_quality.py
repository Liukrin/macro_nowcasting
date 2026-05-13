"""
数据质量检查与文本报告生成。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from .data_contracts import REQUIRED_LONG_COLUMNS, REQUIRED_METADATA_COLUMNS
from ..exceptions import DataContractError
from ..utils import quarter_end, json_dumps


@dataclass
class QualityCheck:
    check_name: str
    passed: bool
    details: str
    severity: str = "info"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    name: str
    rows: int
    columns: int
    start_date: Optional[str]
    end_date: Optional[str]
    n_regions: int
    n_indicators: int
    missing_ratio: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataQualityAuditor:
    """
    针对“长表+元数据”结构做轻量审计。
    """

    def __init__(self) -> None:
        self.checks: List[QualityCheck] = []
        self.summaries: List[DatasetSummary] = []

    def reset(self) -> None:
        self.checks = []
        self.summaries = []

    def _append(self, check_name: str, passed: bool, details: str, severity: str = "info") -> None:
        self.checks.append(QualityCheck(check_name, passed, details, severity))

    def _summary(self, name: str, df: pd.DataFrame) -> DatasetSummary:
        start = None
        end = None
        if "date" in df.columns and not df.empty:
            dt = pd.to_datetime(df["date"], errors="coerce")
            if dt.notna().any():
                start = dt.min().strftime("%Y-%m-%d")
                end = dt.max().strftime("%Y-%m-%d")
        n_regions = int(df["region"].nunique()) if "region" in df.columns else 0
        n_indicators = int(df["indicator_name"].nunique()) if "indicator_name" in df.columns else 0
        denom = max(df.shape[0] * max(df.shape[1], 1), 1)
        missing_ratio = float(df.isna().sum().sum() / denom)
        item = DatasetSummary(
            name=name,
            rows=int(df.shape[0]),
            columns=int(df.shape[1]),
            start_date=start,
            end_date=end,
            n_regions=n_regions,
            n_indicators=n_indicators,
            missing_ratio=missing_ratio,
        )
        self.summaries.append(item)
        return item

    def validate_long_table(self, name: str, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_LONG_COLUMNS if c not in df.columns]
        self._append(
            f"{name}: required_columns",
            len(missing) == 0,
            "缺失字段: " + ", ".join(missing) if missing else "字段齐全",
            "error" if missing else "info",
        )

        if "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            bad = int(dt.isna().sum())
            self._append(
                f"{name}: parsable_dates",
                bad == 0,
                f"不可解析日期数量={bad}",
                "error" if bad else "info",
            )

        if {"date", "region", "indicator_name"}.issubset(df.columns):
            dup = int(df.duplicated(subset=["date", "region", "indicator_name"]).sum())
            self._append(
                f"{name}: duplicates",
                dup == 0,
                f"重复记录数={dup}",
                "warning" if dup else "info",
            )

        if "indicator_value" in df.columns:
            numeric = pd.to_numeric(df["indicator_value"], errors="coerce")
            bad = int(numeric.isna().sum())
            self._append(
                f"{name}: numeric_indicator_value",
                bad == 0,
                f"indicator_value 非数值个数={bad}",
                "warning" if bad else "info",
            )

        if "frequency" in df.columns:
            uniq = sorted(df["frequency"].astype(str).dropna().unique().tolist())
            self._append(
                f"{name}: frequency_values",
                True,
                f"frequency={uniq}",
                "info",
            )

        self._summary(name, df)

    def validate_metadata(self, name: str, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
        self._append(
            f"{name}: required_columns",
            len(missing) == 0,
            "缺失字段: " + ", ".join(missing) if missing else "字段齐全",
            "error" if missing else "info",
        )
        self._summary(name, df)

    def validate_quarter_alignment(self, monthly_df: pd.DataFrame, quarterly_df: pd.DataFrame) -> None:
        if monthly_df.empty or quarterly_df.empty:
            self._append("quarter_alignment", False, "monthly 或 quarterly 为空", "warning")
            return
        m = monthly_df.copy()
        m["quarter_end"] = quarter_end(m["date"])
        q_quarters = set(pd.to_datetime(quarterly_df["date"], errors="coerce").dt.to_period("Q").dt.to_timestamp("Q").dropna())
        m_quarters = set(pd.to_datetime(m["quarter_end"]).dropna())
        overlap = len(q_quarters & m_quarters)
        self._append(
            "quarter_alignment",
            overlap > 0,
            f"季度重叠数={overlap}; monthly_quarters={len(m_quarters)}; quarterly_quarters={len(q_quarters)}",
            "warning" if overlap == 0 else "info",
        )

    def validate_target_history(self, quarterly_target_df: pd.DataFrame, min_quarters: int = 8) -> None:
        if quarterly_target_df.empty:
            self._append("target_history", False, "季度目标表为空", "error")
            return
        if "indicator_name" in quarterly_target_df.columns:
            target = quarterly_target_df[quarterly_target_df["indicator_name"].astype(str).str.contains("GDP", na=False)]
        else:
            target = quarterly_target_df
        quarters = int(pd.to_datetime(target["date"], errors="coerce").dt.to_period("Q").nunique())
        self._append(
            "target_history",
            quarters >= min_quarters,
            f"目标季度数={quarters}, 建议至少={min_quarters}",
            "warning" if quarters < min_quarters else "info",
        )

    def validate_monthly_density(self, monthly_df: pd.DataFrame, expected_min_months: int = 12) -> None:
        if monthly_df.empty:
            self._append("monthly_density", False, "月度表为空", "warning")
            return
        months = int(pd.to_datetime(monthly_df["date"], errors="coerce").dt.to_period("M").nunique())
        self._append(
            "monthly_density",
            months >= expected_min_months,
            f"月度期数={months}, 建议至少={expected_min_months}",
            "warning" if months < expected_min_months else "info",
        )

    def run_full_audit(
        self,
        quarterly_target_df: pd.DataFrame,
        monthly_local_df: pd.DataFrame,
        monthly_national_df: pd.DataFrame,
        quarterly_panel_df: Optional[pd.DataFrame] = None,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        self.reset()
        self.validate_long_table("quarterly_target", quarterly_target_df)
        self.validate_long_table("monthly_local", monthly_local_df)
        self.validate_long_table("monthly_national", monthly_national_df)

        if quarterly_panel_df is not None:
            self.validate_long_table("quarterly_panel", quarterly_panel_df)
        if metadata_df is not None:
            self.validate_metadata("metadata", metadata_df)

        self.validate_quarter_alignment(monthly_national_df, quarterly_target_df)
        self.validate_quarter_alignment(monthly_local_df, quarterly_target_df)
        self.validate_target_history(quarterly_target_df)
        self.validate_monthly_density(monthly_national_df)
        self.validate_monthly_density(monthly_local_df, expected_min_months=3)

        passed = all(c.passed or c.severity != "error" for c in self.checks)
        warnings = [c.details for c in self.checks if not c.passed]
        return {
            "status": "passed" if passed else "failed",
            "summary": [s.as_dict() for s in self.summaries],
            "checks": [c.as_dict() for c in self.checks],
            "warnings": warnings,
        }

    def build_text_report(self, audit_result: Dict[str, Any], dataset_mode: str) -> str:
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("四川省GDP混频预测项目 - 自动数据审计报告")
        lines.append("=" * 80)
        lines.append(f"数据模式: {dataset_mode}")
        lines.append("")

        lines.append("一、数据摘要")
        lines.append("-" * 80)
        for item in audit_result.get("summary", []):
            lines.append(
                f"{item['name']}: rows={item['rows']}, cols={item['columns']}, "
                f"start={item['start_date']}, end={item['end_date']}, "
                f"regions={item['n_regions']}, indicators={item['n_indicators']}, "
                f"missing_ratio={item['missing_ratio']:.4f}"
            )

        lines.append("")
        lines.append("二、质量检查")
        lines.append("-" * 80)
        for item in audit_result.get("checks", []):
            flag = "✓" if item["passed"] else "✗"
            lines.append(f"{flag} [{item['severity']}] {item['check_name']} -> {item['details']}")

        lines.append("")
        lines.append("三、结论")
        lines.append("-" * 80)
        lines.append(f"审计状态: {audit_result.get('status')}")
        if audit_result.get("warnings"):
            lines.append("警告:")
            for w in audit_result["warnings"]:
                lines.append(f"- {w}")
        else:
            lines.append("无明显警告。")

        lines.append("")
        lines.append("四、建议")
        lines.append("-" * 80)
        lines.append("1. 如果真实季度目标样本太短，优先使用小样本友好模型。")
        lines.append("2. 如果月度本地指标只有最近一期，保留全国 PMI/工业/投资等领先因子。")
        lines.append("3. 尽量继续补历史序列，让回测窗口数量更稳定。")
        lines.append("")
        return "\n".join(lines)
