"""
数据质量检查与文本报告生成。
审计检查（重复、缺失、日期范围、期数统计等）基于 SQLite SQL 完成。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import logging
import pandas as pd
from .data_contracts import REQUIRED_LONG_COLUMNS, REQUIRED_METADATA_COLUMNS
from .sql_store import (
    SqlDataStore,
    sql_month_key_expr,
    sql_quarter_key_expr,
)

_logger = logging.getLogger("sc_macro_agent.data_quality")


def run_adf_tests(
    series_dict: Dict[str, pd.Series],
    max_series: int = 30,
) -> List[Dict[str, Any]]:
    """对多条序列执行 ADF 平稳性检验。

    Args:
        series_dict: {名称: 序列} 的字典
        max_series: 最多检验的序列数，防止页面卡顿

    Returns:
        [{indicator, n_obs, adf_stat, p_value, used_lag,
          critical_1pct, critical_5pct, is_stationary, conclusion}]
        statsmodels 缺失时返回空列表。
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        _logger.warning("statsmodels 未安装，跳过 ADF 平稳性检验")
        return []

    results: List[Dict[str, Any]] = []
    count = 0

    for name, series in series_dict.items():
        if count >= max_series:
            break
        ser = series.dropna().astype(float)
        n_obs = len(ser)

        if n_obs < 12:
            results.append({
                "indicator": name,
                "n_obs": n_obs,
                "adf_stat": None,
                "p_value": None,
                "used_lag": None,
                "critical_1pct": None,
                "critical_5pct": None,
                "is_stationary": False,
                "conclusion": "样本不足（需 ≥12 个观测）",
            })
            count += 1
            continue

        try:
            adf_result = adfuller(ser.values, autolag="AIC")
            adf_stat = float(adf_result[0])
            p_value = float(adf_result[1])
            used_lag = int(adf_result[2])
            crit = adf_result[4]
            crit_1 = float(crit.get("1%", float("nan")))
            crit_5 = float(crit.get("5%", float("nan")))
            is_stat = p_value < 0.05

            if is_stat:
                conclusion = f"平稳（p={p_value:.4f} < 0.05，拒绝单位根原假设）"
            else:
                conclusion = f"非平稳（p={p_value:.4f} ≥ 0.05，不能拒绝单位根原假设）"

            results.append({
                "indicator": name,
                "n_obs": n_obs,
                "adf_stat": adf_stat,
                "p_value": p_value,
                "used_lag": used_lag,
                "critical_1pct": crit_1,
                "critical_5pct": crit_5,
                "is_stationary": is_stat,
                "conclusion": conclusion,
            })
        except Exception as exc:
            results.append({
                "indicator": name,
                "n_obs": n_obs,
                "adf_stat": None,
                "p_value": None,
                "used_lag": None,
                "critical_1pct": None,
                "critical_5pct": None,
                "is_stationary": False,
                "conclusion": f"检验失败: {exc}",
            })
        count += 1

    return results


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

    实现方式：将待审表载入内存 SQLite（SqlDataStore），
    重复检测、缺失率、日期边界、期数统计等检查均以 SQL 查询完成。
    """

    def __init__(self) -> None:
        self.checks: List[QualityCheck] = []
        self.summaries: List[DatasetSummary] = []
        self._store: Optional[SqlDataStore] = None

    def reset(self) -> None:
        self.checks = []
        self.summaries = []

    def _append(self, check_name: str, passed: bool, details: str, severity: str = "info") -> None:
        self.checks.append(QualityCheck(check_name, passed, details, severity))

    def _get_store(self) -> SqlDataStore:
        if self._store is None:
            self._store = SqlDataStore.memory()
        return self._store

    def _table_summary(self, name: str, df: pd.DataFrame) -> DatasetSummary:
        """用单条聚合 SQL 产出表级摘要（行数/日期边界/去重计数/缺失率）。"""
        store = self._get_store()
        cols = store.business_columns(name)
        if not cols:
            item = DatasetSummary(name=name, rows=0, columns=len(df.columns),
                                  start_date=None, end_date=None,
                                  n_regions=0, n_indicators=0, missing_ratio=0.0)
            self.summaries.append(item)
            return item

        has_date = "date" in cols
        select_parts = ["COUNT(*) AS rows_"]
        if has_date:
            select_parts.append('MIN(date) AS start_date, MAX(date) AS "end_date"')
        if "region" in cols:
            select_parts.append("COUNT(DISTINCT region) AS n_regions")
        if "indicator_name" in cols:
            select_parts.append("COUNT(DISTINCT indicator_name) AS n_indicators")
        select_parts.append(f"{store.missing_null_expression(name)} AS null_count")
        row = store.query_df(f'SELECT {", ".join(select_parts)} FROM "{name}"').iloc[0]

        rows = int(row["rows_"])
        denom = max(rows * max(len(cols), 1), 1)
        item = DatasetSummary(
            name=name,
            rows=rows,
            columns=len(cols),
            start_date=row.get("start_date") if has_date else None,
            end_date=row.get("end_date") if has_date else None,
            n_regions=int(row["n_regions"]) if "region" in cols else 0,
            n_indicators=int(row["n_indicators"]) if "indicator_name" in cols else 0,
            missing_ratio=float(int(row["null_count"]) / denom),
        )
        self.summaries.append(item)
        return item

    def validate_long_table(self, name: str, df: pd.DataFrame) -> None:
        store = self._get_store()
        store.load_dataframe(name, df)
        cols = store.business_columns(name)

        missing = [c for c in REQUIRED_LONG_COLUMNS if c not in cols]
        self._append(
            f"{name}: required_columns",
            len(missing) == 0,
            "缺失字段: " + ", ".join(missing) if missing else "字段齐全",
            "error" if missing else "info",
        )

        if "date" in cols:
            # 不可解析日期在入表时已转为 NULL，用 SQL 计数
            bad = int(store.query_one(
                f'SELECT COUNT(*) FROM "{name}" WHERE date IS NULL', default=0))
            self._append(
                f"{name}: parsable_dates",
                bad == 0,
                f"不可解析日期数量={bad}",
                "error" if bad else "info",
            )

        if {"date", "region", "indicator_name"}.issubset(cols):
            dup = int(store.query_one(
                f'SELECT COALESCE(SUM(cnt - 1), 0) FROM ('
                f"SELECT COUNT(*) AS cnt FROM \"{name}\" "
                f"GROUP BY date, region, indicator_name HAVING COUNT(*) > 1)",
                default=0))
            self._append(
                f"{name}: duplicates",
                dup == 0,
                f"重复记录数={dup}",
                "warning" if dup else "info",
            )

        if "indicator_value" in cols:
            bad = int(store.query_one(
                f'SELECT COUNT(*) FROM "{name}" WHERE indicator_value IS NULL', default=0))
            self._append(
                f"{name}: numeric_indicator_value",
                bad == 0,
                f"indicator_value 非数值个数={bad}",
                "warning" if bad else "info",
            )

        if "frequency" in cols:
            uniq_row = store.query_one(
                f'SELECT GROUP_CONCAT(DISTINCT frequency) FROM "{name}" '
                f"WHERE frequency IS NOT NULL",
                default="")
            uniq = sorted([u for u in str(uniq_row).split(",") if u]) if uniq_row else []
            self._append(
                f"{name}: frequency_values",
                True,
                f"frequency={uniq}",
                "info",
            )

        self._table_summary(name, df)

    def validate_metadata(self, name: str, df: pd.DataFrame) -> None:
        store = self._get_store()
        store.load_dataframe(name, df)
        cols = store.business_columns(name)
        missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in cols]
        self._append(
            f"{name}: required_columns",
            len(missing) == 0,
            "缺失字段: " + ", ".join(missing) if missing else "字段齐全",
            "error" if missing else "info",
        )
        self._table_summary(name, df)

    def validate_quarter_alignment(self, monthly_table: str, quarterly_table: str) -> None:
        store = self._get_store()
        if not store.table_exists(monthly_table) or not store.table_exists(quarterly_table):
            self._append("quarter_alignment", False, "monthly 或 quarterly 为空", "warning")
            return
        # 用 INTERSECT 求两表季度集合的交集大小（SQL）
        q_expr = sql_quarter_key_expr()
        overlap = int(store.query_one(
            f"SELECT COUNT(*) FROM ("
            f'SELECT DISTINCT {q_expr} AS qk FROM "{monthly_table}" WHERE date IS NOT NULL '
            f"INTERSECT "
            f'SELECT DISTINCT {q_expr} AS qk FROM "{quarterly_table}" WHERE date IS NOT NULL)',
            default=0))
        m_quarters = int(store.query_one(
            f'SELECT COUNT(DISTINCT {q_expr}) FROM "{monthly_table}" WHERE date IS NOT NULL', default=0))
        q_quarters = int(store.query_one(
            f'SELECT COUNT(DISTINCT {q_expr}) FROM "{quarterly_table}" WHERE date IS NOT NULL', default=0))
        self._append(
            "quarter_alignment",
            overlap > 0,
            f"季度重叠数={overlap}; monthly_quarters={m_quarters}; quarterly_quarters={q_quarters}",
            "warning" if overlap == 0 else "info",
        )

    def validate_target_history(self, table_name: str = "quarterly_target", min_quarters: int = 8) -> None:
        store = self._get_store()
        if not store.table_exists(table_name):
            self._append("target_history", False, "季度目标表为空", "error")
            return
        cols = store.table_columns(table_name)
        gdp_filter = " WHERE indicator_name LIKE '%GDP%'" if "indicator_name" in cols else ""
        quarters = int(store.query_one(
            f'SELECT COUNT(DISTINCT {sql_quarter_key_expr()}) FROM "{table_name}"'
            f"{gdp_filter} AND date IS NOT NULL" if gdp_filter else
            f'SELECT COUNT(DISTINCT {sql_quarter_key_expr()}) FROM "{table_name}" WHERE date IS NOT NULL',
            default=0))
        self._append(
            "target_history",
            quarters >= min_quarters,
            f"目标季度数={quarters}, 建议至少={min_quarters}",
            "warning" if quarters < min_quarters else "info",
        )

    def validate_monthly_density(self, table_name: str, expected_min_months: int = 12) -> None:
        store = self._get_store()
        if not store.table_exists(table_name):
            self._append("monthly_density", False, "月度表为空", "warning")
            return
        months = int(store.query_one(
            f'SELECT COUNT(DISTINCT {sql_month_key_expr()}) FROM "{table_name}" WHERE date IS NOT NULL',
            default=0))
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
        if self._store is not None:
            self._store.close()
            self._store = None

        self.validate_long_table("quarterly_target", quarterly_target_df)
        self.validate_long_table("monthly_local", monthly_local_df)
        self.validate_long_table("monthly_national", monthly_national_df)

        if quarterly_panel_df is not None:
            self.validate_long_table("quarterly_panel", quarterly_panel_df)
        if metadata_df is not None:
            self.validate_metadata("metadata", metadata_df)

        self.validate_quarter_alignment("monthly_national", "quarterly_target")
        self.validate_quarter_alignment("monthly_local", "quarterly_target")
        self.validate_target_history("quarterly_target")
        self.validate_monthly_density("monthly_national")
        self.validate_monthly_density("monthly_local", expected_min_months=3)

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
