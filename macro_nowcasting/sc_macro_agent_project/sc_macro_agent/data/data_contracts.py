"""
数据契约定义。

这里把长表格式下的关键列、频率和默认文件名尽量集中起来，
方便以后做更严格的 schema 校验或者迁移到 pydantic/pandera。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


REQUIRED_LONG_COLUMNS = [
    "date",
    "region",
    "indicator_name",
    "indicator_value",
    "frequency",
    "unit",
    "source_name",
    "source_url",
    "note",
]

REQUIRED_METADATA_COLUMNS = [
    "original_name",
    "standard_name",
    "category",
    "frequency",
    "is_target",
    "is_yoy",
    "is_cumulative",
    "leading_indicator",
    "data_quality",
    "notes",
]


@dataclass(frozen=True)
class Frequency:
    DAILY: str = "daily"
    WEEKLY: str = "weekly"
    MONTHLY: str = "monthly"
    QUARTERLY: str = "quarterly"


LONG_TABLE_NAME_MAP: Dict[str, str] = {
    "quarterly_target_real": "quarterly_target_real.csv",
    "monthly_local_real": "monthly_local_features_real.csv",
    "monthly_national_real": "monthly_national_features_real.csv",
    "quarterly_panel_real": "quarterly_feature_panel_real.csv",
    "metadata_real": "metadata_real.csv",
    "quarterly_target_demo": "quarterly_target.csv",
    "monthly_local_demo": "monthly_local_features.csv",
    "monthly_national_demo": "monthly_national_features.csv",
    "quarterly_panel_demo": "quarterly_feature_panel.csv",
    "metadata_demo": "metadata.csv",
}


TARGET_CANDIDATES: List[str] = [
    "GDP_同比增速",
    "四川省GDP_同比增速",
    "gdp_yoy",
]


DEFAULT_LEADING_INDICATORS: List[str] = [
    "PMI_PMI",
    "PMI_生产",
    "PMI_新订单",
    "规模以上工业增加值_同比增速",
    "固定资产投资（不含农户）_同比增速",
    "社会消费品零售总额_同比增速",
    "全社会用电量_同比增速",
]
