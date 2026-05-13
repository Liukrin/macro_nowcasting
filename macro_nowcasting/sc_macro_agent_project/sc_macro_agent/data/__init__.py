"""数据层：数据获取、校验、合约、生成。"""
from .data_contracts import REQUIRED_LONG_COLUMNS, REQUIRED_METADATA_COLUMNS
from .data_generator import MacroDataGenerator
from .data_manager import DataManager, DataBundle
from .data_quality import DataQualityAuditor

__all__ = [
    "REQUIRED_LONG_COLUMNS",
    "REQUIRED_METADATA_COLUMNS",
    "MacroDataGenerator",
    "DataManager",
    "DataBundle",
    "DataQualityAuditor",
]
