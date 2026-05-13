"""
数据管理层。

负责：
- 识别 real / demo / hybrid 数据模式
- 读取标准化长表
- 做基础规范化
- 给下游返回统一的 DataBundle
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from ..config import DataConfig
from ..exceptions import DataNotReadyError, ArtifactError
from ..logging_utils import get_logger
from ..utils import ensure_datetime, infer_data_mode, quarter_end


@dataclass
class DataBundle:
    dataset_mode: str
    quarterly_target: pd.DataFrame
    monthly_local: pd.DataFrame
    monthly_national: pd.DataFrame
    quarterly_panel: pd.DataFrame
    metadata: pd.DataFrame
    notes: List[str]

    def tables(self) -> Dict[str, pd.DataFrame]:
        return {
            "quarterly_target": self.quarterly_target,
            "monthly_local": self.monthly_local,
            "monthly_national": self.monthly_national,
            "quarterly_panel": self.quarterly_panel,
            "metadata": self.metadata,
        }


class DataManager:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.logger = get_logger("sc_macro_agent.data_manager")
        self.bundle: Optional[DataBundle] = None

    def _csv_exists(self, filename: str) -> bool:
        return (self.config.resolve_dir() / filename).exists()

    def detect_mode(self) -> str:
        real_available = all([
            self._csv_exists(self.config.quarterly_target_file),
            self._csv_exists(self.config.monthly_local_file),
            self._csv_exists(self.config.monthly_national_file),
        ])
        demo_available = all([
            self._csv_exists(self.config.demo_quarterly_target_file),
            self._csv_exists(self.config.demo_monthly_local_file),
            self._csv_exists(self.config.demo_monthly_national_file),
        ])
        mode = infer_data_mode(real_available, demo_available, self.config.dataset_mode)

        if mode == "real" and not real_available and self.config.allow_demo_fallback and demo_available:
            return "demo"
        return mode

    def _read_csv(self, filename: str) -> pd.DataFrame:
        path = self.config.resolve_dir() / filename
        if not path.exists():
            raise DataNotReadyError(f"文件不存在: {path}")
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = ensure_datetime(df["date"])
        return df

    def _standardize_long_table(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "date" in out.columns:
            out["date"] = ensure_datetime(out["date"])
        if "indicator_value" in out.columns:
            out["indicator_value"] = pd.to_numeric(out["indicator_value"], errors="coerce")
        if "region" in out.columns:
            out["region"] = out["region"].astype(str).str.strip()
        if "indicator_name" in out.columns:
            out["indicator_name"] = out["indicator_name"].astype(str).str.strip()
        if "frequency" in out.columns:
            out["frequency"] = out["frequency"].astype(str).str.strip().str.lower()
        return out

    def _read_mode_tables(self, mode: str) -> DataBundle:
        if mode == "real":
            quarterly_target = self._standardize_long_table(self._read_csv(self.config.quarterly_target_file))
            monthly_local = self._standardize_long_table(self._read_csv(self.config.monthly_local_file))
            monthly_national = self._standardize_long_table(self._read_csv(self.config.monthly_national_file))
            quarterly_panel = self._standardize_long_table(self._read_csv(self.config.quarterly_panel_file))                     if self._csv_exists(self.config.quarterly_panel_file) else pd.DataFrame()
            metadata = self._read_csv(self.config.metadata_file) if self._csv_exists(self.config.metadata_file) else pd.DataFrame()
            notes = ["using_real_data"]
        elif mode == "demo":
            quarterly_target = self._standardize_long_table(self._read_csv(self.config.demo_quarterly_target_file))
            monthly_local = self._standardize_long_table(self._read_csv(self.config.demo_monthly_local_file))
            monthly_national = self._standardize_long_table(self._read_csv(self.config.demo_monthly_national_file))
            quarterly_panel = self._standardize_long_table(self._read_csv(self.config.demo_quarterly_panel_file))                     if self._csv_exists(self.config.demo_quarterly_panel_file) else pd.DataFrame()
            metadata = self._read_csv(self.config.demo_metadata_file) if self._csv_exists(self.config.demo_metadata_file) else pd.DataFrame()
            notes = ["using_demo_data"]
        elif mode == "hybrid":
            real_bundle = self._read_mode_tables("real")
            demo_bundle = self._read_mode_tables("demo")
            quarterly_target = real_bundle.quarterly_target
            monthly_local = pd.concat([demo_bundle.monthly_local, real_bundle.monthly_local], ignore_index=True)
            monthly_national = pd.concat([demo_bundle.monthly_national, real_bundle.monthly_national], ignore_index=True)
            quarterly_panel = pd.concat([demo_bundle.quarterly_panel, real_bundle.quarterly_panel], ignore_index=True)
            metadata = pd.concat([demo_bundle.metadata, real_bundle.metadata], ignore_index=True).drop_duplicates()
            notes = ["using_hybrid_data", "target_real_features_hybrid"]
        else:
            raise DataNotReadyError(f"不支持的数据模式: {mode}")

        return DataBundle(
            dataset_mode=mode,
            quarterly_target=quarterly_target,
            monthly_local=monthly_local,
            monthly_national=monthly_national,
            quarterly_panel=quarterly_panel,
            metadata=metadata,
            notes=notes,
        )

    def initialize(self) -> None:
        mode = self.detect_mode()
        self.bundle = self._read_mode_tables(mode)
        self.logger.info("数据管理器初始化完成 | mode=%s", self.bundle.dataset_mode)

    def get_bundle(self) -> DataBundle:
        if self.bundle is None:
            self.initialize()
        assert self.bundle is not None
        return self.bundle

    def get_table(self, name: str) -> pd.DataFrame:
        bundle = self.get_bundle()
        tables = bundle.tables()
        if name not in tables:
            raise DataNotReadyError(f"未知表名: {name}")
        return tables[name].copy()

    def get_latest_snapshot(self) -> Dict[str, Any]:
        bundle = self.get_bundle()
        quarterly_target = bundle.quarterly_target.copy()
        if quarterly_target.empty:
            return {"status": "empty"}

        latest_date = quarterly_target["date"].max()
        latest_frame = quarterly_target[quarterly_target["date"] == latest_date].copy()
        monthly_local = bundle.monthly_local.copy()
        monthly_national = bundle.monthly_national.copy()

        return {
            "dataset_mode": bundle.dataset_mode,
            "latest_target_date": latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else None,
            "latest_target_rows": int(len(latest_frame)),
            "target_indicators": sorted(latest_frame["indicator_name"].unique().tolist()),
            "local_latest_month": monthly_local["date"].max().strftime("%Y-%m-%d") if not monthly_local.empty else None,
            "national_latest_month": monthly_national["date"].max().strftime("%Y-%m-%d") if not monthly_national.empty else None,
            "notes": bundle.notes,
        }

    def get_data_availability(self) -> Dict[str, Any]:
        bundle = self.get_bundle()
        items = []
        for name, df in bundle.tables().items():
            if df is None:
                continue
            start = None
            end = None
            if "date" in df.columns and not df.empty:
                dt = pd.to_datetime(df["date"], errors="coerce")
                if dt.notna().any():
                    start = dt.min().strftime("%Y-%m-%d")
                    end = dt.max().strftime("%Y-%m-%d")
            denom = max(df.shape[0] * max(df.shape[1], 1), 1)
            items.append({
                "name": name,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "start": start,
                "end": end,
                "missing_ratio": float(df.isna().sum().sum() / denom),
                "indicators": int(df["indicator_name"].nunique()) if "indicator_name" in df.columns else 0,
                "regions": int(df["region"].nunique()) if "region" in df.columns else 0,
            })
        return {
            "dataset_mode": bundle.dataset_mode,
            "items": items,
            "notes": bundle.notes,
        }

    def query_indicator(self, table_name: str, indicator_keyword: str) -> pd.DataFrame:
        df = self.get_table(table_name)
        if "indicator_name" not in df.columns:
            return df.iloc[0:0].copy()
        mask = df["indicator_name"].astype(str).str.contains(indicator_keyword, case=False, na=False)
        return df.loc[mask].copy()

    def build_training_signal_overview(self) -> Dict[str, Any]:
        bundle = self.get_bundle()
        quarterly_target = bundle.quarterly_target.copy()
        monthly_national = bundle.monthly_national.copy()
        monthly_local = bundle.monthly_local.copy()

        target_quarters = int(pd.to_datetime(quarterly_target["date"]).dt.to_period("Q").nunique()) if not quarterly_target.empty else 0
        nat_months = int(pd.to_datetime(monthly_national["date"]).dt.to_period("M").nunique()) if not monthly_national.empty else 0
        local_months = int(pd.to_datetime(monthly_local["date"]).dt.to_period("M").nunique()) if not monthly_local.empty else 0

        local_indicators = sorted(monthly_local["indicator_name"].unique().tolist()) if "indicator_name" in monthly_local.columns else []
        national_indicators = sorted(monthly_national["indicator_name"].unique().tolist()) if "indicator_name" in monthly_national.columns else []

        return {
            "dataset_mode": bundle.dataset_mode,
            "target_quarters": target_quarters,
            "national_months": nat_months,
            "local_months": local_months,
            "national_indicator_count": len(national_indicators),
            "local_indicator_count": len(local_indicators),
            "local_indicators_sample": local_indicators[:10],
            "national_indicators_sample": national_indicators[:10],
            "notes": bundle.notes,
        }

    def export_snapshot(self, out_dir: Optional[str] = None) -> Dict[str, str]:
        bundle = self.get_bundle()
        out = Path(out_dir or self.config.resolve_artifact_dir() / "snapshots")
        out.mkdir(parents=True, exist_ok=True)
        mapping: Dict[str, str] = {}
        for name, df in bundle.tables().items():
            path = out / f"{name}.csv"
            df.to_csv(path, index=False, encoding="utf-8-sig")
            mapping[name] = str(path)
        return mapping
