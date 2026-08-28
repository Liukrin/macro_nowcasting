"""
数据管理（SQLite 版）
- 识别 real / demo / hybrid 数据模式
- 将标准化长表加载进 SQLite（data/macro.db）
- 所有表读取、指标查询、快照与可用性统计均通过 SQL 完成
- 给下游返回统一的 DataBundle
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from ..config import DataConfig
from ..exceptions import DataNotReadyError
from ..logging_utils import get_logger
from ..utils import ensure_datetime, infer_data_mode
from .sql_store import (
    SqlDataStore,
    sql_month_key_expr,
    sql_quarter_key_expr,
)


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
        # SQLite 存储：持久化到 data/ 下的 .db 文件
        db_path = self.config.resolve_dir() / self.config.sqlite_db_file
        self.sql = SqlDataStore(db_path)

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

    # ------------------------------------------------------------------
    # CSV → SQLite 加载
    # ------------------------------------------------------------------
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

    def _load_table_to_sql(self, table_name: str, filename: str, long_table: bool = True) -> None:
        """读取 CSV、规范化，并写入 SQLite。文件缺失时清空同名表。"""
        if not self._csv_exists(filename):
            self.sql.load_dataframe(table_name, pd.DataFrame())
            return
        df = self._read_csv(filename)
        if long_table:
            df = self._standardize_long_table(df)
        self.sql.load_dataframe(table_name, df)

    def _load_mode_to_sql(self, mode: str) -> None:
        """将指定模式的 5 张表加载进 SQLite（表名即逻辑表名）。"""
        if mode == "real":
            self._load_table_to_sql("quarterly_target", self.config.quarterly_target_file)
            self._load_table_to_sql("monthly_local", self.config.monthly_local_file)
            self._load_table_to_sql("monthly_national", self.config.monthly_national_file)
            self._load_table_to_sql("quarterly_panel", self.config.quarterly_panel_file)
            self._load_table_to_sql("metadata", self.config.metadata_file, long_table=False)
        elif mode == "demo":
            self._load_table_to_sql("quarterly_target", self.config.demo_quarterly_target_file)
            self._load_table_to_sql("monthly_local", self.config.demo_monthly_local_file)
            self._load_table_to_sql("monthly_national", self.config.demo_monthly_national_file)
            self._load_table_to_sql("quarterly_panel", self.config.demo_quarterly_panel_file)
            self._load_table_to_sql("metadata", self.config.demo_metadata_file, long_table=False)
        elif mode == "hybrid":
            # real 先写入主表，demo 以临时表载入后用 SQL UNION ALL 合并
            self._load_mode_to_sql("real")
            demo = DataManager(self.config)
            demo.sql = SqlDataStore.memory()
            demo._load_mode_to_sql("demo")
            for table in ["monthly_local", "monthly_national", "quarterly_panel", "metadata"]:
                if demo.sql.table_exists(table):
                    demo_df = demo.query_table(table)
                    self.sql.append_dataframe(table, demo_df)
            # metadata 去重（等价于原 concat().drop_duplicates()，按业务列 GROUP BY）
            if self.sql.table_exists("metadata"):
                biz_cols = self.sql.business_columns("metadata")
                if biz_cols:
                    cols_sql = ", ".join(f'"{c}"' for c in biz_cols)
                    self.sql.execute(
                        f"CREATE TABLE metadata_dedup AS SELECT {cols_sql} "
                        f"FROM metadata GROUP BY {cols_sql}"
                    )
                    self.sql.execute("DROP TABLE metadata")
                    self.sql.execute("ALTER TABLE metadata_dedup RENAME TO metadata")
            demo.sql.close()
        else:
            raise DataNotReadyError(f"不支持的数据模式: {mode}")

    # ------------------------------------------------------------------
    # SQL 读取
    # ------------------------------------------------------------------
    def query_table(self, name: str) -> pd.DataFrame:
        """从 SQLite 读取整表（剔除系统列）；表不存在时返回空 DataFrame。"""
        if not self.sql.table_exists(name):
            return pd.DataFrame()
        biz_cols = self.sql.business_columns(name)
        select_cols = ", ".join(f'"{c}"' for c in biz_cols) if biz_cols else "*"
        df = self.sql.query_df(f'SELECT {select_cols} FROM "{name}"')
        if "date" in df.columns:
            df["date"] = ensure_datetime(df["date"])
        return df

    def _read_mode_tables(self, mode: str) -> DataBundle:
        self._load_mode_to_sql(mode)
        if mode == "real":
            notes = ["using_real_data", "loaded_into_sqlite"]
        elif mode == "demo":
            notes = ["using_demo_data", "loaded_into_sqlite"]
        elif mode == "hybrid":
            notes = ["using_hybrid_data", "target_real_features_hybrid", "loaded_into_sqlite"]
        else:
            raise DataNotReadyError(f"不支持的数据模式: {mode}")

        return DataBundle(
            dataset_mode=mode,
            quarterly_target=self.query_table("quarterly_target"),
            monthly_local=self.query_table("monthly_local"),
            monthly_national=self.query_table("monthly_national"),
            quarterly_panel=self.query_table("quarterly_panel"),
            metadata=self.query_table("metadata"),
            notes=notes,
        )

    def initialize(self) -> None:
        mode = self.detect_mode()
        self.bundle = self._read_mode_tables(mode)
        self.logger.info(
            "数据管理器初始化完成 | mode=%s | sqlite=%s",
            self.bundle.dataset_mode, self.sql.db_path,
        )

    def get_bundle(self) -> DataBundle:
        if self.bundle is None:
            self.initialize()
        assert self.bundle is not None
        return self.bundle

    def get_table(self, name: str) -> pd.DataFrame:
        if name not in DataBundle.__dataclass_fields__:
            raise DataNotReadyError(f"未知表名: {name}")
        self.get_bundle()  # 确保数据已加载进 SQLite
        return self.query_table(name).copy()

    # ------------------------------------------------------------------
    # SQL 快照与统计查询
    # ------------------------------------------------------------------
    def get_latest_snapshot(self) -> Dict[str, Any]:
        bundle = self.get_bundle()
        if bundle.quarterly_target.empty:
            return {"status": "empty"}

        latest_date = self.sql.query_one("SELECT MAX(date) FROM quarterly_target")
        latest_frame = self.sql.query_df(
            "SELECT indicator_name FROM quarterly_target WHERE date = ?", (latest_date,)
        )
        local_latest = self.sql.query_one("SELECT MAX(date) FROM monthly_local")
        national_latest = self.sql.query_one("SELECT MAX(date) FROM monthly_national")

        return {
            "dataset_mode": bundle.dataset_mode,
            "latest_target_date": latest_date,
            "latest_target_rows": int(self.sql.query_one(
                "SELECT COUNT(*) FROM quarterly_target WHERE date = ?", (latest_date,), default=0)),
            "target_indicators": sorted(latest_frame["indicator_name"].dropna().unique().tolist()),
            "local_latest_month": local_latest,
            "national_latest_month": national_latest,
            "notes": bundle.notes,
        }

    def get_data_availability(self) -> Dict[str, Any]:
        bundle = self.get_bundle()
        items = []
        for name in bundle.tables():
            if not self.sql.table_exists(name):
                items.append({
                    "name": name, "rows": 0, "columns": 0, "start": None, "end": None,
                    "missing_ratio": 0.0, "indicators": 0, "regions": 0,
                })
                continue
            cols = self.sql.business_columns(name)
            has_date = "date" in cols
            start_end = self.sql.query_df(
                f'SELECT MIN(date) AS start, MAX(date) AS "end" FROM "{name}"'
            ) if has_date else pd.DataFrame({"start": [None], "end": [None]})
            stats = self.sql.query_df(
                f'SELECT COUNT(*) AS rows_, '
                f'{self.sql.missing_null_expression(name)} AS null_count, '
                f'{f"COUNT(DISTINCT indicator_name) AS indicators," if "indicator_name" in cols else "0 AS indicators,"} '
                f'{f"COUNT(DISTINCT region) AS regions" if "region" in cols else "0 AS regions"} '
                f'FROM "{name}"'
            )
            rows = int(stats["rows_"].iloc[0])
            null_count = int(stats["null_count"].iloc[0])
            denom = max(rows * max(len(cols), 1), 1)
            items.append({
                "name": name,
                "rows": rows,
                "columns": len(cols),
                "start": start_end["start"].iloc[0],
                "end": start_end["end"].iloc[0],
                "missing_ratio": float(null_count / denom),
                "indicators": int(stats["indicators"].iloc[0]),
                "regions": int(stats["regions"].iloc[0]),
            })
        return {
            "dataset_mode": bundle.dataset_mode,
            "items": items,
            "notes": bundle.notes,
        }

    def query_indicator(self, table_name: str, indicator_keyword: str) -> pd.DataFrame:
        if table_name not in DataBundle.__dataclass_fields__:
            raise DataNotReadyError(f"未知表名: {table_name}")
        self.get_bundle()  # 确保数据已加载进 SQLite
        cols = self.sql.table_columns(table_name)
        if "indicator_name" not in cols:
            return pd.DataFrame()
        # SQL LIKE 大小写不敏感（ASCII），等价于原 str.contains(case=False)
        df = self.sql.query_df(
            f'SELECT * FROM "{table_name}" '
            f"WHERE LOWER(indicator_name) LIKE '%' || LOWER(?) || '%'",
            (indicator_keyword,),
        )
        if "date" in df.columns:
            df["date"] = ensure_datetime(df["date"])
        return df

    def build_training_signal_overview(self) -> Dict[str, Any]:
        bundle = self.get_bundle()

        def _count_periods(table: str, key_expr: str) -> int:
            if not self.sql.table_exists(table):
                return 0
            return int(self.sql.query_one(
                f"SELECT COUNT(DISTINCT {key_expr}) FROM \"{table}\" WHERE date IS NOT NULL",
                default=0,
            ))

        def _sorted_indicators(table: str) -> List[str]:
            if not self.sql.table_exists(table):
                return []
            df = self.sql.query_df(
                f'SELECT DISTINCT indicator_name FROM "{table}" ORDER BY indicator_name'
            )
            return df["indicator_name"].dropna().astype(str).tolist()

        local_indicators = _sorted_indicators("monthly_local")
        national_indicators = _sorted_indicators("monthly_national")

        return {
            "dataset_mode": bundle.dataset_mode,
            "target_quarters": _count_periods("quarterly_target", sql_quarter_key_expr()),
            "national_months": _count_periods("monthly_national", sql_month_key_expr()),
            "local_months": _count_periods("monthly_local", sql_month_key_expr()),
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
