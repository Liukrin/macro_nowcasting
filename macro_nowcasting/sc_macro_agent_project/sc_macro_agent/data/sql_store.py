"""
SQLite 数据存储层。

职责：
- 将标准化长表（DataFrame）加载进 SQLite（持久化 .db 文件或内存）
- 提供参数化 SQL 查询，结果以 DataFrame 返回
- 提供季度/月度时间键的 SQL 表达式常量，供各模块统一引用

说明：
- SQLite 未内置 sqrt，此处通过 create_function 注册，
  以支持样本标准差等统计聚合直接在 SQL 中完成。
- 日期列统一存为 'YYYY-MM-DD' 文本，保证字典序比较与时间序一致。
"""
from __future__ import annotations

import math
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd

from ..logging_utils import get_logger


def _sqrt_udf(x: Any) -> Optional[float]:
    """SQLite sqrt UDF：NULL 进 NULL 出，负数按 0 处理（浮点噪声防护）。"""
    if x is None:
        return None
    v = float(x)
    return math.sqrt(v) if v > 0 else 0.0


def sql_quarter_end_expr(date_col: str = "date") -> str:
    """SQL 表达式：日期 → 所在季度的季末日（'YYYY-MM-DD'），与 pandas to_timestamp('Q') 对齐。"""
    return (
        f"strftime('%Y', {date_col}) || '-' || CASE "
        f"((CAST(strftime('%m', {date_col}) AS INTEGER) - 1) / 3) + 1 "
        f"WHEN 1 THEN '03-31' WHEN 2 THEN '06-30' "
        f"WHEN 3 THEN '09-30' ELSE '12-31' END"
    )


def sql_quarter_key_expr(date_col: str = "date") -> str:
    """SQL 表达式：日期 → 'YYYYQN' 季度键，用于 COUNT(DISTINCT) 统计季度期数。"""
    return (
        f"strftime('%Y', {date_col}) || 'Q' || "
        f"(((CAST(strftime('%m', {date_col}) AS INTEGER) - 1) / 3) + 1)"
    )


def sql_month_key_expr(date_col: str = "date") -> str:
    """SQL 表达式：日期 → 'YYYY-MM' 月度键，用于 COUNT(DISTINCT) 统计月度期数。"""
    return f"strftime('%Y-%m', {date_col})"


# 业务表清单（与 DataBundle.tables() 对齐）
MANAGED_TABLES: Sequence[str] = (
    "quarterly_target",
    "monthly_local",
    "monthly_national",
    "quarterly_panel",
    "metadata",
)

_IDENT_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

# 系统列：由存储层自动维护，不属于业务数据（如入表顺序号）
SYSTEM_COLUMNS: Sequence[str] = ("sql_seq",)


def _quote_ident(name: str) -> str:
    """表名/列名白名单校验 + 双引号包裹，防止 SQL 注入。"""
    if not name or not all(ch in _IDENT_SAFE for ch in name):
        raise ValueError(f"非法标识符: {name!r}")
    return f'"{name}"'


class SqlDataStore:
    """基于 SQLite 的数据存储封装。

    - 文件模式：持久化到 .db 文件，可用任意 SQLite 客户端检视
    - 内存模式：`SqlDataStore.memory()`，供一次性计算（审计/特征构建）使用
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:") -> None:
        self.db_path = Path(db_path)
        self.logger = get_logger("sc_macro_agent.sql_store")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.create_function("sqrt", 1, _sqrt_udf)

    # ------------------------------------------------------------------
    # 构造与生命周期
    # ------------------------------------------------------------------
    @classmethod
    def memory(cls) -> "SqlDataStore":
        return cls(":memory:")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
        """写入前规范化：日期列转 'YYYY-MM-DD' 文本，便于 SQL 比较与 strftime。
        数值列 indicator_value 强制转 float（非法值→NULL），与 to_numeric 语义一致。"""
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.strftime("%Y-%m-%d")
        if "indicator_value" in out.columns and not pd.api.types.is_numeric_dtype(out["indicator_value"]):
            out["indicator_value"] = pd.to_numeric(out["indicator_value"], errors="coerce")
        return out

    def load_dataframe(self, table_name: str, df: pd.DataFrame) -> int:
        """REPLACE 方式写入整表。空表则删除同名表。返回写入行数。
        附加 sql_seq 列记录入表顺序（pandas to_sql 生成的表无隐式 rowid）。"""
        ident = _quote_ident(table_name)
        with self._lock:
            self._conn.execute(f"DROP TABLE IF EXISTS {ident}")
            self._conn.commit()
            if df is None or df.empty:
                return 0
            prepared = self._prepare_frame(df)
            prepared.insert(0, "sql_seq", range(len(prepared)))
            prepared.to_sql(table_name, self._conn, if_exists="replace", index=False)
        return int(len(df))

    def load_csv(self, csv_path: Union[str, Path], table_name: str,
                 parse_dates: Sequence[str] = ("date",)) -> int:
        """读取 CSV 并写入表。存在的日期列解析为 datetime 后再规范化。"""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {path}")
        df = pd.read_csv(path)
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return self.load_dataframe(table_name, df)

    def append_dataframe(self, table_name: str, df: pd.DataFrame) -> int:
        """向已存在表追加行（用于 hybrid 模式合并 demo + real）。
        sql_seq 从当前最大值续编，保持全表顺序唯一。"""
        if df is None or df.empty:
            return 0
        prepared = self._prepare_frame(df)
        ident = _quote_ident(table_name)
        with self._lock:
            if "sql_seq" in self.table_columns(table_name):
                cur_max = self._conn.execute(
                    f'SELECT COALESCE(MAX(sql_seq), -1) FROM {ident}').fetchone()[0]
                prepared.insert(0, "sql_seq", range(int(cur_max) + 1, int(cur_max) + 1 + len(prepared)))
            prepared.to_sql(table_name, self._conn, if_exists="append", index=False)
        return int(len(df))

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def query_df(self, sql: str, params: Optional[Sequence[Any]] = None) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(sql, self._conn, params=params)

    def query_one(self, sql: str, params: Optional[Sequence[Any]] = None,
                  default: Any = None) -> Any:
        """执行查询并返回首行首列；无结果时返回 default。"""
        with self._lock:
            row = self._conn.execute(sql, params or ()).fetchone()
        if row is None or row[0] is None:
            return default
        return row[0]

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> int:
        """执行 DML/DDL，返回受影响行数。"""
        with self._lock:
            cur = self._conn.execute(sql, params or ())
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    def table_exists(self, table_name: str) -> bool:
        ident = _quote_ident(table_name)
        found = self.query_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table_name,),
        )
        return found is not None and bool(ident)

    def table_columns(self, table_name: str) -> List[str]:
        ident = _quote_ident(table_name)
        df = self.query_df(f"PRAGMA table_info({ident})")
        return df["name"].tolist() if not df.empty else []

    def business_columns(self, table_name: str) -> List[str]:
        """业务列（排除系统列如 sql_seq）。"""
        return [c for c in self.table_columns(table_name) if c not in SYSTEM_COLUMNS]

    def row_count(self, table_name: str) -> int:
        ident = _quote_ident(table_name)
        return int(self.query_one(f"SELECT COUNT(*) FROM {ident}", default=0))

    def missing_null_expression(self, table_name: str) -> str:
        """生成统计全表 NULL 总数的 SQL 表达式（各列 CASE WHEN 求和）。"""
        cols = self.table_columns(table_name)
        if not cols:
            return "0"
        parts = [f"SUM(CASE WHEN {_quote_ident(c)} IS NULL THEN 1 ELSE 0 END)" for c in cols]
        return " + ".join(f"COALESCE({p}, 0)" for p in parts)
