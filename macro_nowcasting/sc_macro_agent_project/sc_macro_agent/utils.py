"""
通用工具函数。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

import numpy as np
import pandas as pd


def ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def quarter_end(series: pd.Series) -> pd.Series:
    dt = ensure_datetime(series)
    return cast(pd.Series, dt.dt.to_period("Q").dt.to_timestamp(how="end"))


def month_end(series: pd.Series) -> pd.Series:
    dt = ensure_datetime(series)
    return cast(pd.Series, dt.dt.to_period("M").dt.to_timestamp(how="end"))


def safe_slug(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[\s/\\]+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "", text)
    return text.strip("_").lower() or "unnamed"


def winsorize_series(series: pd.Series, q: float = 0.01) -> pd.Series:
    if series.dropna().empty:
        return series
    lower = series.quantile(q)
    upper = series.quantile(1 - q)
    return series.clip(lower, upper)


def safe_trend(values: Sequence[float]) -> float:
    arr = pd.Series(values).dropna().astype(float).to_numpy()
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr))
    coef = np.polyfit(x, arr, 1)[0]
    return float(coef)


def safe_change(first: Optional[float], last: Optional[float]) -> float:
    if first is None or last is None:
        return 0.0
    if pd.isna(first) or pd.isna(last):
        return 0.0
    return float(last - first)


def safe_pct_change(first: Optional[float], last: Optional[float]) -> float:
    if first is None or last is None:
        return 0.0
    if pd.isna(first) or pd.isna(last) or abs(first) < 1e-8:
        return 0.0
    return float((last - first) / abs(first))


def recent_non_na_count(values: Sequence[float]) -> int:
    return int(pd.Series(values).notna().sum())


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(arr_true - arr_pred)))


def mape(y_true: Sequence[float], y_pred: Sequence[float], eps: float = 1e-6) -> float:
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.clip(np.abs(arr_true), eps, None)
    return float(np.mean(np.abs((arr_true - arr_pred) / denom)) * 100.0)


def smape(y_true: Sequence[float], y_pred: Sequence[float], eps: float = 1e-6) -> float:
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.clip((np.abs(arr_true) + np.abs(arr_pred)) / 2.0, eps, None)
    return float(np.mean(np.abs(arr_true - arr_pred) / denom) * 100.0)


def r2_like(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((arr_true - arr_pred) ** 2))
    ss_tot = float(np.sum((arr_true - np.mean(arr_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def metrics_dict(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2_like(y_true, y_pred),
    }


def top_abs_series(series: pd.Series, top_n: int = 20) -> pd.Series:
    if series.empty:
        return series
    return series.abs().sort_values(ascending=False).head(top_n)


def flatten_dict(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, new_key))
        else:
            out[new_key] = value
    return out


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def json_dumps(obj: Any, ensure_ascii: bool = False, indent: int = 2) -> str:
    if is_dataclass(obj) and not isinstance(obj, type):
        payload: Any = asdict(obj)
    else:
        payload = obj
    return json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent, default=_json_default)


def save_text(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def save_json(path: str | Path, payload: Any) -> Path:
    return save_text(path, json_dumps(payload))

def safe_mean(values: Sequence[float]) -> float:
    ser = pd.Series(values, dtype=float)
    if ser.dropna().empty:
        return 0.0
    return float(ser.mean())

def safe_std(values: Sequence[float]) -> float:
    ser = pd.Series(values, dtype=float)
    if len(ser.dropna()) < 2:
        return 0.0
    return float(ser.std())

def safe_min(values: Sequence[float]) -> float:
    ser = pd.Series(values, dtype=float)
    if ser.dropna().empty:
        return 0.0
    return float(ser.min())

def safe_max(values: Sequence[float]) -> float:
    ser = pd.Series(values, dtype=float)
    if ser.dropna().empty:
        return 0.0
    return float(ser.max())

def safe_last(values: Sequence[float]) -> float:
    ser = pd.Series(values, dtype=float).dropna()
    if ser.empty:
        return 0.0
    return float(ser.iloc[-1])

def rank_features_by_variance(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    numeric = [c for c in df.select_dtypes(include=["number"]).columns if c not in exclude]
    if not numeric:
        return []
    stats = df[numeric].var().sort_values(ascending=False)
    return list(stats.index)

def cap_feature_count(df: pd.DataFrame, protected: Iterable[str], max_features: int) -> pd.DataFrame:
    protected = list(protected)
    current_features = [c for c in df.columns if c not in protected]
    if len(current_features) <= max_features:
        return df
    numeric_features = [c for c in current_features if c in df.select_dtypes(include=["number"]).columns]
    non_numeric_features = [c for c in current_features if c not in numeric_features]
    ranked = rank_features_by_variance(df[numeric_features])
    numeric_slots = max(0, max_features - len(non_numeric_features))
    kept = non_numeric_features + ranked[:numeric_slots]
    return df[protected + kept].copy()

def pretty_quarter(dt_like: Any) -> str:
    ts = pd.to_datetime(dt_like)
    return f"{ts.year}Q{ts.quarter}"

def infer_data_mode(real_available: bool, demo_available: bool, requested: str) -> str:
    requested = requested.lower().strip()
    if requested in {"real", "demo", "hybrid"}:
        return requested
    if real_available:
        return "real"
    if demo_available:
        return "demo"
    return "real"

def pad_list(values: List[Any], size: int, pad_value: Any = None) -> List[Any]:
    if len(values) >= size:
        return values[:size]
    return values + [pad_value] * (size - len(values))

def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    s1 = pd.Series(a, dtype=float)
    s2 = pd.Series(b, dtype=float)
    pair = pd.concat([s1, s2], axis=1).dropna()
    if len(pair) < 2:
        return 0.0
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))

def nan_to_zero_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["number"]).columns:
        out[col] = out[col].fillna(0.0)
    return out

def sorted_unique(values: Sequence[Any]) -> List[Any]:
    return sorted(pd.Series(list(values)).dropna().unique().tolist())


def decumulate_ytd(cum_value_series: pd.Series) -> pd.Series:
    """\n    从累计值序列反推单季值，再计算单季同比。\n\n    方法：累计值差分法。\n      单季值_Q1 = 累计值_Q1\n      单季值_Qn = 累计值_Qn - 累计值_Q(n-1)\n      单季同比_t = 单季值_t / 单季值_{t-4} - 1\n\n    注意：基于名义累计值差分，得到名义增速，与统计局不变价实际增速\n    存在系统性偏差，仅用于模型评估的相对比较。\n\n    Args:\n        cum_value_series: 按时间排序的累计值序列（如 GDP 累计值，亿元）。\n            Index 为日期，含 quarter 属性。\n\n    Returns:\n        单季同比序列（%），缺失季度为 NaN。\n    """
    result = pd.Series(np.nan, index=cum_value_series.index, dtype=float)
    if not hasattr(cum_value_series.index, 'quarter'):
        return result

    # Step 1: Decumulate to single-quarter values
    single_q = pd.Series(np.nan, index=cum_value_series.index, dtype=float)
    for i, idx in enumerate(cum_value_series.index):
        val = cum_value_series.iloc[i]
        if pd.isna(val):
            continue
        q = idx.quarter
        if q == 1:
            single_q.iloc[i] = float(val)
        else:
            # Q2-Q4: current cumulative - previous quarter cumulative
            if i >= 1 and not pd.isna(cum_value_series.iloc[i-1]):
                single_q.iloc[i] = float(val) - float(cum_value_series.iloc[i-1])

    # Step 2: Compute single-quarter YoY = single_q_t / single_q_{t-4} - 1
    for i, idx in enumerate(cum_value_series.index):
        if i < 4 or pd.isna(single_q.iloc[i]):
            continue
        sq_prev = single_q.iloc[i-4]
        sq_curr = single_q.iloc[i]
        if pd.notna(sq_prev) and pd.notna(sq_curr) and abs(sq_prev) > 1e-8:
            result.iloc[i] = (sq_curr / sq_prev - 1.0) * 100.0

    return result


class Diagnostics:
    def feature_target_correlations(self, panel: pd.DataFrame, target_col: str = "target_value", top_n: int = 20) -> List[Dict[str, Any]]:
        if panel.empty or target_col not in panel.columns:
            return []
        rows: List[Dict[str, Any]] = []
        for col in panel.select_dtypes(include=["number"]).columns:
            if col == target_col:
                continue
            rows.append({
                "feature": col,
                "corr_with_target": safe_corr(panel[col].tolist(), panel[target_col].tolist()),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return []
        df["abs_corr"] = df["corr_with_target"].abs()
        df = df.sort_values("abs_corr", ascending=False).head(top_n)
        return cast(List[Dict[str, Any]], df.to_dict("records"))

    def residual_summary(self, y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, Any]:
        true = pd.Series(y_true, dtype=float)
        pred = pd.Series(y_pred, dtype=float)
        resid = true - pred
        return {
            "count": int(len(resid)),
            "mean": float(resid.mean()),
            "std": float(resid.std()) if len(resid) > 1 else 0.0,
            "min": float(resid.min()),
            "max": float(resid.max()),
            "median": float(resid.median()),
        }
