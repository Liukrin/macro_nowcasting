"""
Convert raw NBS Excel data to project standard long-table format.

Input:
  - data/四川省数据202512.xlsx  (Sichuan quarterly + monthly, 2010-2025)
  - data/国家数据202512.xlsx    (National quarterly + monthly, 2010-2025)

Output (overwrites existing CSVs):
  - data/quarterly_target_real.csv
  - data/monthly_local_features_real.csv
  - data/monthly_national_features_real.csv
  - data/quarterly_feature_panel_real.csv
  - data/metadata_real.csv
"""
from __future__ import annotations

import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from calendar import monthrange

# Fix Windows GBK console encoding
reconfigure = getattr(sys.stdout, "reconfigure", None)
if reconfigure is not None:
    reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"


def parse_quarter_time(val) -> str | None:
    """201003 -> 2010-03-31"""
    try:
        s = str(int(val))
        y, m = int(s[:4]), int(s[4:])
        q_end_month = ((m - 1) // 3 + 1) * 3
        last_day = monthrange(y, q_end_month)[1]
        return f"{y:04d}-{q_end_month:02d}-{last_day:02d}"
    except (ValueError, TypeError):
        return None


def parse_month_time(val) -> str | None:
    """201002 -> 2010-02-28"""
    try:
        s = str(int(val))
        y, m = int(s[:4]), int(s[4:])
        last_day = monthrange(y, m)[1]
        return f"{y:04d}-{m:02d}-{last_day:02d}"
    except (ValueError, TypeError):
        return None


def sf(val) -> float | None:
    """Safe float conversion."""
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def convert_sichuan(df: pd.DataFrame):
    """Convert Sichuan Excel to quarterly_target + monthly_local."""
    print(f"  Processing Sichuan: {df.shape[0]} rows x {df.shape[1]} cols")

    quarterly_rows = []
    monthly_rows = []

    for idx, row in df.iterrows():
        # --- Quarterly indicators (columns 0-11) ---
        time_raw = row.iloc[0]
        if not pd.isna(time_raw):
            date = parse_quarter_time(time_raw)
            if date:
                # Columns 2,4,6,8 are growth INDEX (上年=100), need -100 to get %
                q_indicators = [
                    # (indicator_name, unit, iloc_col, index_adjust)
                    ("GDP_同比增速", "%", 2, True),
                    ("GDP_累计值", "亿元", 1, False),
                    ("第一产业增加值_同比增速", "%", 4, True),
                    ("第一产业增加值_累计值", "亿元", 3, False),
                    ("第二产业增加值_同比增速", "%", 6, True),
                    ("第二产业增加值_累计值", "亿元", 5, False),
                    ("第三产业增加值_同比增速", "%", 8, True),
                    ("第三产业增加值_累计值", "亿元", 7, False),
                    # These columns are already in %, not index
                    ("社会消费品零售总额_同比增速", "%", 9, False),
                    ("固定资产投资（不含农户）_同比增速", "%", 10, False),
                    ("规模以上工业增加值_同比增速", "%", 11, False),
                ]
                for name, unit, ci, is_index in q_indicators:
                    v = sf(row.iloc[ci])
                    if v is not None:
                        if is_index and "增速" in name:
                            v = v - 100.0  # index -> growth rate
                        quarterly_rows.append({
                            "date": date, "region": "四川省",
                            "indicator_name": name, "indicator_value": v,
                            "frequency": "quarterly", "unit": unit,
                            "source_name": "四川省统计局",
                            "source_url": "https://data.stats.gov.cn",
                            "note": "",
                        })

        # --- Monthly indicators (columns 12-17) ---
        month_raw = row.iloc[12]  # 月份 column
        if not pd.isna(month_raw):
            mdate = parse_month_time(month_raw)
            if mdate:
                m_indicators = [
                    # (indicator_name, unit, iloc_col)
                    ("房地产开发投资_累计值", "亿元", 13),
                    ("房地产开发投资_同比增速", "%", 14),
                    ("规模以上工业增加值_同比增速", "%", 15),
                    ("社会消费品零售总额_累计值", "亿元", 16),
                    ("社会消费品零售总额_同比增速", "%", 17),
                ]
                for name, unit, ci in m_indicators:
                    v = sf(row.iloc[ci])
                    if v is not None:
                        monthly_rows.append({
                            "date": mdate, "region": "四川省",
                            "indicator_name": name, "indicator_value": v,
                            "frequency": "monthly", "unit": unit,
                            "source_name": "四川省统计局",
                            "source_url": "https://data.stats.gov.cn",
                            "note": "",
                        })

    qt = pd.DataFrame(quarterly_rows)
    ml = pd.DataFrame(monthly_rows)
    if not qt.empty:
        qt = qt.sort_values(["date", "indicator_name"]).reset_index(drop=True)
    if not ml.empty:
        ml = ml.sort_values(["date", "indicator_name"]).reset_index(drop=True)
    return qt, ml


def convert_national(df: pd.DataFrame):
    """Convert National Excel to quarterly_panel + monthly_national."""
    print(f"  Processing National: {df.shape[0]} rows x {df.shape[1]} cols")

    quarterly_rows = []
    monthly_rows = []

    for idx, row in df.iterrows():
        # --- Quarterly indicators (columns 0-17) ---
        time_raw = row.iloc[0]
        if not pd.isna(time_raw):
            date = parse_quarter_time(time_raw)
            if date:
                q_indicators = [
                    # (indicator_name, unit, iloc_col, index_adjust)
                    ("GDP_累计值", "亿元", 2, False),
                    ("GDP_同比增速", "%", 3, True),       # index: 112.2 -> 12.2%
                    ("第一产业增加值_累计值", "亿元", 4, False),
                    ("第一产业增加值_同比增速", "%", 5, True),   # index
                    ("第二产业增加值_累计值", "亿元", 6, False),
                    ("第二产业增加值_同比增速", "%", 7, True),   # index
                    ("第三产业增加值_累计值", "亿元", 8, False),
                    ("第三产业增加值_同比增速", "%", 9, True),   # index
                    ("工业增加值_累计值", "亿元", 10, False),
                    ("工业增加值_同比增速", "%", 11, True),      # index
                    # These are already in %, not index
                    ("房地产开发投资_累计值", "亿元", 12, False),
                    ("房地产开发投资_同比增速", "%", 13, False),
                    ("社会消费品零售总额_累计值", "亿元", 14, False),
                    ("社会消费品零售总额_同比增速", "%", 15, False),
                    ("固定资产投资（不含农户）_累计值", "亿元", 16, False),
                    ("固定资产投资（不含农户）_同比增速", "%", 17, False),
                ]
                for name, unit, ci, is_index in q_indicators:
                    v = sf(row.iloc[ci])
                    if v is not None:
                        if is_index and "增速" in name:
                            v = v - 100.0
                    if v is not None:
                        quarterly_rows.append({
                            "date": date, "region": "全国",
                            "indicator_name": name, "indicator_value": v,
                            "frequency": "quarterly", "unit": unit,
                            "source_name": "国家统计局",
                            "source_url": "https://data.stats.gov.cn",
                            "note": "",
                        })

        # --- Monthly indicators (columns 25-30 on the right side) ---
        period_str = None
        if len(row) > 26:
            period_str = str(row.iloc[26]).strip() if not pd.isna(row.iloc[26]) else None
        if period_str and any(c.isdigit() for c in period_str):
            match = re.match(r"(\d{4})年(\d{1,2})月", period_str)
            if match:
                y, m = int(match.group(1)), int(match.group(2))
                last_day = monthrange(y, m)[1]
                mdate = f"{y:04d}-{m:02d}-{last_day:02d}"

                m_indicators = [
                    ("固定资产投资（不含农户）_同比增速", "%", 27),
                    ("房地产开发投资_累计值", "亿元", 28),
                    ("社会消费品零售总额_累计值", "亿元", 29),
                    ("工业增加值_同比增速", "%", 30),
                ]
                for name, unit, ci in m_indicators:
                    if ci < len(row):
                        v = sf(row.iloc[ci])
                        if v is not None:
                            monthly_rows.append({
                                "date": mdate, "region": "全国",
                                "indicator_name": name, "indicator_value": v,
                                "frequency": "monthly", "unit": unit,
                                "source_name": "国家统计局",
                                "source_url": "https://data.stats.gov.cn",
                                "note": "",
                            })

    nq = pd.DataFrame(quarterly_rows)
    nm = pd.DataFrame(monthly_rows)
    if not nq.empty:
        nq = nq.sort_values(["date", "indicator_name"]).reset_index(drop=True)
    if not nm.empty:
        nm = nm.sort_values(["date", "indicator_name"]).reset_index(drop=True)
    return nq, nm


def build_quarterly_panel(monthly_local: pd.DataFrame, monthly_national: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly data into quarterly panel with mean/last/std/min/max/trend."""
    all_m = pd.concat([monthly_local, monthly_national], ignore_index=True)
    all_m["date"] = pd.to_datetime(all_m["date"])
    all_m["quarter_end"] = all_m["date"].dt.to_period("Q").dt.to_timestamp("Q")

    rows = []
    for (region, indicator, q_end), grp in all_m.groupby(["region", "indicator_name", "quarter_end"]):
        grp = grp.sort_values("date")
        vals = grp["indicator_value"].astype(float)
        n = len(vals)
        agg = {
            "mean": float(vals.mean()),
            "last": float(vals.iloc[-1]),
            "std": float(vals.std()) if n > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "trend": float(np.polyfit(np.arange(n), vals.to_numpy(), 1)[0]) if n > 1 else 0.0,
        }
        for suffix, v in agg.items():
            rows.append({
                "date": q_end, "region": region,
                "indicator_name": f"{indicator}_{suffix}",
                "indicator_value": v,
                "frequency": "quarterly", "unit": "%",
                "source_name": "聚合计算", "source_url": "-",
                "note": f"quarterly aggregation from monthly {indicator}",
            })
    return pd.DataFrame(rows).sort_values(["date", "region", "indicator_name"]).reset_index(drop=True)


def build_metadata(qt: pd.DataFrame, ml: pd.DataFrame, mn: pd.DataFrame) -> pd.DataFrame:
    """Generate metadata from indicator names."""
    all_indicators = set()
    for df in [qt, ml, mn]:
        if not df.empty and "indicator_name" in df.columns:
            all_indicators.update(df["indicator_name"].unique())

    cat_map = {
        "GDP": "目标", "产业": "产业", "工业": "工业", "投资": "投资",
        "消费": "消费", "房地产": "房地产", "PMI": "PMI",
        "PPI": "价格", "CPI": "价格", "M2": "金融", "用电": "能源",
    }

    rows = []
    for name in sorted(all_indicators):
        cat = "其他"
        for k, c in cat_map.items():
            if k in name:
                cat = c; break
        rows.append({
            "original_name": name, "standard_name": name,
            "category": cat,
            "frequency": "quarterly" if ("GDP" in name or "产业" in name) else "monthly",
            "is_target": name == "GDP_同比增速",
            "is_yoy": "同比" in name or "增速" in name,
            "is_cumulative": "累计" in name,
            "leading_indicator": name != "GDP_同比增速",
            "data_quality": "high",
            "notes": "converted from NBS raw Excel",
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Raw Excel -> Standard Long Table Conversion")
    print("=" * 60)

    # 1. Read raw Excel
    print("\n[1/5] Reading raw Excel files...")
    sc_raw = pd.read_excel(DATA_DIR / "四川省数据202512.xlsx")
    nat_raw = pd.read_excel(DATA_DIR / "国家数据202512.xlsx", sheet_name=0)
    print(f"  Sichuan: {sc_raw.shape}")
    print(f"  National: {nat_raw.shape}")

    # 2. Convert to long format
    print("\n[2/5] Converting to standardized long tables...")
    quarterly_target, monthly_local = convert_sichuan(sc_raw)
    national_q, national_m = convert_national(nat_raw)

    print(f"  quarterly_target:      {len(quarterly_target):5d} rows")
    print(f"  monthly_local:         {len(monthly_local):5d} rows")
    print(f"  national_quarterly:    {len(national_q):5d} rows (info only)")
    print(f"  national_monthly:      {len(national_m):5d} rows")

    # 3. Merge existing PMI data
    print("\n[3/5] Merging existing PMI data...")
    existing_csv = DATA_DIR / "monthly_national_features_real.csv"
    if existing_csv.exists():
        old = pd.read_csv(existing_csv)
        pmi = old[old["indicator_name"].astype(str).str.startswith("PMI_")]
        monthly_national = pd.concat([national_m, pmi], ignore_index=True)
        print(f"  Merged {len(pmi)} PMI rows from existing CSV")
    else:
        monthly_national = national_m

    # 4. Build quarterly panel
    print("\n[4/5] Building quarterly panel from monthly data...")
    quarterly_panel = build_quarterly_panel(monthly_local, monthly_national)
    print(f"  quarterly_panel:       {len(quarterly_panel):5d} rows")

    # 5. Build metadata
    print("\n[5/5] Generating metadata...")
    metadata = build_metadata(quarterly_target, monthly_local, monthly_national)
    print(f"  metadata:              {len(metadata):5d} rows")

    # 6. Save
    print("\nWriting CSVs...")
    outputs = {
        "quarterly_target_real.csv": quarterly_target,
        "monthly_local_features_real.csv": monthly_local,
        "monthly_national_features_real.csv": monthly_national,
        "quarterly_feature_panel_real.csv": quarterly_panel,
        "metadata_real.csv": metadata,
    }

    for fname, df in outputs.items():
        path = DATA_DIR / fname
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  [OK] {fname}: {df.shape[0]} rows x {df.shape[1]} cols")

    # 7. Summary
    print("\n" + "=" * 60)
    print("Conversion complete! Data summary:")
    if not quarterly_target.empty:
        print(f"  quarterly_target:  {len(quarterly_target)} rows")
        print(f"    date range: {quarterly_target['date'].min()} ~ {quarterly_target['date'].max()}")
        print(f"    unique quarters: {quarterly_target['date'].nunique()}")
        print(f"    indicators: {quarterly_target['indicator_name'].nunique()}")
    if not monthly_local.empty:
        print(f"  monthly_local:     {len(monthly_local)} rows")
        print(f"    date range: {monthly_local['date'].min()} ~ {monthly_local['date'].max()}")
    if not monthly_national.empty:
        print(f"  monthly_national:  {len(monthly_national)} rows")
        print(f"    date range: {monthly_national['date'].min()} ~ {monthly_national['date'].max()}")
    if not quarterly_panel.empty:
        print(f"  quarterly_panel:   {len(quarterly_panel)} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
