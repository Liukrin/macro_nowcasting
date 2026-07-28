"""
Phase 0 ETL v2: Simplified, robust date parsing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import re

DATA_DIR = Path(__file__).resolve().parent / "data"

def parse_any_date(val):
    """Parse date from various formats. Returns datetime or NaT."""
    if pd.isna(val):
        return pd.NaT
    try:
        if isinstance(val, (int, float, np.integer, np.floating)):
            v = int(float(val))
            # YYYYMM integer (e.g., 201003, 202512) — check FIRST
            # because values like 201002 > 40000 overlap with Excel serial range
            y, m = divmod(v, 100)
            if 1900 <= y <= 2100 and 1 <= m <= 12:
                return datetime(y, m, 1)
            # Excel serial number (days since 1899-12-30, range ~40000-48000 for 2010-2030)
            if 40000 <= v <= 60000:
                d = datetime(1899, 12, 30) + timedelta(days=v)
                return d
        # String
        s = str(val).strip()
        m = re.match(r'(\d{4})\s*年\s*(\d{1,2})\s*月', s)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), 1)
        for fmt in ['%Y-%m', '%Y-%m-%d', '%Y%m', '%Y.%m']:
            try:
                return datetime.strptime(s, fmt)
            except:
                pass
    except:
        pass
    return pd.NaT

def pq(dt):
    """Quarter label"""
    if pd.isna(dt): return 'NaT'
    try:
        return f'{dt.year}Q{(dt.month-1)//3+1}'
    except:
        return str(dt)

# ============================================================
print("="*60)
print("PHASE 0: Sichuan Data Extraction")
print("="*60)

sc = pd.read_excel(DATA_DIR / "四川省数据202512.xlsx", sheet_name=0)
sc_cols = sc.columns.tolist()
print(f"Columns: {sc_cols}")

# Monthly extraction (col 12 = time axis for monthly indicators)
m_time_raw = sc[sc_cols[12]]
m_dates = [parse_any_date(v) for v in m_time_raw]
m_time = pd.to_datetime(m_dates)
print(f"\nMonthly date range: {m_time.min().date()} ~ {m_time.max().date()}")
print(f"Monthly rows: {(~pd.isna(m_time)).sum()}")

# Build Sichuan monthly long table
sc_monthly = pd.DataFrame()
sc_monthly['date'] = m_time
sc_monthly = sc_monthly[pd.notna(sc_monthly['date'])].copy()

sc_monthly['房地产开发投资_累计同比'] = sc[sc_cols[14]]
sc_monthly['规模以上工业增加值_累计同比'] = sc[sc_cols[15]]
sc_monthly['社会消费品零售总额_累计同比'] = sc[sc_cols[17]]

# Melt
sc_melt = sc_monthly.melt(id_vars=['date'], var_name='indicator_name', value_name='indicator_value')
sc_melt['region'] = '四川省'
sc_melt['frequency'] = 'monthly'
sc_melt = sc_melt.dropna(subset=['indicator_value'])
sc_melt = sc_melt[sc_melt['indicator_value'].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))]

print(f"\nSichuan monthly melt: {len(sc_melt)} rows")
for ind in sorted(sc_melt['indicator_name'].unique()):
    n = len(sc_melt[sc_melt['indicator_name']==ind])
    dmin = sc_melt[sc_melt['indicator_name']==ind]['date'].min().date()
    dmax = sc_melt[sc_melt['indicator_name']==ind]['date'].max().date()
    print(f"  {ind}: {n} obs, {dmin} ~ {dmax}")

# Quarterly extraction (col 0 = time axis)
q_time_raw = sc[sc_cols[0]]
q_dates = [parse_any_date(v) for v in q_time_raw]
# For quarterly, if YYYYMM, convert to quarter-end
q_dt_list = []
for v in q_time_raw:
    if pd.isna(v):
        q_dt_list.append(pd.NaT)
        continue
    try:
        vi = int(float(v))
        y, m = divmod(vi, 100)
        if m == 3: q_dt_list.append(datetime(y, 3, 31))
        elif m == 6: q_dt_list.append(datetime(y, 6, 30))
        elif m == 9: q_dt_list.append(datetime(y, 9, 30))
        elif m == 12: q_dt_list.append(datetime(y, 12, 31))
        else: q_dt_list.append(pd.NaT)
    except:
        q_dt_list.append(pd.NaT)

q_time = pd.to_datetime(q_dt_list)
print(f"\nQuarterly date range: {q_time.min().date()} ~ {q_time.max().date()}")
print(f"Quarterly rows: {(~pd.isna(q_time)).sum()}")

# Build Sichuan quarterly long table
sc_quarterly = pd.DataFrame()
sc_quarterly['date'] = q_time
sc_quarterly = sc_quarterly[pd.notna(sc_quarterly['date'])].copy()

# Col 1=GDP累计值, Col 2=GDP累计同比(指数形式, 117.7=+17.7% → 转为实际增速 17.7)
sc_quarterly['GDP_累计值'] = sc[sc_cols[1]]
# Save as GDP_同比增速 (matching config target_indicator) and GDP_累计值 for decumulation
sc_quarterly['GDP_同比增速'] = sc[sc_cols[2]].apply(lambda x: float(x) - 100.0 if pd.notna(x) else np.nan)

# Remove the intermediate column to avoid confusion
sc_quarterly = sc_quarterly.drop(columns=['GDP_累计同比'], errors='ignore')

# Melt
sc_q_melt = sc_quarterly.melt(id_vars=['date'], var_name='indicator_name', value_name='indicator_value')
sc_q_melt['region'] = '四川省'
sc_q_melt['frequency'] = 'quarterly'
sc_q_melt = sc_q_melt.dropna(subset=['indicator_value'])
sc_q_melt = sc_q_melt[sc_q_melt['indicator_value'].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))]

print(f"\nSichuan quarterly melt: {len(sc_q_melt)} rows")
for ind in sorted(sc_q_melt['indicator_name'].unique()):
    n = len(sc_q_melt[sc_q_melt['indicator_name']==ind])
    dmin = sc_q_melt[sc_q_melt['indicator_name']==ind]['date'].min().date()
    dmax = sc_q_melt[sc_q_melt['indicator_name']==ind]['date'].max().date()
    print(f"  {ind}: {n} obs, {dmin} ~ {dmax}")

# Save
sc_q_melt.to_csv(DATA_DIR / "quarterly_target_real.csv", index=False, encoding='utf-8-sig')
sc_melt.to_csv(DATA_DIR / "monthly_local_features_real.csv", index=False, encoding='utf-8-sig')
print("\nSaved Sichuan quarterly and monthly CSVs.")

# ============================================================
print("\n" + "="*60)
print("PHASE 0: National Data Extraction")
print("="*60)

nat = pd.read_excel(DATA_DIR / "国家数据202512.xlsx", sheet_name=1)
nat_cols = nat.columns.tolist()
print(f"Monthly columns: {nat_cols}")

# Patch 2: Row 189 (Excel serial 45992 = 2025-12) has all-NaN data — drop it.
n_time_raw = nat[nat_cols[0]]
n_all_nan = nat[nat_cols[1]].isna() & nat[nat_cols[2]].isna() & nat[nat_cols[3]].isna() & nat[nat_cols[4]].isna()
empty_rows = n_all_nan.sum()
if empty_rows > 0:
    print(f"Patch 2: Dropping {empty_rows} empty row(s) (indices: {nat.index[n_all_nan].tolist()})")
    nat = nat[~n_all_nan].copy()
    n_time_raw = nat[nat_cols[0]]

n_dates = [parse_any_date(v) for v in n_time_raw]
n_time = pd.to_datetime(n_dates)
valid = ~pd.isna(n_time)
print(f"National monthly: {valid.sum()} rows, {n_time[valid].min().date()} ~ {n_time[valid].max().date()}")

# Patch 1: Check for duplicate dates and compare values
date_series = pd.Series(n_dates, index=nat.index)
dup_dates = date_series[date_series.duplicated(keep=False)]
if len(dup_dates) > 0:
    print(f"\nPatch 1: Found {len(dup_dates)} rows with duplicate dates:")
    for dt in sorted(set(d for d in dup_dates if pd.notna(d))):
        idxs = date_series[date_series == dt].index.tolist()
        print(f"  {dt.strftime('%Y-%m-%d')}: rows {idxs}")
        for idx in idxs:
            vals = [nat.loc[idx, nat_cols[j]] for j in range(1, 5)]
            print(f"    row {idx}: {vals}")
        # Check if values are identical
        vals_list = [[nat.loc[idx, nat_cols[j]] for j in range(1, 5)] for idx in idxs]
        all_same = all(v == vals_list[0] for v in vals_list)
        if all_same:
            print(f"    -> Values identical, keeping first occurrence")
        else:
            print(f"    -> Values DIFFER (likely vintage difference), keeping later row (revised value)")
    # Dedup: keep last occurrence for each date
    nat = nat.drop_duplicates(subset=['_date_tmp'], keep='last') if '_date_tmp' in nat.columns else nat
else:
    print("Patch 1: No duplicate dates found. OK.")

# Print date parse issues
bad = [(i, nat[nat_cols[0]].iloc[i]) for i in range(len(n_time_raw)) if pd.isna(n_dates[i])]
if bad:
    print(f"Date parse failures ({len(bad)}):")
    for i, v in bad[:10]:
        print(f"  row {i}: {repr(v)}")

# Patch 3: Build national monthly with correct statistical calibers
#   col1 固定资产投资额累计增长(%)  → YTD cumulative YoY   → _ytd_yoy
#   col2 房地产投资累计值(亿元)      → cumulative level      → excluded from features
#   col3 社会消费品零售总额累计值(亿元) → cumulative level   → excluded from features
#   col4 工业增加值同比增长(%)       → monthly YoY (当月同比) → _mom_yoy
nat_monthly = pd.DataFrame()
nat_monthly['date'] = n_time
nat_monthly = nat_monthly[valid].copy()
nat_idx = nat.index[valid]

# Store as long-format with correct indicator names reflecting caliber
rows = []
for idx in nat_idx:
    dt = n_dates[idx] if idx < len(n_dates) else pd.NaT
    if pd.isna(dt): continue
    # Col1: YTD cumulative YoY
    v1 = nat.loc[idx, nat_cols[1]]
    if pd.notna(v1):
        rows.append({'date': dt, 'indicator_name': '固定资产投资（不含农户）_累计同比_ytd_yoy', 'indicator_value': float(v1), 'region': '全国', 'frequency': 'monthly', 'caliber': 'ytd_yoy'})
    # Col2: cumulative level — save but mark as excluded from feature pool
    v2 = nat.loc[idx, nat_cols[2]]
    if pd.notna(v2):
        rows.append({'date': dt, 'indicator_name': '房地产开发投资_累计值', 'indicator_value': float(v2), 'region': '全国', 'frequency': 'monthly', 'caliber': 'cum_level'})
    # Col3: cumulative level — save but mark as excluded
    v3 = nat.loc[idx, nat_cols[3]]
    if pd.notna(v3):
        rows.append({'date': dt, 'indicator_name': '社会消费品零售总额_累计值', 'indicator_value': float(v3), 'region': '全国', 'frequency': 'monthly', 'caliber': 'cum_level'})
    # Col4: monthly YoY (NOT cumulative!)
    v4 = nat.loc[idx, nat_cols[4]]
    if pd.notna(v4):
        rows.append({'date': dt, 'indicator_name': '工业增加值_当月同比_mom_yoy', 'indicator_value': float(v4), 'region': '全国', 'frequency': 'monthly', 'caliber': 'mom_yoy'})

nat_melt = pd.DataFrame(rows)
print(f"\nNational monthly melt: {len(nat_melt)} rows")
for ind in sorted(nat_melt['indicator_name'].unique()):
    sub = nat_melt[nat_melt['indicator_name']==ind]
    cal = sub['caliber'].iloc[0]
    print(f"  {ind}: {len(sub)} obs, {sub['date'].min().date()} ~ {sub['date'].max().date()} [{cal}]")

# ============================================================
print("\n" + "="*60)
print("PHASE 0: PMI Data Extraction")
print("="*60)

pmi = pd.read_csv(DATA_DIR / "pmi_data.csv")
pmi['date'] = pd.to_datetime(pmi['日期'])
pmi = pmi.drop(columns=['日期'])
pmi_cols = [c for c in pmi.columns if c != 'date']
pmi_melt = pmi.melt(id_vars=['date'], value_vars=pmi_cols, var_name='indicator_name', value_name='indicator_value')
pmi_melt['region'] = '全国'
pmi_melt['frequency'] = 'monthly'
pmi_melt = pmi_melt.dropna(subset=['indicator_value'])
print(f"PMI: {len(pmi_melt)} rows, {len(pmi_cols)} indicators")
for ind in sorted(pmi_cols):
    sub = pmi_melt[pmi_melt['indicator_name']==ind]
    print(f"  PMI_{ind}: {len(sub)} obs, {sub['date'].min().date()} ~ {sub['date'].max().date()}")

# Merge PMI into national monthly
nat_melt = pd.concat([nat_melt, pmi_melt], ignore_index=True)
print(f"\nNational monthly + PMI: {len(nat_melt)} rows")

# Save
nat_melt.to_csv(DATA_DIR / "monthly_national_features_real.csv", index=False, encoding='utf-8-sig')
print("Saved national monthly CSV.")

# ============================================================
print("\n" + "="*60)
print("PHASE 0.5: Data Health Check")
print("="*60)

for label, df in [("Sichuan quarterly", sc_q_melt), ("Sichuan monthly", sc_melt), ("National monthly", nat_melt)]:
    print(f"\n--- {label} ---")
    for ind in sorted(df['indicator_name'].unique()):
        sub = df[df['indicator_name']==ind]
        n = len(sub)
        gaps = 0
        if n >= 2:
            sd = sub.sort_values('date')
            # Count gaps > 3 months (quarterly) or > 2 months (monthly)
            freq_threshold = timedelta(days=120) if 'quarterly' in str(sub['frequency'].iloc[0]) else timedelta(days=62)
            for i in range(1, len(sd)):
                if (sd['date'].iloc[i] - sd['date'].iloc[i-1]) > freq_threshold:
                    gaps += 1
        dmin = sub['date'].min().date()
        dmax = sub['date'].max().date()
        passes = "PASS" if n >= 40 else "FAIL (<40)"
        info = " [ORPHAN]" if n < 5 else ""
        print(f"  {ind:<30s} {n:4d} obs  {str(dmin):<12s}~ {str(dmax):<12s}  gaps:{gaps}  {passes}{info}")

print("\nDone!")
