"""
诊断脚本：定位 query_indicator 对标准查询返回 found=false 的根因。
全程不调 LLM，不修改任何业务代码。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sc_macro_agent.config import AppConfig
from sc_macro_agent.rag_service import (
    RAGService,
    normalize_query_aliases,
    _normalize_chinese_numbers,
    _INDICATOR_ALIASES,
)

config = AppConfig()
rag = RAGService(config=config, engine=None)

SEP = "=" * 70
BR = "-" * 70


# ================================================================
# 【1】语料概况
# ================================================================
print(SEP)
print("【1】语料概况")
print(SEP)

docs = rag.documents
n_total = len(docs)
card_docs = [d for d in docs if d.get("metadata", {}).get("type") == "indicator_card"]
n_card = len(card_docs)
print(f"documents 总数:               {n_total}")
print(f"type=='indicator_card' 数量:  {n_card}")

if card_docs:
    sample = card_docs[0]
    meta = sample["metadata"]
    print(f"\n任意一张 indicator_card metadata 全部键名:")
    for k in sorted(meta.keys()):
        v = meta[k]
        print(f"  {k:<20} = {v!r:<40}  type={type(v).__name__}")

    # 特意打印关键字段的类型
    print(f"\n关键字段类型确认:")
    for key in ["year", "quarter", "month", "region", "indicator", "value"]:
        val = meta.get(key)
        print(f"  {key:<20}  type={type(val).__name__},  value={val!r}")


# ================================================================
# 【2】GDP 相关卡片
# ================================================================
print(f"\n{SEP}")
print("【2】GDP 相关卡片")
print(SEP)

gdp_cards = [d for d in card_docs
             if "GDP" in str(d["metadata"].get("indicator", ""))]
gdp_names = sorted(set(d["metadata"]["indicator"] for d in gdp_cards))
print(f"indicator 含 'GDP' 的卡片数: {len(gdp_cards)}")
print(f"去重 indicator 名:            {gdp_names}")

# 检查 year==2025（int 和 str 都试）
print(f"\nyear==2025 (int) 的 GDP 卡片数: "
      f"{sum(1 for d in gdp_cards if d['metadata'].get('year') == 2025)}")
print(f"year=='2025' (str) 的 GDP 卡片数: "
      f"{sum(1 for d in gdp_cards if str(d['metadata'].get('year')) == '2025')}")

# 前 5 张 year==2025 卡片
gdp_2025 = [d for d in gdp_cards if d["metadata"].get("year") == 2025][:5]
if gdp_2025:
    print(f"\nyear==2025 前 {len(gdp_2025)} 张 GDP 卡片完整 metadata:")
    for i, d in enumerate(gdp_2025):
        print(f"\n  --- 卡片 {i+1} ---")
        for k, v in sorted(d["metadata"].items()):
            print(f"    {k:<20} = {v!r}")
else:
    print("\n⚠️ 未找到 year==2025 的 GDP 卡片")
    # 看看最近一年有什么
    years = sorted(set(d["metadata"].get("year") for d in gdp_cards))
    print(f"  GDP 卡片存在的年份: {years}")


# ================================================================
# 【3】逐层过滤计数
# ================================================================
print(f"\n{SEP}")
print("【3】逐层过滤计数（query_indicator 逻辑复现）")
print(SEP)

query_indicator = "GDP累计同比增速"
query_year = 2025
query_quarter = 2
query_region = "四川省"

# L1: type filter
pool = [d for d in docs if d.get("metadata", {}).get("type") == "indicator_card"]
n0 = len(pool)
print(f"type=='indicator_card'                              → n0 = {n0}")

# L2a: region filter
pool = [d for d in pool if d["metadata"].get("region", "") == query_region]
n1 = len(pool)
print(f"+ region == '{query_region}'                        → n1 = {n1}")

# L2b: year filter
pool_year_int = [d for d in pool if d["metadata"].get("year") == query_year]
n2_int = len(pool_year_int)
pool_year_str = [d for d in pool if str(d["metadata"].get("year")) == str(query_year)]
n2_str = len(pool_year_str)
print(f"+ year  == {query_year}  (int比较)                   → n2 = {n2_int}")
if n2_int != n2_str:
    print(f"  ⚠️ str 比较结果不同: {n2_str}")

# L2c: quarter filter
pool_q = [d for d in pool_year_int if d["metadata"].get("quarter") == query_quarter]
n3 = len(pool_q)
print(f"+ quarter == {query_quarter}                            → n3 = {n3}")

if n3 == 0:
    print(f"\n  ❌ n3=0！quarter 过滤后为空。")
    # 看看 n2 后有哪些 quarter
    quarters_at_n2 = sorted(set(d["metadata"].get("quarter") for d in pool_year_int))
    print(f"  n2 中存在的 quarter 值: {quarters_at_n2}")
    # 也看看 year 过滤前有哪些 year
    years_at_n1 = sorted(set(d["metadata"].get("year") for d in pool))
    print(f"  n1 中存在的 year 值: {years_at_n1}")
elif n2_int == 0:
    print(f"\n  ❌ n2=0！year 过滤后为空。")
elif n1 == 0:
    print(f"\n  ❌ n1=0！region 过滤后为空。")

pool_final = pool_q  # for later use


# ================================================================
# 【4】别名与模糊匹配
# ================================================================
print(f"\n{SEP}")
print("【4】别名与模糊匹配")
print(SEP)

normalized = normalize_query_aliases(query_indicator)
print(f"normalize_query_aliases('{query_indicator}') = '{normalized}'")

# 检查别名表是否有相关条目
print(f"\n别名表 _INDICATOR_ALIASES 中与 'GDP' 相关的:")
for alias, canonical in _INDICATOR_ALIASES.items():
    if "GDP" in alias or "GDP" in canonical or "经济" in alias:
        print(f"  '{alias}' → '{canonical}'")

# 对每个真实 GDP 标准名做 fuzzy match
print(f"\n对每个 GDP 标准名调用 _indicator_fuzzy_match('{normalized}', target):")
for name in gdp_names:
    score = rag._indicator_fuzzy_match(normalized, name)
    print(f"  score={score:.2f}  target='{name}'")
    if score == 0:
        q_normed = rag._norm(normalized)
        t_normed = rag._norm(name)
        q_set = set(q_normed)
        t_set = set(t_normed)
        extra = q_set - t_set
        print(f"    q_normed='{q_normed}'")
        print(f"    t_normed='{t_normed}'")
        print(f"    q_set - t_set = {extra}  ← issubset 失败原因")


# ================================================================
# 【5】真实调用
# ================================================================
print(f"\n{SEP}")
print("【5】真实调用")
print(SEP)

import json as _json
result = rag.query_indicator(
    indicator=query_indicator, year=query_year,
    quarter=query_quarter, region=query_region,
)
print(f"query_indicator('{query_indicator}', {query_year}, quarter={query_quarter}, region='{query_region}')")
print(_json.dumps(result, ensure_ascii=False, indent=2))


# ================================================================
# 【6】变体对照
# ================================================================
print(f"\n{SEP}")
print("【6】变体对照")
print(SEP)

variants = [
    ("a) 去掉 quarter", query_indicator, query_year, None, query_region),
    ("b) 去掉 region",  query_indicator, query_year, query_quarter, ""),
    ("c) 简写 'GDP'",  "GDP",           query_year, query_quarter, query_region),
]

# d) 真实标准名
real_names = [n for n in gdp_names if n != query_indicator]
d_name = real_names[0] if real_names else gdp_names[0] if gdp_names else query_indicator
variants.append(("d) 真实标准名", d_name, query_year, query_quarter, query_region))

# e) 换一年
variants.append(("e) 2024年", query_indicator, 2024, query_quarter, query_region))

for label, ind, yr, qtr, reg in variants:
    r = rag.query_indicator(indicator=ind, year=yr, quarter=qtr, region=reg)
    print(f"\n{label}: query_indicator('{ind}', {yr}, quarter={qtr}, region='{reg}')")
    print(f"  found={r['found']}, score={r['score']:.4f}, matched='{r.get('matched_indicator', '')}'")
    if r.get("candidates"):
        for c in r["candidates"]:
            print(f"    candidate: {c['name']} score={c['score']:.4f}")
    if not r["found"]:
        print(f"  → FAIL: found={r['found']}")

print(f"\n{SEP}")
print("诊断完成。")
print(SEP)
