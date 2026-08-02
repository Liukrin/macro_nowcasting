"""
RAG 检索基线评估 v5 — 分池检索 + 别名接入查询解析。
输出: artifacts/prompt_evals/retrieval_baseline_v5.json
"""
from __future__ import annotations

import json, os, sys, time as _time
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from dotenv import load_dotenv; load_dotenv()
from sc_macro_agent.config import AppConfig
from sc_macro_agent.prediction_engine import PredictionEngine
from sc_macro_agent.rag_service import RAGService, _extract_query_filters, normalize_query_aliases
from sc_macro_agent.llm.client import LLMClient

config = AppConfig(); config.data.dataset_mode = "real"
LLMClient.set_artifact_dir(config.data.resolve_artifact_dir(create=True))

t0 = _time.perf_counter()
print("Initializing engine ...")
engine = PredictionEngine(config=config)
engine.run_agent(goal="audit_build_train_backtest_report", save_artifacts=True)
t1 = _time.perf_counter()
print(f"Engine ready: {engine.selected_model_name}, {len(engine.leaderboard)} models ({t1-t0:.1f}s)")

rag = RAGService(config, engine)
t2 = _time.perf_counter()
print(f"RAGService: {len(rag.documents)} docs, card={len(rag.idx_card)}, doc={len(rag.idx_doc)} ({t2-t1:.1f}s)")
print(f"  vec_card features: {rag.mat_card.shape[1] if rag.mat_card is not None else 0}")
print(f"  vec_doc features: {rag.mat_doc.shape[1] if rag.mat_doc is not None else 0}")

QUERIES = [
    "2024年三季度四川GDP增速",
    "2025年6月四川经济怎么样",
    "四川的生产情况",                     # KNOWN_FAIL: TF-IDF 无法理解语义等价
    "全国PMI最近怎么样",
    "哪个模型RMSE最低",
    "数据有哪些已知局限",
    "系统最终选用哪个模型",
    "人口出生率",
    "PMI最近怎么样",
    "四川和全国比工业增速差多少",
]

KNOWN_FAILS = {
    "四川的生产情况": "四川省仅5个指标，语料文本不含'生产'一词；TF-IDF为纯词汇匹配，无法从'生产'跨越到'规模以上工业增加值'。语义检索（BGE）可解，但本地嵌入权重不完整，已在known_limitations记录。",
}

results = {}
for i, q in enumerate(QUERIES, 1):
    filters = _extract_query_filters(q, rag._entity_set)
    alias_applied = normalize_query_aliases(q)
    tq0 = _time.perf_counter()
    sources = rag.search(q, top_k=5, min_score=0.15)
    tq1 = _time.perf_counter()

    top5 = []
    for j, (text, score, meta) in enumerate(sources):
        top5.append({
            "rank": j + 1,
            "text_preview": text[:80].replace("\n", " "),
            "score": round(score, 4),
            "region": meta.get("region", ""),
            "indicator": meta.get("indicator", ""),
            "type": meta.get("type", ""),
            "pool": meta.get("pool", ""),
        })

    n_sichuan = sum(1 for r in top5 if r["region"] == "四川省")
    n_national = sum(1 for r in top5 if r["region"] == "全国")
    status = "KNOWN_FAIL" if q in KNOWN_FAILS else "PASS"

    results[q] = {
        "query": q, "query_id": i, "status": status,
        "reason": KNOWN_FAILS.get(q, ""),
        "alias_applied": alias_applied != q,
        "filters": {
            "year": filters["year"], "quarter": filters["quarter"],
            "month": filters["month"], "region": filters["region"],
            "indicators": filters["indicators"],
            "has_time_constraint": filters["has_time_constraint"],
        },
        "n_results": len(top5), "top1_score": top5[0]["score"] if top5 else 0.0,
        "n_sichuan_in_top5": n_sichuan, "n_national_in_top5": n_national,
        "elapsed_ms": round((tq1 - tq0) * 1000, 1),
        "top5": top5,
    }

    print(f"[{i}/10] [{status}] {q}")
    print(f"  filters: y={filters['year']} q={filters['quarter']} m={filters['month']} "
          f"r={filters['region']} ind={filters['indicators']}")
    if top5:
        pools = set(r["pool"] for r in top5)
        regions = set(r["region"] for r in top5 if r["region"])
        print(f"  top1: pool={top5[0]['pool']} score={top5[0]['score']:.4f} type={top5[0]['type']} "
              f"region={top5[0]['region']} | {top5[0]['text_preview'][:60]}...")
        print(f"  n_results={len(top5)} pools={pools} regions={regions} sichuan={n_sichuan} national={n_national}")
    else:
        print(f"  NO RESULTS")
    print()

out_dir = config.data.resolve_artifact_dir(create=True) / "prompt_evals"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "retrieval_baseline_v5.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_docs": len(rag.documents),
        "pool_card_docs": len(rag.idx_card), "pool_doc_docs": len(rag.idx_doc),
        "vec_card_features": rag.mat_card.shape[1] if rag.mat_card is not None else 0,
        "vec_doc_features": rag.mat_doc.shape[1] if rag.mat_doc is not None else 0,
        "min_score_card": 0.15, "min_score_doc": 0.08, "local_region_boost": 0.05,
        "queries": results,
    }, f, ensure_ascii=False, indent=2)

print(f"Baseline saved to: {out_path}")
print("Done.")
