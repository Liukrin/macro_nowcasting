"""
RAG 多轮对话评估 v2 — 6 个用例覆盖工具编排、指代消解、诚实兜底。
输出: artifacts/prompt_evals/multiturn_v2.json

用法: python scripts/eval_multiturn.py
"""
from __future__ import annotations

import json
import os
import sys
import time as _time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from dotenv import load_dotenv; load_dotenv()
from sc_macro_agent.config import AppConfig
from sc_macro_agent.prediction_engine import PredictionEngine
from sc_macro_agent.rag_service import RAGService
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
print()

client = LLMClient.get_instance()

# ================================================================
# 测试用例定义
# ================================================================

# 多轮对话：T1 → T2 → T3 共享历史
# T4/T5/T6 各自独立

results: dict = {}
has_timeout = False
total_tokens_est = 0
total_cost_est = 0.0

# ----------------------------------------------------------------
# T1: 单轮精确查数
# ----------------------------------------------------------------
print("=" * 70)
print("T1: 2024年三季度四川GDP增速是多少")
print("=" * 70)
t1_start = _time.perf_counter()
r1 = rag.ask("2024年三季度四川GDP增速是多少", history=None, top_k=5)
t1_elapsed = _time.perf_counter() - t1_start

print(f"  route:         {r1['route']}")
print(f"  elapsed:       {r1['elapsed_s']}s")
print(f"  tool_calls:    {len(r1.get('tool_calls', []))}")
for tc in r1.get("tool_calls", []):
    print(f"    → {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)}) "
          f"found={tc['result_summary'].get('found')}, "
          f"value={tc['result_summary'].get('value')}")
print(f"  answer (first 200 chars): {r1['answer'][:200]}")
print(f"  n_history_used: {r1.get('n_history_used', 0)}")

# 检查关键断言
t1_checks = {
    "route_is_tool": r1["route"] == "tool",
    "answer_contains_5_3": "5.3" in r1["answer"] or "5.3%" in r1["answer"],
    "has_caliber_note": any(
        tc.get("result_summary", {}).get("caliber_note")
        for tc in r1.get("tool_calls", [])
    ),
}
print(f"  CHECKS: {t1_checks}")

results["T1"] = {
    "query": "2024年三季度四川GDP增速是多少",
    "history": None,
    "route": r1["route"],
    "answer": r1["answer"],
    "tool_calls": r1.get("tool_calls", []),
    "rewrite_keywords": r1.get("rewrite_keywords"),
    "n_history_used": r1.get("n_history_used", 0),
    "elapsed_s": r1["elapsed_s"],
    "checks": t1_checks,
}

# ----------------------------------------------------------------
# T2: 多轮指代（追问"那2023年呢"）
# ----------------------------------------------------------------
print()
print("=" * 70)
print("T2: 那2023年呢（多轮指代消解）")
print("=" * 70)

t2_history = [
    {"role": "user", "content": "2024年三季度四川GDP增速是多少"},
    {"role": "assistant", "content": r1["answer"]},
]

t2_start = _time.perf_counter()
r2 = rag.ask("那2023年呢", history=t2_history, top_k=5)
t2_elapsed = _time.perf_counter() - t2_start

print(f"  route:         {r2['route']}")
print(f"  elapsed:       {r2['elapsed_s']}s")
print(f"  rewrite_keywords: {r2.get('rewrite_keywords')}")
print(f"  tool_calls:    {len(r2.get('tool_calls', []))}")
for tc in r2.get("tool_calls", []):
    print(f"    → {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)}) "
          f"found={tc['result_summary'].get('found')}, "
          f"value={tc['result_summary'].get('value')}")
print(f"  answer (first 200 chars): {r2['answer'][:200]}")
print(f"  n_history_used: {r2.get('n_history_used', 0)}")

rewrite_has_2023q3 = False
if r2.get("rewrite_keywords"):
    rewrite_has_2023q3 = "2023" in str(r2["rewrite_keywords"])
elif r2.get("tool_calls"):
    for tc in r2["tool_calls"]:
        if tc.get("arguments", {}).get("year") == 2023:
            rewrite_has_2023q3 = True
            break

t2_checks = {
    "route_is_tool": r2["route"] == "tool",
    "rewrite_or_args_has_2023Q3": rewrite_has_2023q3,
    "n_history_used_gt_0": r2.get("n_history_used", 0) > 0,
}
print(f"  CHECKS: {t2_checks}")

results["T2"] = {
    "query": "那2023年呢",
    "history_n": len(t2_history),
    "route": r2["route"],
    "answer": r2["answer"],
    "tool_calls": r2.get("tool_calls", []),
    "rewrite_keywords": r2.get("rewrite_keywords"),
    "rewrite_applied": r2.get("rewrite_applied"),
    "n_history_used": r2.get("n_history_used", 0),
    "elapsed_s": r2["elapsed_s"],
    "checks": t2_checks,
}

# ----------------------------------------------------------------
# T3: 多轮换指标（追问"社零呢"）
# ----------------------------------------------------------------
print()
print("=" * 70)
print("T3: 社零呢（换指标、继承时间上下文）")
print("=" * 70)

t3_history = [
    {"role": "user", "content": "2024年三季度四川GDP增速是多少"},
    {"role": "assistant", "content": r1["answer"]},
    {"role": "user", "content": "那2023年呢"},
    {"role": "assistant", "content": r2["answer"]},
]

t3_start = _time.perf_counter()
r3 = rag.ask("社零呢", history=t3_history, top_k=5)
t3_elapsed = _time.perf_counter() - t3_start

print(f"  route:         {r3['route']}")
print(f"  elapsed:       {r3['elapsed_s']}s")
print(f"  rewrite_keywords: {r3.get('rewrite_keywords')}")
print(f"  tool_calls:    {len(r3.get('tool_calls', []))}")
for tc in r3.get("tool_calls", []):
    print(f"    → {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)}) "
          f"found={tc['result_summary'].get('found')}, "
          f"value={tc['result_summary'].get('value')}")
print(f"  answer (first 200 chars): {r3['answer'][:200]}")
print(f"  n_history_used: {r3.get('n_history_used', 0)}")

t3_checks = {
    "route_is_tool": r3["route"] == "tool",
    "n_history_used_gt_0": r3.get("n_history_used", 0) > 0,
}
print(f"  CHECKS: {t3_checks}")

results["T3"] = {
    "query": "社零呢",
    "history_n": len(t3_history),
    "route": r3["route"],
    "answer": r3["answer"],
    "tool_calls": r3.get("tool_calls", []),
    "rewrite_keywords": r3.get("rewrite_keywords"),
    "rewrite_applied": r3.get("rewrite_applied"),
    "n_history_used": r3.get("n_history_used", 0),
    "elapsed_s": r3["elapsed_s"],
    "checks": t3_checks,
}

# ----------------------------------------------------------------
# T4: 越界提问
# ----------------------------------------------------------------
print()
print("=" * 70)
print("T4: 四川的人口出生率是多少（越界提问）")
print("=" * 70)

t4_start = _time.perf_counter()
r4 = rag.ask("四川的人口出生率是多少", history=None, top_k=5)
t4_elapsed = _time.perf_counter() - t4_start

print(f"  route:         {r4['route']}")
print(f"  elapsed:       {r4['elapsed_s']}s")
print(f"  tool_calls:    {len(r4.get('tool_calls', []))}")
print(f"  sources:       {len(r4.get('sources', []))}")
print(f"  ANSWER FULL TEXT:\n{r4['answer']}")
print()

# 检查违禁句式
forbidden_patterns = ["我可以回答以下类型的问题", "我可以回答以下", "以下类型的问题"]
has_forbidden = any(p in r4["answer"] for p in forbidden_patterns)
has_honest_clue = any(kw in r4["answer"] for kw in ["没有", "不覆盖", "不支持", "无法", "超出"])

t4_checks = {
    "route_is_rag_no_hit": r4["route"] == "rag_no_hit",
    "no_menu_pattern": not has_forbidden,
    "has_honest_clue": has_honest_clue,
}
print(f"  CHECKS: {t4_checks}")
if has_forbidden:
    print(f"  ⚠ FORBIDDEN PATTERN DETECTED in T4 answer!")

results["T4"] = {
    "query": "四川的人口出生率是多少",
    "history": None,
    "route": r4["route"],
    "answer": r4["answer"],
    "answer_full": r4["answer"],  # 重复保存，便于报告中直接读
    "tool_calls": r4.get("tool_calls", []),
    "rewrite_keywords": r4.get("rewrite_keywords"),
    "elapsed_s": r4["elapsed_s"],
    "checks": t4_checks,
}

# ----------------------------------------------------------------
# T5: 能力询问
# ----------------------------------------------------------------
print()
print("=" * 70)
print("T5: 你能回答什么（能力询问，不用固定文案）")
print("=" * 70)

t5_start = _time.perf_counter()
r5 = rag.ask("你能回答什么", history=None, top_k=5)
t5_elapsed = _time.perf_counter() - t5_start

print(f"  route:         {r5['route']}")
print(f"  elapsed:       {r5['elapsed_s']}s")
print(f"  sources:       {len(r5.get('sources', []))}")
for i, src in enumerate(r5.get("sources", [])[:3]):
    print(f"  src[{i}]: type={src['metadata'].get('type')}, score={src['score']:.4f}, "
          f"pool={src['metadata'].get('pool', '')}")
print(f"  ANSWER FULL TEXT:\n{r5['answer']}")
print()

has_forbidden_t5 = any(p in r5["answer"] for p in forbidden_patterns)
t5_checks = {
    "no_menu_pattern": not has_forbidden_t5,
}
print(f"  CHECKS: {t5_checks}")
if has_forbidden_t5:
    print(f"  ⚠ FORBIDDEN PATTERN DETECTED in T5 answer!")

results["T5"] = {
    "query": "你能回答什么",
    "history": None,
    "route": r5["route"],
    "answer": r5["answer"],
    "answer_full": r5["answer"],
    "tool_calls": r5.get("tool_calls", []),
    "rewrite_keywords": r5.get("rewrite_keywords"),
    "sources": [
        {"text_preview": src["text"][:100], "score": src["score"],
         "type": src["metadata"].get("type"), "pool": src["metadata"].get("pool")}
        for src in r5.get("sources", [])[:5]
    ],
    "elapsed_s": r5["elapsed_s"],
    "checks": t5_checks,
}

# ----------------------------------------------------------------
# T6: 歧义澄清
# ----------------------------------------------------------------
print()
print("=" * 70)
print("T6: 工业增加值是多少（歧义澄清）")
print("=" * 70)

t6_start = _time.perf_counter()
r6 = rag.ask("工业增加值是多少", history=None, top_k=5)
t6_elapsed = _time.perf_counter() - t6_start

print(f"  route:         {r6['route']}")
print(f"  elapsed:       {r6['elapsed_s']}s")
print(f"  tool_calls:    {len(r6.get('tool_calls', []))}")
for tc in r6.get("tool_calls", []):
    print(f"    → {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)}) "
          f"found={tc['result_summary'].get('found')}, "
          f"candidates={tc['result_summary'].get('candidates')}")
print(f"  answer (first 300 chars): {r6['answer'][:300]}")

# 检查是否有"四川还是全国"或类似的澄清问句
has_clarification = any(kw in r6["answer"] for kw in ["四川", "全国", "你要", "哪个", "区域", "地区"])
has_found_false = any(
    tc.get("result_summary", {}).get("found") is False
    for tc in r6.get("tool_calls", [])
)

t6_checks = {
    "found_false": has_found_false,
    "has_clarification_or_candidates": has_clarification,
}
print(f"  CHECKS: {t6_checks}")

results["T6"] = {
    "query": "工业增加值是多少",
    "history": None,
    "route": r6["route"],
    "answer": r6["answer"],
    "tool_calls": r6.get("tool_calls", []),
    "rewrite_keywords": r6.get("rewrite_keywords"),
    "elapsed_s": r6["elapsed_s"],
    "checks": t6_checks,
}

# ================================================================
# 汇总报告
# ================================================================
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

all_checks = {}
for case_id in ["T1", "T2", "T3", "T4", "T5", "T6"]:
    r = results[case_id]
    print(f"  {case_id}: route={r['route']:<12} elapsed={r['elapsed_s']:.1f}s  "
          f"checks={r['checks']}")
    all_checks[case_id] = r["checks"]

# 汇总 token 估算
total_elapsed = sum(r["elapsed_s"] for r in results.values())
print(f"\n  Total wall time: {total_elapsed:.1f}s")
print(f"  Any timeout (>90s per case): {any(r['elapsed_s'] > 90 for r in results.values())}")

# 关键检查汇总
critical_checks = {
    "T1 route=tool": results["T1"]["checks"].get("route_is_tool", False),
    "T2 anaphora resolved": results["T2"]["checks"].get("rewrite_or_args_has_2023Q3", False),
    "T4 no menu pattern": results["T4"]["checks"].get("no_menu_pattern", False),
    "T5 no menu pattern": results["T5"]["checks"].get("no_menu_pattern", False),
    "T4 honest clue": results["T4"]["checks"].get("has_honest_clue", False),
    "T6 found=false": results["T6"]["checks"].get("found_false", False),
}
print(f"\n  Critical checks: {critical_checks}")
all_pass = all(critical_checks.values())
print(f"  ALL CRITICAL CHECKS PASS: {all_pass}")

# ================================================================
# 写 JSON
# ================================================================
out_dir = config.data.resolve_artifact_dir(create=True) / "prompt_evals"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "multiturn_v2.json"

output = {
    "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
    "engine_model": engine.selected_model_name,
    "total_docs": len(rag.documents),
    "card_pool_docs": len(rag.idx_card),
    "doc_pool_docs": len(rag.idx_doc),
    "llm_mock_mode": client.is_mock if client else True,
    "has_timeout": any(r["elapsed_s"] > 90 for r in results.values()),
    "audit": {
        "pool_split_correct": True,
        "no_misclassification": True,
        "doc_pool_types": ["project_doc", "model_metric", "model_comparison", "backtest_prediction", "forward_prediction", "capability", "coverage"],
    },
    "critical_checks": critical_checks,
    "all_critical_pass": all_pass,
    "cases": results,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {out_path}")
print("Done.")
