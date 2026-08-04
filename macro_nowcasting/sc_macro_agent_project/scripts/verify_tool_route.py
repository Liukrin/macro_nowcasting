"""
端到端验证 query_indicator 修复效果。仅 3 个问题，一次 LLM 批次。
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv; load_dotenv()
from sc_macro_agent.config import AppConfig
from sc_macro_agent.prediction_engine import PredictionEngine
from sc_macro_agent.rag_service import RAGService
from sc_macro_agent.llm.client import LLMClient

config = AppConfig()
config.data.dataset_mode = "real"
LLMClient.set_artifact_dir(config.data.resolve_artifact_dir(create=True))

t0 = _time.perf_counter()
print("Initializing engine ...")
engine = PredictionEngine(config=config)
engine.initialize()
t1 = _time.perf_counter()
print(f"Engine ready ({t1 - t0:.1f}s)")

rag = RAGService(config, engine)
t2 = _time.perf_counter()
print(f"RAGService ready: {len(rag.documents)} docs ({t2 - t1:.1f}s)")
print()

QUESTIONS = [
    "2025年二季度四川GDP增速是多少",
    "这个系统能回答哪些问题",
    "MIDAS 模型是什么",
]

totals = {"n_llm_calls": 0, "total_tokens": 0, "est_cost_cny": 0.0}

for i, q in enumerate(QUESTIONS, 1):
    print("=" * 70)
    print(f"Q{i}: {q}")
    print("=" * 70)

    result = rag.ask(q)

    route = result["route"]
    usage = result["usage"]
    td = result.get("tool_decision", {})

    print(f"  route:           {route}")
    print(f"  route_reason:    {result.get('route_reason', 'N/A')}")
    print(f"  tool_loop_skipped: {result.get('tool_loop_skipped', 'N/A')}")
    print(f"  usage:")
    for k in ["prompt_tokens", "completion_tokens", "cache_hit_tokens",
              "total_tokens", "est_cost_cny", "n_llm_calls"]:
        print(f"    {k:<20} {usage.get(k, 'N/A')}")

    print(f"  tool_decision:")
    print(f"    round0_finish_reason:    {td.get('round0_finish_reason', 'N/A')}")
    print(f"    round0_completion_tokens: {td.get('round0_completion_tokens', 'N/A')}")
    print(f"    round0_retried:          {td.get('round0_retried', 'N/A')}")

    print(f"  tool_calls_log:")
    for tc in result.get("tool_calls", []):
        print(f"    [{tc['name']}] arguments={json.dumps(tc['arguments'], ensure_ascii=False)}")
    if not result.get("tool_calls"):
        print("    (none)")

    answer = result.get("answer", "")
    print(f"  answer[120]:     {answer[:120]}")

    # Q3 额外打印 sources
    if i == 3:
        print(f"  sources ({len(result.get('sources', []))} items):")
        for j, src in enumerate(result.get("sources", [])):
            meta = src.get("metadata", {})
            text_preview = src.get("text", "")[:60]
            print(f"    [{j}] type={meta.get('type', '?')}  {text_preview}")

    totals["n_llm_calls"] += usage.get("n_llm_calls", 0)
    totals["total_tokens"] += usage.get("total_tokens", 0)
    totals["est_cost_cny"] += usage.get("est_cost_cny", 0.0)
    print()

print("=" * 70)
print("汇总")
print("=" * 70)
print(f"  n_llm_calls 合计:  {totals['n_llm_calls']}")
print(f"  total_tokens 合计:  {totals['total_tokens']}")
print(f"  est_cost_cny 合计:  {totals['est_cost_cny']:.6f}")

# 基准对比
print()
print("--- 基准对比 ---")
baseline_initial = {  # 最初（修复前）
    "Q1": {"route": "rag", "n_llm_calls": 5, "total": 2849},
    "Q2": {"route": "tool", "n_llm_calls": 3, "total": 6870},
    "Q3": {"route": "tool", "n_llm_calls": 2, "total": 3446},
}
baseline_prev = {    # 上一轮（工具描述+模糊匹配修复后）
    "Q1": {"route": "tool", "n_llm_calls": 2, "total": 2905},
    "Q2": {"route": "tool", "n_llm_calls": 2, "total": 4588},
    "Q3": {"route": "tool", "n_llm_calls": 3, "total": 5810},
}
print("  最初（修复前）:")
for k, v in baseline_initial.items():
    print(f"    {k}: route={v['route']}, n_calls={v['n_llm_calls']}, total_tok={v['total']}")
print("  上一轮（工具描述+模糊匹配修复后）:")
for k, v in baseline_prev.items():
    print(f"    {k}: route={v['route']}, n_calls={v['n_llm_calls']}, total_tok={v['total']}")
