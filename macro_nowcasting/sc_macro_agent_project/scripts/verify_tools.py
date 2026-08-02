"""
阶段 3 验收脚本：验证模糊匹配修正、全国数据、工具函数。

A. 不调 LLM，直连测试 9 次
B. 调真实 LLM 走完整一轮 tool calling

用法: python scripts/verify_tools.py
"""
from __future__ import annotations

import json
import os
import sys
import time as _time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv()

from sc_macro_agent.config import AppConfig
from sc_macro_agent.rag_service import RAGService, get_indicator_aliases
from sc_macro_agent.tools import TOOL_SCHEMAS, dispatch
from sc_macro_agent.llm.client import LLMClient, sanitize_assistant_message
from sc_macro_agent.prediction_engine import PredictionEngine

# ================================================================
# 初始化
# ================================================================
print("=" * 70)
print("初始化 ...")
config = AppConfig()
config.data.dataset_mode = "real"
LLMClient.set_artifact_dir(config.data.resolve_artifact_dir(create=True))

t0 = _time.perf_counter()
engine = None
try:
    engine = PredictionEngine(config=config)
    engine.run_agent(goal="audit_build_train_backtest_report", save_artifacts=True)
    print(f"Engine trained: {bool(engine.leaderboard)}, model={engine.selected_model_name}")
except Exception as exc:
    print(f"Engine init skipped: {exc}")

t1 = _time.perf_counter()
rag = RAGService(config, engine)
t2 = _time.perf_counter()
print(f"RAGService built: {len(rag.documents)} docs, {t2-t1:.1f}s (TF-IDF: see log)")
print(f"Aliases: {get_indicator_aliases()}")
print(f"Total init time: {t2-t0:.1f}s")

# ================================================================
# A. 直连测试（9 次）
# ================================================================
print("\n" + "=" * 70)
print("A. 直连测试（不调 LLM）")
print("=" * 70)


def _p(label, result):
    print(f"\n--- {label} ---")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    # 额外打印 candidates 分数便于肉眼检查
    if result.get("candidates"):
        print("  Candidates scores:", [(c["name"], c.get("score")) for c in result["candidates"]])


# A1: GDP增速 + 2024Q3 → found=TRUE（alias: GDP增速→GDP_同比增速）
result_a1 = dispatch(rag, "query_indicator", {
    "indicator": "GDP增速", "year": 2024, "quarter": 3, "region": "四川省",
})
_p("A1: 'GDP增速' 2024Q3 [alias → GDP_同比增速, expect found=TRUE]", result_a1)

# A2: GDP + 2024Q3 → found=false（歧义：GDP_同比增速 vs GDP_累计值）
result_a2 = dispatch(rag, "query_indicator", {
    "indicator": "GDP", "year": 2024, "quarter": 3, "region": "四川省",
})
_p("A2: 'GDP' 2024Q3 [歧义拦截, expect found=false]", result_a2)

# A3: 社零 + 2025-06 → found=TRUE（alias: 社零→社会消费品零售总额）
result_a3 = dispatch(rag, "query_indicator", {
    "indicator": "社零", "year": 2025, "month": 6, "region": "四川省",
})
_p("A3: '社零' 2025-06 [alias, expect found=TRUE]", result_a3)

# A4: 人口出生率 → found=false（语料不存在，score=0）
result_a4 = dispatch(rag, "query_indicator", {
    "indicator": "人口出生率", "year": 2024, "region": "四川省",
})
_p("A4: '人口出生率' 2024 [不存在, expect found=false]", result_a4)

# A5: PMI 2025-06 region=全国 → found=TRUE（全国数据验证）
result_a5 = dispatch(rag, "query_indicator", {
    "indicator": "PMI", "year": 2025, "month": 6, "region": "全国",
})
_p("A5: 'PMI' 2025-06 region=全国 [全国数据, expect found=TRUE]", result_a5)

# A6a: "工业增加值" 2025-06 不限区域 → found=false（四川 vs 全国歧义）
result_a6a = dispatch(rag, "query_indicator", {
    "indicator": "工业增加值", "year": 2025, "month": 6, "region": "",
})
_p("A6a: '工业增加值' 2025-06 region='' [跨区域歧义, expect found=false]", result_a6a)

# A6b: "工业增加值" 2025-06 region=四川省 → found=TRUE
result_a6b = dispatch(rag, "query_indicator", {
    "indicator": "工业增加值", "year": 2025, "month": 6, "region": "四川省",
})
_p("A6b: '工业增加值' 2025-06 region=四川省 [单区域, expect found=TRUE]", result_a6b)

# A7: list_indicators
result_a7 = dispatch(rag, "list_indicators", {})
print(f"\n--- A7: list_indicators ---")
indicators = result_a7.get("indicators", [])
aliases = result_a7.get("aliases", {})
print(f"Total unique indicators: {len(indicators)}")
print(f"Aliases: {aliases}")
for ind in indicators:
    print(f"  [{ind['region']}] {ind['name']:<50} {ind['freq']:<10} {ind['start']} ~ {ind['end']}")

# A8: get_model_leaderboard
result_a8 = dispatch(rag, "get_model_leaderboard", {})
_p("A8: get_model_leaderboard", result_a8)
if result_a8.get("available") and result_a8.get("leaderboard"):
    print("\n  Top 3:")
    for m in result_a8["leaderboard"][:3]:
        sel = " ← SELECTED" if m.get("is_selected") else ""
        base = " (baseline)" if m.get("is_baseline") else ""
        print(f"    #{m['rank']} {m['name']:<20} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f}{sel}{base}")

# A9: get_prediction
result_a9 = dispatch(rag, "get_prediction", {})
_p("A9: get_prediction", result_a9)

# ================================================================
# B. LLM 工具循环
# ================================================================
print("\n" + "=" * 70)
print("B. LLM 工具循环（真实 API）")
print("=" * 70)

client = LLMClient()
if client.is_mock:
    print("SKIP: LLM in mock mode (no DEEPSEEK_API_KEY)")
else:
    system_prompt = (
        "你是一个经济数据分析助手。当用户询问经济指标数值时，"
        "请调用 query_indicator 工具查询。"
        "如果返回 found=true 且 candidates 为空，直接给出数值。"
        "如果返回 found=false 且有 candidates，列出候选让用户选择，不要猜测。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "2024年三季度四川GDP增速是多少？"},
    ]

    print("\n[Round 1] 发送消息（附带 tools）...")
    resp1 = client.chat_messages(
        messages=messages, tools=TOOL_SCHEMAS,
        temperature=0.1, max_tokens=800, caller="verify_tools",
    )
    print(f"  finish_reason: {resp1['finish_reason']}")
    print(f"  content (first 100): {resp1['content'][:100]}")
    print(f"  tool_calls count: {len(resp1['tool_calls'])}")
    if resp1["tool_calls"]:
        for tc in resp1["tool_calls"]:
            print(f"  → tool_call: {tc['function']['name']}({tc['function']['arguments']})")

    if resp1["tool_calls"]:
        # sanitize + append assistant message
        clean_msg = resp1["raw_message"] if resp1["raw_message"] else {
            "role": "assistant", "content": resp1["content"], "tool_calls": resp1["tool_calls"],
        }
        clean_msg = sanitize_assistant_message(clean_msg)
        messages.append(clean_msg)

        # 执行每个 tool call
        for tc in resp1["tool_calls"]:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])
            tool_result = dispatch(rag, fn_name, fn_args)
            print(f"\n  dispatch {fn_name}(...) → found={tool_result.get('found')}, "
                  f"score={tool_result.get('score'):.4f}, value={tool_result.get('value')}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

        # Round 2: 获取最终答案
        print(f"\n[Round 2] 回传工具结果（共 {len(messages)} 条消息）...")
        resp2 = client.chat_messages(
            messages=messages, temperature=0.3, max_tokens=1200, caller="verify_tools",
        )
        print(f"  finish_reason: {resp2['finish_reason']}")
        print(f"  最终答案:\n{resp2['content']}")
    else:
        print(f"\n  Model did not call any tool. Answer:\n{resp1['content']}")

print("\n" + "=" * 70)
print("VERIFY_TOOLS COMPLETE")
print("=" * 70)
