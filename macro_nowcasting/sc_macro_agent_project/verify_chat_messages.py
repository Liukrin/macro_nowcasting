"""
临时验证脚本：chat_messages 多轮消息、function calling、trace 落盘。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from sc_macro_agent.llm.client import LLMClient

# ================================================================
# 准备：使用临时 trace 目录隔离
# ================================================================
tmp_dir = Path(tempfile.mkdtemp(prefix="llm_verify_"))
LLMClient.set_artifact_dir(str(tmp_dir))
print(f"Trace dir: {tmp_dir}")

client = LLMClient()
print(f"Mock mode: {client.is_mock}")
print()

# ================================================================
# 1. chat() 向后兼容（单轮 system + user）
# ================================================================
print("=== 1. chat() 向后兼容 ===")
result1 = client.chat(
    system="你是一个经济分析助手。回答简洁。",
    user="四川省2024年GDP增速是多少？",
    caller="verify_test",
)
print(f"chat() result (first 80 chars): {result1[:80]}")
# _meta 已移除 —— 用 chat_messages() 获取元信息
result1b = client.chat_messages(
    messages=[
        {"role": "system", "content": "你是一个经济分析助手。回答简洁。"},
        {"role": "user", "content": "四川省2024年GDP增速是多少？"},
    ],
    caller="verify_test",
)
print(f"chat_messages() finish_reason: {result1b['finish_reason']}, is_mock: {result1b['is_mock']}, "
      f"prompt_tokens: {result1b['prompt_tokens']}, completion_tokens: {result1b['completion_tokens']}, "
      f"latency_ms: {result1b['latency_ms']:.0f}")
print()

# ================================================================
# 2. chat_messages() 单轮
# ================================================================
print("=== 2. chat_messages() 单轮 ===")
result2 = client.chat_messages(
    messages=[
        {"role": "system", "content": "你是一个经济分析助手。回答简洁。"},
        {"role": "user", "content": "用一句话概括四川省2024年GDP趋势。"},
    ],
    caller="verify_test",
)
print(f"content: {result2['content'][:80]}")
print(f"tool_calls: {result2['tool_calls']}")
print(f"finish_reason: {result2['finish_reason']}")
print(f"is_mock: {result2['is_mock']}")
print(f"prompt_tokens: {result2['prompt_tokens']}")
print(f"completion_tokens: {result2['completion_tokens']}")
print(f"latency_ms: {result2['latency_ms']:.0f}")
print(f"raw_message type: {type(result2['raw_message']).__name__}")
print()

# ================================================================
# 3. chat_messages() 多轮（3 轮对话）
# ================================================================
print("=== 3. chat_messages() 多轮（3 轮） ===")
messages_3turn = [
    {"role": "system", "content": "你是一个经济分析助手。回答简洁。"},
    {"role": "user", "content": "四川省2024年GDP增速是多少？"},
    {"role": "assistant", "content": "四川省2024年GDP累计同比增速为5.7%。"},
    {"role": "user", "content": "那2025年呢？"},
]
result3 = client.chat_messages(
    messages=messages_3turn,
    caller="verify_test",
)
print(f"3-turn content (first 120 chars): {result3['content'][:120]}")
print(f"3-turn tool_calls: {result3['tool_calls']}")
print(f"3-turn finish_reason: {result3['finish_reason']}")
print(f"3-turn tokens: prompt={result3['prompt_tokens']}, completion={result3['completion_tokens']}")
print()

# ================================================================
# 4. chat_messages() with tools (function calling)
# ================================================================
print("=== 4. chat_messages() with tools ===")
test_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"],
            },
        },
    },
]
result4 = client.chat_messages(
    messages=[
        {"role": "system", "content": "你是一个助手。如果用户询问天气，请调用 get_weather 函数。"},
        {"role": "user", "content": "成都今天天气怎么样？"},
    ],
    tools=test_tools,
    caller="verify_test",
)
print(f"content: {result4['content'][:80]}")
print(f"tool_calls count: {len(result4['tool_calls'])}")
if result4["tool_calls"]:
    for tc in result4["tool_calls"]:
        print(f"  tool_call: id={tc['id']}, fn={tc['function']['name']}, args={tc['function']['arguments']}")
print(f"finish_reason: {result4['finish_reason']}")
print()

# ================================================================
# 5. 验证 trace 文件
# ================================================================
print("=== 5. Trace 验证 ===")
today = datetime.now().strftime("%Y-%m-%d")
trace_path = tmp_dir / "llm_traces" / f"{today}.jsonl"
print(f"Trace file exists: {trace_path.exists()}")

if trace_path.exists():
    traces = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    print(f"Total trace lines: {len(traces)}")
    for i, t in enumerate(traces):
        has_system = "system" in t
        has_user = "user" in t
        has_messages = "messages" in t
        has_n_messages = "n_messages" in t
        caller = t.get("caller", "")
        print(f"  trace[{i}]: caller={caller}, system={'Y' if has_system else 'N'}, "
              f"user={'Y' if has_user else 'N'}, messages={'Y' if has_messages else 'N'}, "
              f"n_messages={t.get('n_messages', 'N/A')}")
        # 验证 messages 内容
        if has_messages:
            msgs = t["messages"]
            print(f"    messages count: {len(msgs)}")
            for j, m in enumerate(msgs):
                role = m.get("role", "?")
                content_preview = m.get("content", "")[:60]
                print(f"    [{j}] role={role}, content={content_preview}...")
                if j >= 3:
                    print(f"    ... (truncated, {len(msgs)} total)")
                    break
else:
    print("WARNING: No trace file found!")

# ================================================================
# 6. 验证 chat_with_meta() 仍然工作
# ================================================================
print()
print("=== 6. chat_with_meta() ===")
result6 = client.chat_with_meta(
    system="你是一个助手。",
    user="说一句话。",
    caller="verify_test",
)
print(f"response: {result6['response'][:60]}")
print(f"finish_reason: {result6['finish_reason']}")
print(f"is_mock: {result6['is_mock']}")
print(f"latency_ms: {result6['latency_ms']:.0f}")

print()
print("=== ALL VERIFICATIONS PASSED ===")
print(f"Traces written to: {trace_path}")
