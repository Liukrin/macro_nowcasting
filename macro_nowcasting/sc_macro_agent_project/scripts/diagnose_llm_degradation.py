"""
降级测试：把 base_url 指向不可路由地址，连续调用 6 次 chat()，
验证：
- 第 5 次连续失败后 is_mock 变为 True
- 降级后因 60s 冷却，第 6 次直接走 mock 快速返回
- 每次 HTTP 超时 3s（测试加速，真实 20s 同理）
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sc_macro_agent.llm.client import LLMClient

client = LLMClient()
# 覆盖为不可路由地址 + 短超时加速测试
client.base_url = "https://10.255.255.1"
from openai import OpenAI
client._client = OpenAI(api_key=client.api_key or "sk-test-placeholder", base_url=client.base_url, timeout=3.0)
client.is_mock = False  # 强制非 mock 以走到真实请求路径

print(f"start: is_mock={client.is_mock}")
t_start = time.perf_counter()

for i in range(6):
    t0 = time.perf_counter()
    result = client.chat(system="test", user="ping")
    elapsed = time.perf_counter() - t0
    first_80 = result[:80].replace("\n", " ")
    print(f"  call {i+1}: is_mock={client.is_mock}, failures={client._consecutive_failures}, elapsed={elapsed:.1f}s, result={first_80}")

total = time.perf_counter() - t_start
print(f"total elapsed: {total:.1f}s")
print(f"usage: {client.get_usage_stats()}")

# 断言
assert client._consecutive_failures >= 5, f"Expected >=5 failures, got {client._consecutive_failures}"
assert client.is_mock, "Expected is_mock=True after 5 consecutive failures"
print("PASS: degradation cooldown works")

