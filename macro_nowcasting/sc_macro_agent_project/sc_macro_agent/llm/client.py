"""
DeepSeek LLM 客户端（兼容 OpenAI SDK）。
无 API Key 时自动降级为 mock 模式。
"""
from __future__ import annotations

import os
import time
from typing import Optional

from ..logging_utils import get_logger


class LLMClient:
    """DeepSeek Chat 客户端。"""

    def __init__(self) -> None:
        self.logger = get_logger("sc_macro_agent.llm")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-chat"
        self.is_mock = self.api_key is None

        self._total_calls = 0
        self._total_tokens = 0
        self._total_time = 0.0
        self._client = None

        if not self.is_mock:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=60.0)
                self.logger.info("LLM client initialized (DeepSeek)")
            except ImportError:
                self.logger.warning("openai not installed, falling back to mock")
                self.is_mock = True
        else:
            self.logger.info("LLM client in MOCK mode (no DEEPSEEK_API_KEY)")

    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1500) -> str:
        """发送对话请求。失败重试 2 次（指数退避）。"""
        self._total_calls += 1

        if self.is_mock or self._client is None:
            return self._mock_response(system, user)

        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                elapsed = time.perf_counter() - t0
                self._total_time += elapsed

                content = resp.choices[0].message.content or ""
                usage = resp.usage
                if usage:
                    self._total_tokens += usage.total_tokens
                return content

            except Exception as exc:
                self.logger.warning("LLM call attempt %d failed: %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    self.is_mock = True
                    return self._mock_response(system, user)

        return self._mock_response(system, user)

    def _mock_response(self, _system: str, user: str) -> str:
        """Mock 响应：回显问题关键信息。"""
        preview = user[:200].replace("\n", " ")
        return f"[MOCK LLM] 收到提问: {preview}...（无 API Key 或调用失败，占位响应）"

    def get_usage_stats(self) -> dict:
        avg_time = self._total_time / max(self._total_calls, 1)
        # deepseek-chat pricing (2025-07, subject to change per official pricing page)
        PRICE_PROMPT_PER_1K = 0.001   # CNY per 1K prompt tokens
        PRICE_COMPLETION_PER_1K = 0.002  # CNY per 1K completion tokens
        est_cost = (self._total_tokens * PRICE_COMPLETION_PER_1K / 1000)  # rough estimate
        return {
            "is_mock": self.is_mock,
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "avg_latency_s": round(avg_time, 2),
            "est_cost_cny": round(est_cost, 6),
            "pricing_note": "deepseek-chat: 1RMB/1M input, 2RMB/1M output (2025-07 official)",
        }

    @classmethod
    def get_instance(cls) -> "LLMClient":
        """模块级单例。"""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance
