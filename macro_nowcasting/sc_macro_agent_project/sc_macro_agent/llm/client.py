"""
DeepSeek LLM 客户端（兼容 OpenAI SDK）。
无 API Key 时自动降级为 mock 模式。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger

# deepseek-v4-flash 单价（2026-07 核对，USD 转 CNY 约 7.2）
# 正式上线前到 https://api-docs.deepseek.com/quick_start/pricing 核对
# 注：cache hit 输入价为 $0.0028/1M，本估算按 cache miss 计，属保守上界
_PRICE_PROMPT_PER_1K = 0.001       # CNY / 1K input tokens
_PRICE_COMPLETION_PER_1K = 0.002   # CNY / 1K output tokens

# 单次 chat 总耗时上限（含重试）：超过则直接降级 mock，避免云端长时间挂起
_MAX_CHAT_TOTAL_SECONDS = 60.0


class LLMClient:
    """DeepSeek Chat 客户端（单例）。"""

    # 由入口 set_artifact_dir() 推入，trace 落盘与此对齐
    _override_traces_dir: Optional[Path] = None

    @classmethod
    def set_artifact_dir(cls, path: str | Path) -> None:
        """覆盖 trace 落盘目录。run.py / main.py / app.py 在 engine 构造后调用。"""
        cls._override_traces_dir = Path(path).expanduser().resolve()
        instance = cls.__dict__.get("_instance")
        if instance is not None:
            instance._traces_dir = None

    def __init__(self) -> None:
        self.logger = get_logger("sc_macro_agent.llm")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-v4-flash"
        self.is_mock = self.api_key is None

        self._total_calls = 0
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._consecutive_failures = 0
        self._last_failure_ts = 0.0
        self._total_time = 0.0
        self._client = None
        self._meta: dict = {}  # 最近一次 chat() 的元信息，供 chat_with_meta() 读取

        # trace 落盘目录，惰性计算一次后缓存
        self._traces_dir: Optional[Path] = None

        if not self.is_mock:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=20.0)
                self.logger.info("LLM client initialized (DeepSeek)")
            except ImportError:
                self.logger.warning("openai not installed, falling back to mock")
                self.is_mock = True
        else:
            self.logger.info("LLM client in MOCK mode (no DEEPSEEK_API_KEY)")

    # ------------------------------------------------------------------
    # trace 目录（惰性计算）
    # ------------------------------------------------------------------
    @property
    def traces_dir(self) -> Path:
        if self._traces_dir is None:
            if self._override_traces_dir is not None:
                self._traces_dir = self._override_traces_dir / "llm_traces"
            else:
                from ..config import AppConfig
                # 无显式覆盖时回退到 AppConfig 默认（Streamlit 路径若未调用
                # set_artifact_dir 则走这里，与 engine 路径一致）
                self._traces_dir = AppConfig().data.resolve_artifact_dir(create=True) / "llm_traces"
        return self._traces_dir

    # ------------------------------------------------------------------
    # 内部：单次请求，重试 2 次 + 固定 1s 退避 + 60s 总耗时上限，返回结构化 dict
    # ------------------------------------------------------------------
    def _do_chat(self, system: str, user: str, temperature: float, max_tokens: int) -> dict:
        """发送请求，内部重试 2 次（固定 1s 退避），单次调用总耗时上限 60s。
        返回 {"content": str, "prompt_tokens": int, "completion_tokens": int,
               "cache_hit_tokens": int, "latency_ms": float, "finish_reason": str}
        """
        deadline = time.perf_counter() + _MAX_CHAT_TOTAL_SECONDS
        for attempt in range(2):
            if time.perf_counter() >= deadline:
                self.logger.error("LLM total time exceeded %.0fs budget; degrading to mock",
                                  _MAX_CHAT_TOTAL_SECONDS)
                raise RuntimeError(f"LLM call exceeded {_MAX_CHAT_TOTAL_SECONDS:.0f}s total budget")
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
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._total_time += elapsed_ms / 1000

                content = resp.choices[0].message.content or ""
                usage = resp.usage
                prompt_tokens = 0
                completion_tokens = 0
                cache_hit_tokens = 0
                if usage:
                    self._total_tokens += usage.total_tokens
                    if hasattr(usage, 'prompt_tokens') and usage.prompt_tokens:
                        prompt_tokens = usage.prompt_tokens
                        self._prompt_tokens += usage.prompt_tokens
                    if hasattr(usage, 'completion_tokens') and usage.completion_tokens:
                        completion_tokens = usage.completion_tokens
                        self._completion_tokens += usage.completion_tokens
                    # DeepSeek 返回：usage.prompt_tokens_details.cached_tokens
                    if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details is not None:
                        cache_hit_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                finish_reason = "stop"
                if resp.choices and resp.choices[0].finish_reason:
                    finish_reason = resp.choices[0].finish_reason
                return {
                    "content": content,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cache_hit_tokens": cache_hit_tokens,
                    "latency_ms": elapsed_ms,
                    "finish_reason": finish_reason,
                }

            except Exception as exc:
                self.logger.warning("LLM attempt %d/2 failed: %s", attempt + 1, exc)
                if attempt < 1:
                    time.sleep(1.0)  # 固定退避：云端场景快速失败优于长时间等待

        raise RuntimeError("LLM call failed after 2 attempts")

    # ------------------------------------------------------------------
    # trace 写入（不影响主流程）
    # ------------------------------------------------------------------
    def _write_trace(self, **fields) -> None:
        """追加一行 JSON 到 llm_traces/{YYYY-MM-DD}.jsonl。失败只记 warning。"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            traces_path = self.traces_dir / f"{today}.jsonl"
            # 确保目录存在（trace_dir 的父目录已在 resolve_artifact_dir 中创建，
            # 但 llm_traces 子目录可能在首次写入时才需要）
            traces_path.parent.mkdir(parents=True, exist_ok=True)
            fields.setdefault("timestamp", datetime.now().isoformat())
            fields.setdefault("trace_id", uuid.uuid4().hex)
            line = json.dumps(fields, ensure_ascii=False, default=str)
            with open(traces_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            self.logger.warning("Failed to write LLM trace: %s", exc)

    def get_traces(self, date: Optional[str] = None, limit: int = 100) -> list[dict]:
        """按日期读回 trace 记录。date 为 None 时读当天。"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        path = self.traces_dir / f"{date}.jsonl"
        if not path.exists():
            return []
        lines: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if len(lines) >= limit:
                    break
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        return lines

    # ------------------------------------------------------------------
    # 公开接口：chat() 签名向后兼容
    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1500,
             prompt_id: str = "", prompt_version: str = "", caller: str = "") -> str:
        """发送对话请求。"""
        if prompt_id:
            self.logger.debug("prompt=%s v=%s", prompt_id, prompt_version)
        self._total_calls += 1

        t_start = time.perf_counter()
        trace_base = {
            "caller": caller,
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system,
            "user": user,
        }

        # --- 降级或 client 未初始化 ---
        if self.is_mock or self._client is None:

            # 恢复冷却：距上次失败不足 60s 直接 mock，不卡重试
            if self._client is not None and time.perf_counter() - self._last_failure_ts < 60:
                response = self._mock_response(system, user)
                lat = (time.perf_counter() - t_start) * 1000
                self._meta = {"finish_reason": "", "is_mock": True, "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0, "latency_ms": lat}
                self._write_trace(
                    response=response, is_mock=True, error=None,
                    prompt_tokens=0, completion_tokens=0, cache_hit_tokens=0,
                    finish_reason="",
                    latency_ms=lat,
                    **trace_base,
                )
                return response

            # 尝试恢复
            if self._client is not None:
                try:
                    resp = self._do_chat(system, user, temperature, max_tokens)
                    self._consecutive_failures = 0
                    self.is_mock = False
                    self._meta = {"finish_reason": resp["finish_reason"], "is_mock": False, "prompt_tokens": resp["prompt_tokens"], "completion_tokens": resp["completion_tokens"], "cache_hit_tokens": resp["cache_hit_tokens"], "latency_ms": resp["latency_ms"]}
                    self.logger.info("LLM recovered from mock mode")
                    self._write_trace(
                        response=resp["content"], is_mock=False, error=None,
                        prompt_tokens=resp["prompt_tokens"],
                        completion_tokens=resp["completion_tokens"],
                        cache_hit_tokens=resp["cache_hit_tokens"],
                        finish_reason=resp["finish_reason"],
                        latency_ms=resp["latency_ms"],
                        **trace_base,
                    )
                    return resp["content"]
                except Exception as exc:
                    self.logger.debug("Recovery attempt failed: %s", exc)
                    self._last_failure_ts = time.perf_counter()

            response = self._mock_response(system, user)
            lat = (time.perf_counter() - t_start) * 1000
            self._meta = {"finish_reason": "", "is_mock": True, "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0, "latency_ms": lat}
            self._write_trace(
                response=response, is_mock=True, error=None,
                prompt_tokens=0, completion_tokens=0, cache_hit_tokens=0,
                finish_reason="",
                latency_ms=lat,
                **trace_base,
            )
            return response

        # --- 正常模式 ---
        try:
            resp = self._do_chat(system, user, temperature, max_tokens)
            self._consecutive_failures = 0
            self._meta = {"finish_reason": resp["finish_reason"], "is_mock": False, "prompt_tokens": resp["prompt_tokens"], "completion_tokens": resp["completion_tokens"], "cache_hit_tokens": resp["cache_hit_tokens"], "latency_ms": resp["latency_ms"]}
            self._write_trace(
                response=resp["content"], is_mock=False, error=None,
                prompt_tokens=resp["prompt_tokens"],
                completion_tokens=resp["completion_tokens"],
                cache_hit_tokens=resp["cache_hit_tokens"],
                finish_reason=resp["finish_reason"],
                latency_ms=resp["latency_ms"],
                **trace_base,
            )
            return resp["content"]
        except Exception as exc:
            self.logger.error("LLM call failed after 3 retries: %s", exc)
            self._consecutive_failures += 1
            if self._consecutive_failures >= 5:
                self.is_mock = True
                self._last_failure_ts = time.perf_counter()
                self.logger.error(
                    "LLM degraded to MOCK after %d consecutive failures",
                    self._consecutive_failures,
                )
            response = self._mock_response(system, user)
            lat = (time.perf_counter() - t_start) * 1000
            self._meta = {"finish_reason": "", "is_mock": True, "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0, "latency_ms": lat}
            self._write_trace(
                response=response, is_mock=True, error=str(exc),
                prompt_tokens=0, completion_tokens=0, cache_hit_tokens=0,
                finish_reason="",
                latency_ms=lat,
                **trace_base,
            )
            return response

    def chat_with_meta(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1500,
                       prompt_id: str = "", prompt_version: str = "", caller: str = "") -> dict:
        """调用 chat() 并返回完整元信息。"""
        response = self.chat(system, user, temperature, max_tokens,
                             prompt_id=prompt_id, prompt_version=prompt_version, caller=caller)
        return {
            "response": response,
            "finish_reason": self._meta.get("finish_reason", ""),
            "is_mock": self._meta.get("is_mock", False),
            "prompt_tokens": self._meta.get("prompt_tokens", 0),
            "completion_tokens": self._meta.get("completion_tokens", 0),
            "cache_hit_tokens": self._meta.get("cache_hit_tokens", 0),
            "latency_ms": self._meta.get("latency_ms", 0.0),
        }

    def _mock_response(self, _system: str, user: str) -> str:
        """Mock 响应：回显问题关键信息。"""
        preview = user[:200].replace("\n", " ")
        return f"[MOCK LLM] 收到提问: {preview}...（无 API Key 或调用失败，占位响应）"

    def get_usage_stats(self) -> dict:
        avg_time = self._total_time / max(self._total_calls, 1)
        est_cost = (
            self._prompt_tokens * _PRICE_PROMPT_PER_1K / 1000
            + self._completion_tokens * _PRICE_COMPLETION_PER_1K / 1000
        )
        return {
            "is_mock": self.is_mock,
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "avg_latency_s": round(avg_time, 2),
            "est_cost_cny": round(est_cost, 6),
            "pricing_note": "deepseek-v4-flash: $0.14/1M input, $0.28/1M output (2026-07核对)",
        }

    @classmethod
    def get_instance(cls) -> "LLMClient":
        """模块级单例。"""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance
