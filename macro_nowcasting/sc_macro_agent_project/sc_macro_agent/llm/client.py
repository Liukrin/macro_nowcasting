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

# 整轮工具循环总耗时上限（阶段 2 使用，本阶段只提供常量与辅助函数）
_MAX_TOOL_LOOP_SECONDS = 90.0


# ================================================================
# 模块级工具函数
# ================================================================

def _strip_none(obj):
    """递归剔除 dict / list 中值为 None 的键。"""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def sanitize_assistant_message(msg: dict) -> dict:
    """清洗 assistant 消息，供追加回消息列表。

    递归剔除值为 None 的键，只保留 role / content / tool_calls 三个顶层字段。
    tool_calls 内部只保留 id / type / function（function 内保留 name / arguments）。
    阶段 2 工具循环 append 回消息列表时统一走此函数，避免冗余字段导致 400。
    """
    stripped = _strip_none(msg)

    cleaned: dict = {}
    if "role" in stripped:
        cleaned["role"] = stripped["role"]
    if "content" in stripped:
        cleaned["content"] = stripped["content"]
    else:
        cleaned["content"] = None

    if "tool_calls" in stripped and stripped["tool_calls"]:
        cleaned_tcs: list[dict] = []
        for tc in stripped["tool_calls"]:
            tc_clean: dict = {}
            if "id" in tc:
                tc_clean["id"] = tc["id"]
            if "type" in tc:
                tc_clean["type"] = tc["type"]
            if "function" in tc:
                fn: dict = {}
                if "name" in tc["function"]:
                    fn["name"] = tc["function"]["name"]
                if "arguments" in tc["function"]:
                    fn["arguments"] = tc["function"]["arguments"]
                if fn:
                    tc_clean["function"] = fn
            if tc_clean:
                cleaned_tcs.append(tc_clean)
        if cleaned_tcs:
            cleaned["tool_calls"] = cleaned_tcs

    return cleaned


def make_deadline() -> float:
    """返回工具循环的截止时间（perf_counter 值）。"""
    return time.perf_counter() + _MAX_TOOL_LOOP_SECONDS


def check_deadline(deadline: float) -> bool:
    """检查是否已超过整轮工具循环耗时上限。返回 True 表示未超时。"""
    return time.perf_counter() < deadline


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
    def _do_chat(self, messages: list[dict], temperature: float, max_tokens: int,
                 tools: list | None = None) -> dict:
        """发送请求，内部重试 2 次（固定 1s 退避），单次调用总耗时上限 60s。
        返回 {"content": str, "tool_calls": list[dict], "raw_message": dict|None,
               "prompt_tokens": int, "completion_tokens": int,
               "cache_hit_tokens": int, "latency_ms": float, "finish_reason": str}
        raw_message 落地前已经过 sanitize_assistant_message 清洗。
        """
        deadline = time.perf_counter() + _MAX_CHAT_TOTAL_SECONDS
        for attempt in range(2):
            if time.perf_counter() >= deadline:
                self.logger.error("LLM total time exceeded %.0fs budget; degrading to mock",
                                  _MAX_CHAT_TOTAL_SECONDS)
                raise RuntimeError(f"LLM call exceeded {_MAX_CHAT_TOTAL_SECONDS:.0f}s total budget")
            try:
                t0 = time.perf_counter()
                kwargs: dict = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools

                resp = self._client.chat.completions.create(**kwargs)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._total_time += elapsed_ms / 1000

                choice = resp.choices[0]
                content = choice.message.content or ""
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
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                # 序列化 tool_calls
                tool_calls: list[dict] = []
                msg = choice.message
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        })

                # 序列化并清洗 raw_message
                raw_message: dict | None = None
                try:
                    raw_dict = msg.model_dump()
                except AttributeError:
                    try:
                        raw_dict = msg.dict()
                    except AttributeError:
                        raw_dict = {"role": msg.role, "content": content}
                raw_message = sanitize_assistant_message(raw_dict)

                return {
                    "content": content,
                    "tool_calls": tool_calls,
                    "raw_message": raw_message,
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
    # 私有：核心降级/冷却/恢复/trace 逻辑
    # ------------------------------------------------------------------
    def _call_with_degradation(self, messages: list[dict], temperature: float,
                               max_tokens: int, tools: list | None,
                               trace_base: dict) -> dict:
        """核心 LLM 调用：包含降级/冷却/恢复/trace 全部逻辑。
        返回 {"content", "tool_calls", "raw_message", "finish_reason",
               "is_mock", "prompt_tokens", "completion_tokens",
               "cache_hit_tokens", "latency_ms"}
        """
        t_start = time.perf_counter()

        # --- 辅助：从 messages 取最后一条 user 内容用于 mock ---
        def _last_user_content() -> str:
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    return m.get("content", "")
            return ""

        # --- 辅助：构建 mock 结果 ---
        def _build_mock_result() -> dict:
            user_text = _last_user_content()
            response = self._mock_response("", user_text)
            lat = (time.perf_counter() - t_start) * 1000
            return {
                "content": response, "tool_calls": [], "raw_message": None,
                "finish_reason": "", "is_mock": True,
                "prompt_tokens": 0, "completion_tokens": 0,
                "cache_hit_tokens": 0, "latency_ms": lat,
            }

        # --- 辅助：从 _do_chat 返回值构建完整结果 ---
        def _build_result(resp: dict, is_mock: bool) -> dict:
            return {
                "content": resp["content"],
                "tool_calls": resp.get("tool_calls", []),
                "raw_message": resp.get("raw_message"),
                "finish_reason": resp["finish_reason"],
                "is_mock": is_mock,
                "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"],
                "cache_hit_tokens": resp["cache_hit_tokens"],
                "latency_ms": resp["latency_ms"],
            }

        # --- 辅助：设置 _meta 并写入 trace ---
        def _finalize(result: dict, is_mock: bool, error: str | None) -> None:
            self._meta = {
                "finish_reason": result["finish_reason"],
                "is_mock": is_mock,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "cache_hit_tokens": result["cache_hit_tokens"],
                "latency_ms": result["latency_ms"],
            }
            self._write_trace(
                response=result["content"],
                is_mock=is_mock,
                error=error,
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                cache_hit_tokens=result["cache_hit_tokens"],
                finish_reason=result["finish_reason"],
                latency_ms=result["latency_ms"],
                # tool_calls 写入 trace
                tool_calls=json.dumps(result.get("tool_calls", []), ensure_ascii=False),
                has_tool_call=bool(result.get("tool_calls")),
                **trace_base,
            )

        # --- 降级或 client 未初始化 ---
        if self.is_mock or self._client is None:

            # 恢复冷却：距上次失败不足 60s 直接 mock，不卡重试
            if self._client is not None and time.perf_counter() - self._last_failure_ts < 60:
                result = _build_mock_result()
                _finalize(result, True, None)
                return result

            # 尝试恢复
            if self._client is not None:
                try:
                    resp = self._do_chat(messages, temperature, max_tokens, tools)
                    self._consecutive_failures = 0
                    self.is_mock = False
                    self.logger.info("LLM recovered from mock mode")
                    result = _build_result(resp, False)
                    _finalize(result, False, None)
                    return result
                except Exception as exc:
                    self.logger.debug("Recovery attempt failed: %s", exc)
                    self._last_failure_ts = time.perf_counter()

            result = _build_mock_result()
            _finalize(result, True, None)
            return result

        # --- 正常模式 ---
        try:
            resp = self._do_chat(messages, temperature, max_tokens, tools)
            self._consecutive_failures = 0
            result = _build_result(resp, False)
            _finalize(result, False, None)
            return result
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
            result = _build_mock_result()
            _finalize(result, True, str(exc))
            return result

    # ------------------------------------------------------------------
    # 公开接口：chat_messages() —— 多轮消息 + function calling
    # ------------------------------------------------------------------
    def chat_messages(self, messages: list[dict], temperature: float = 0.3,
                      max_tokens: int = 1500, tools: list | None = None,
                      prompt_id: str = "", prompt_version: str = "",
                      caller: str = "") -> dict:
        """发送多轮对话请求，支持 function calling。

        Args:
            messages: OpenAI 格式的消息列表 [{"role":..., "content":...}, ...]
            temperature: 采样温度
            max_tokens: 最大输出 token 数
            tools: OpenAI 格式的 tools 定义列表，非空时传给 LLM
            prompt_id: 提示词 ID（用于 trace）
            prompt_version: 提示词版本（用于 trace）
            caller: 调用方标识（用于 trace）

        Returns:
            {"content", "tool_calls", "raw_message", "finish_reason",
             "is_mock", "prompt_tokens", "completion_tokens",
             "cache_hit_tokens", "latency_ms"}
        """
        if prompt_id:
            self.logger.debug("prompt=%s v=%s", prompt_id, prompt_version)
        self._total_calls += 1

        # 从 messages 提取 system / user 用于向后兼容的 trace 字段
        system = ""
        user = ""
        for m in messages:
            if isinstance(m, dict):
                if m.get("role") == "system":
                    system = m.get("content", "")
                elif m.get("role") == "user":
                    user = m.get("content", "")

        trace_base = {
            "caller": caller,
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # 旧字段：向后兼容 trace 看板
            "system": system,
            "user": user,
            # 新字段：多轮消息支持
            "messages": messages,
            "n_messages": len(messages),
        }

        return self._call_with_degradation(messages, temperature, max_tokens,
                                           tools, trace_base)

    # ------------------------------------------------------------------
    # 公开接口：chat() 签名向后兼容
    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1500,
             prompt_id: str = "", prompt_version: str = "", caller: str = "") -> str:
        """发送对话请求。签名向后兼容，内部委托给 chat_messages()。"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = self.chat_messages(messages, temperature=temperature,
                                    max_tokens=max_tokens,
                                    prompt_id=prompt_id,
                                    prompt_version=prompt_version,
                                    caller=caller)
        return result["content"]

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
