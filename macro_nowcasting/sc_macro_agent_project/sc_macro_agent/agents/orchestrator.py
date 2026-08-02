"""Orchestrator: DataAgent → ModelAgent → AnalystAgent → CriticAgent。"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from ..agent import ForecastAgent
from ..config import AppConfig
from ..prediction_engine import PredictionEngine
from ..llm.client import LLMClient
from ..logging_utils import get_logger
from .data_agent import DataAgent
from .model_agent import ModelAgent
from .analyst_agent import AnalystAgent
from .critic_agent import CriticAgent


class AgentOrchestrator:
    """四角色多智能体流水线。

    流程：Data → Model → Analyst → Critic
      → Critic 不通过 → Analyst 重写（最多 2 轮）
      → 最终输出含 briefing + review + steps
    """

    MAX_REWRITES = 1

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        cfg = config or AppConfig()
        self.logger = get_logger("sc_macro_agent.orchestrator")
        self.llm = LLMClient.get_instance()
        self.agent = ForecastAgent(cfg.agent)
        self.data_agent = DataAgent()
        self.model_agent = ModelAgent(cfg)
        self.analyst_agent = AnalystAgent(cfg)
        self.critic_agent = CriticAgent()

    def run(self, engine: PredictionEngine, on_step: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        steps: List[Dict[str, Any]] = []
        rewrite_rounds = 0
        tokens_before = self.llm.get_usage_stats().get("total_tokens", 0)

        def _notify(msg: str) -> None:
            if on_step is not None:
                try:
                    on_step(msg)
                except Exception as exc:  # 回调失败不影响主流程
                    self.logger.debug("on_step callback failed: %s", exc)

        # --- Step 1: DataAgent ---
        _notify("正在执行 DataAgent（数据审计）…")
        t0 = time.perf_counter()
        data_step = self.agent.start_step("data_agent")
        data_result = self.data_agent.run(engine)
        steps.append({"name": "data_agent", "result": data_result,
                       "elapsed_s": round(time.perf_counter() - t0, 2)})
        data_step.close("completed" if data_result["data_ok"] else "blocked", data_result)

        if not data_result["data_ok"]:
            return {"status": "blocked_data", "steps": steps,
                    "blocking_issues": data_result["blocking_issues"]}

        # --- Step 2: ModelAgent ---
        _notify("正在执行 ModelAgent（训练/预测）…")
        t0 = time.perf_counter()
        model_step = self.agent.start_step("model_agent")
        model_result = self.model_agent.run(engine)
        steps.append({"name": "model_agent", "result": model_result,
                       "elapsed_s": round(time.perf_counter() - t0, 2)})
        model_step.close("completed", model_result)

        # --- Step 3: AnalystAgent ---
        _notify("正在执行 AnalystAgent（简报撰写）…")
        t0 = time.perf_counter()
        analyst_step = self.agent.start_step("analyst_agent")
        analyst_result = self.analyst_agent.run(engine)
        briefing = analyst_result["briefing"]
        inputs = analyst_result["inputs"]
        steps.append({"name": "analyst_agent",
                       "elapsed_s": round(time.perf_counter() - t0, 2)})
        analyst_step.close("completed", {"briefing_length": len(briefing)})

        # --- Step 4: CriticAgent ---
        review = None
        for rnd in range(1 + self.MAX_REWRITES):
            _notify(f"正在执行 CriticAgent（审阅第 {rnd + 1} 轮）…")
            t0 = time.perf_counter()
            critic_step = self.agent.start_step(f"critic_agent_r{rnd}")
            review = self.critic_agent.review(briefing, inputs)
            steps.append({"name": f"critic_agent_r{rnd}", "review": review,
                           "elapsed_s": round(time.perf_counter() - t0, 2)})

            if review.get("critic_error"):
                critic_step.close("error", {"critic_error": True})
                break

            high_issues = [i for i in review.get("issues", [])
                           if i.get("severity") == "high"]
            if review.get("passed") or not high_issues:
                critic_step.close("passed", {"issues": len(review.get("issues", []))})
                break

            # Rewrite needed
            rewrite_rounds += 1
            critic_step.close("failed", {"high_issues": len(high_issues)})
            if rnd < self.MAX_REWRITES:
                self.logger.info("Critic found %d high issues, rewriting (round %d)",
                                 len(high_issues), rewrite_rounds)
                _notify(f"审阅未通过（{len(high_issues)} 个高优问题），AnalystAgent 重写…")
                analyst_result = self.analyst_agent.run(engine, critic_feedback=high_issues)
                briefing = analyst_result["briefing"]

        # --- Final status ---
        passed = review.get("passed", False) if review else False
        has_error = review.get("critic_error", False) if review else False
        if has_error:
            status = "review_failed"
        elif passed:
            status = "passed_review"
        else:
            status = "failed_review"

        usage = self.llm.get_usage_stats()
        usage["tokens_this_run"] = usage.get("total_tokens", 0) - tokens_before
        return {
            "status": status,
            "briefing": briefing,
            "review": review,
            "steps": steps,
            "rewrite_rounds": rewrite_rounds,
            "token_usage": usage,
        }
