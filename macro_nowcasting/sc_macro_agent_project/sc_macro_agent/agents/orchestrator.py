"""Orchestrator: DataAgent → ModelAgent → AnalystAgent → CriticAgent。"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from ..step_tracker import StepTracker
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
      → Critic 不通过 → Analyst 重写（最多 1 轮）
      → 最终输出含 briefing + review + steps
    """

    MAX_REWRITES = 1

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        cfg = config or AppConfig()
        self.logger = get_logger("sc_macro_agent.orchestrator")
        self.llm = LLMClient.get_instance()
        self.agent = StepTracker(cfg.agent)
        self.data_agent = DataAgent()
        self.model_agent = ModelAgent(cfg)
        self.analyst_agent = AnalystAgent(cfg)
        self.critic_agent = CriticAgent()

    # ------------------------------------------------------------------
    # 私有：统一构造返回 dict
    # ------------------------------------------------------------------
    def _build_result(self, status: str, steps: list,
                      briefing: str = "", review: dict | None = None,
                      rewrite_rounds: int = 0, error: str | None = None,
                      blocking_issues: list | None = None,
                      tokens_before: int = 0) -> dict:
        """所有退出路径统一走此方法，保证 8 键齐全。"""
        usage = self.llm.get_usage_stats()
        usage["tokens_this_run"] = usage.get("total_tokens", 0) - tokens_before
        return {
            "status": status,
            "briefing": briefing,
            "review": review if review is not None else {},
            "steps": steps,
            "rewrite_rounds": rewrite_rounds,
            "token_usage": usage,
            "error": error,
            "blocking_issues": blocking_issues if blocking_issues is not None else [],
        }

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------
    def run(self, engine: PredictionEngine, on_step: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        steps: List[Dict[str, Any]] = []
        rewrite_rounds = 0
        tokens_before = self.llm.get_usage_stats().get("total_tokens", 0)
        briefing = ""
        review: dict = {}
        inputs: dict = {}

        def _notify(msg: str) -> None:
            if on_step is not None:
                try:
                    on_step(msg)
                except Exception as exc:  # 回调失败不影响主流程
                    self.logger.debug("on_step callback failed: %s", exc)

        # ================================================================
        # Step 1: DataAgent
        # ================================================================
        _notify("正在执行 DataAgent（数据审计）…")
        t0 = time.perf_counter()
        data_step = self.agent.start_step("data_agent")
        try:
            data_result = self.data_agent.run(engine)
            steps.append({"name": "data_agent", "result": data_result,
                           "elapsed_s": round(time.perf_counter() - t0, 2)})
            data_step.close("completed" if data_result["data_ok"] else "blocked", data_result)

            if not data_result["data_ok"]:
                return self._build_result(
                    status="blocked_data", steps=steps,
                    blocking_issues=data_result["blocking_issues"],
                    tokens_before=tokens_before,
                )
        except Exception as exc:
            self.logger.exception("DataAgent 执行失败")
            data_step.close("error", {"error": str(exc), "error_type": type(exc).__name__})
            steps.append({"name": "data_agent", "elapsed_s": round(time.perf_counter() - t0, 2),
                          "error": str(exc)})
            _notify(f"DataAgent 执行失败：{exc}")
            return self._build_result(
                status="failed_data_agent", steps=steps,
                error=str(exc), tokens_before=tokens_before,
            )

        # ================================================================
        # Step 2: ModelAgent
        # ================================================================
        _notify("正在执行 ModelAgent（训练/预测）…")
        t0 = time.perf_counter()
        model_step = self.agent.start_step("model_agent")
        try:
            model_result = self.model_agent.run(engine)
            steps.append({"name": "model_agent", "result": model_result,
                           "elapsed_s": round(time.perf_counter() - t0, 2)})
            model_step.close("completed", model_result)
        except Exception as exc:
            self.logger.exception("ModelAgent 执行失败")
            model_step.close("error", {"error": str(exc), "error_type": type(exc).__name__})
            steps.append({"name": "model_agent", "elapsed_s": round(time.perf_counter() - t0, 2),
                          "error": str(exc)})
            _notify(f"ModelAgent 执行失败：{exc}")
            return self._build_result(
                status="failed_model_agent", steps=steps,
                error=str(exc), tokens_before=tokens_before,
            )

        # ================================================================
        # Step 3: AnalystAgent
        # ================================================================
        _notify("正在执行 AnalystAgent（简报撰写）…")
        t0 = time.perf_counter()
        analyst_step = self.agent.start_step("analyst_agent")
        try:
            analyst_result = self.analyst_agent.run(engine)
            briefing = analyst_result["briefing"]
            inputs = analyst_result["inputs"]
            steps.append({"name": "analyst_agent",
                           "result": {"briefing_length": len(briefing)},
                           "elapsed_s": round(time.perf_counter() - t0, 2)})
            analyst_step.close("completed", {"briefing_length": len(briefing)})
        except Exception as exc:
            self.logger.exception("AnalystAgent 执行失败")
            analyst_step.close("error", {"error": str(exc), "error_type": type(exc).__name__})
            steps.append({"name": "analyst_agent", "elapsed_s": round(time.perf_counter() - t0, 2),
                          "error": str(exc)})
            _notify(f"AnalystAgent 执行失败：{exc}")
            return self._build_result(
                status="failed_analyst_agent", steps=steps,
                error=str(exc), tokens_before=tokens_before,
            )

        # ================================================================
        # Step 4: CriticAgent（含重写）
        # ================================================================
        critic_step = None
        t0 = 0.0
        try:
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
                critic_step.close("failed", {"high_issues": len(high_issues)})
                if rnd < self.MAX_REWRITES:
                    rewrite_rounds += 1
                    self.logger.info("Critic found %d high issues, rewriting (round %d)",
                                     len(high_issues), rewrite_rounds)
                    _notify(f"审阅未通过（{len(high_issues)} 个高优问题），AnalystAgent 重写…")

                    # ---- 重写子阶段（异常不致命，break 后走 failed_review 收尾） ----
                    t0_rewrite = time.perf_counter()
                    rewrite_step = self.agent.start_step(f"analyst_agent_rewrite_r{rnd}")
                    try:
                        rewrite_result = self.analyst_agent.rewrite(briefing, high_issues)
                        steps.append({"name": f"analyst_agent_rewrite_r{rnd}",
                                      "result": rewrite_result,
                                      "elapsed_s": round(time.perf_counter() - t0_rewrite, 2)})
                        if not rewrite_result.get("rewritten"):
                            rewrite_step.close("completed", {"rewritten": False})
                            self.logger.warning("rewrite 未生效（rewritten=False），终止重写循环")
                            break
                        briefing = rewrite_result["briefing"]
                        rewrite_step.close("completed", {"rewritten": True, "briefing_length": len(briefing)})
                    except Exception as exc:
                        self.logger.exception("AnalystAgent 重写异常")
                        rewrite_step.close("error", {"error": str(exc), "error_type": type(exc).__name__})
                        steps.append({"name": f"analyst_agent_rewrite_r{rnd}",
                                      "elapsed_s": round(time.perf_counter() - t0_rewrite, 2),
                                      "error": str(exc)})
                        _notify(f"重写失败：{exc}")
                        self.logger.warning("重写异常，保留重写前简报，终止重写循环")
                        break  # 非致命：保留重写前 briefing，走 failed_review 收尾
        except Exception as exc:
            self.logger.exception("CriticAgent 执行失败")
            if critic_step is not None:
                critic_step.close("error", {"error": str(exc), "error_type": type(exc).__name__})
            steps.append({"name": f"critic_agent_r{rewrite_rounds}", "elapsed_s": round(time.perf_counter() - t0, 2),
                          "error": str(exc)})
            _notify(f"CriticAgent 执行失败：{exc}")
            # 简报已生成，保留它；review 为软失败 dict 或空白
            return self._build_result(
                status="failed_critic_agent", steps=steps,
                briefing=briefing, review=review,
                rewrite_rounds=rewrite_rounds,
                error=str(exc), tokens_before=tokens_before,
            )

        # ================================================================
        # Final status
        # ================================================================
        passed = review.get("passed", False)
        has_error = review.get("critic_error", False)
        if has_error:
            status = "review_failed"
        elif passed:
            status = "passed_review"
        else:
            status = "failed_review"

        return self._build_result(
            status=status, steps=steps,
            briefing=briefing, review=review,
            rewrite_rounds=rewrite_rounds,
            tokens_before=tokens_before,
        )
