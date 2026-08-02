"""经济简报生成器——修复时间边界：回顾+展望模式。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .client import LLMClient
from ..config import AppConfig
from ..prediction_engine import PredictionEngine
from ..logging_utils import get_logger
from ..utils import pretty_quarter


def _find_leaderboard_baseline(leaderboard: list, baseline_model: str = "last_value") -> Optional[float]:
    """从 leaderboard 中找基线的 RMSE。找不到返回 None（不伪造 0）。"""
    for entry in leaderboard or []:
        if isinstance(entry, dict) and entry.get("model_name") == baseline_model:
            rmse = entry.get("rmse")
            if rmse is not None:
                return float(rmse)
    return None


class BriefingGenerator:
    """生成四川省宏观经济简报。

    采用"回顾+展望"模式：
      - 第一段（本期概况）：as_of_quarter 的实际值 + 模型回顾
      - 第三段（下期预测）：as_of_quarter 的下一个季度，来自 predict_next()
    """

    def __init__(self, config: AppConfig) -> None:
        self.logger = get_logger("sc_macro_agent.briefing")
        self.llm = LLMClient.get_instance()
        self.config = config

    def build_inputs(self, engine: "PredictionEngine") -> Dict[str, Any]:
        """装配简报所需的结构化数据。返回 dict。"""
        data_dir = self.config.data.resolve_dir()

        # --- 数据截至季度（从 engine 实时取，不依赖 artifacts/final） ---
        panel = engine.feature_artifacts.training_panel if engine.feature_artifacts is not None else None
        if panel is None or panel.empty:
            raise RuntimeError("training_panel 为空，无法生成简报（没有可用季度数据）")
        sorted_panel = panel.sort_values("quarter_end")
        latest_row = sorted_panel.iloc[-1]
        as_of_quarter = pretty_quarter(latest_row["quarter_end"])
        actual_latest = float(latest_row["target_value"])

        # --- 近期 4 季度实际值 ---
        recent_lines = []
        for _, r in sorted_panel.tail(4).iterrows():
            recent_lines.append(
                f"{pretty_quarter(r['quarter_end'])}: GDP累计同比实际值 {float(r['target_value']):.1f}%"
            )

        # --- 下期预测（来自 predict_next，不是 backtest） ---
        try:
            pred = engine.predict_next()
            pred_quarter = pred.get("prediction_quarter", pred.get("nowcast_quarter", "N/A"))
            pred_value = pred.get("prediction_value", 0.0)
            pred_model = pred.get("model_name", "unknown")
            ci = pred.get("confidence_interval") or {}
        except Exception as exc:
            self.logger.error("predict_next failed: %s", exc)
            raise RuntimeError(f"无法生成预测，简报中止: {exc}") from exc

        # --- 回测指标（从 engine.backtest_result 实时取；无则不补跑，用占位文案说明） ---
        if engine.backtest_result is None:
            self.logger.warning("No backtest_result available; briefing will note backtest not run")
        metrics_text = ""
        bt = engine.backtest_result or {}
        m = bt.get("metrics")
        if m:
            metrics_text = (
                f"  选定模型({engine.selected_model_name or 'selected_model'}): "
                f"RMSE={float(m.get('rmse', 0.0)):.2f}, "
                f"MAE={float(m.get('mae', 0.0)):.2f}, "
                f"R²={float(m.get('r2', 0.0)):.2f}, "
                f"方向准确率={float(m.get('direction_accuracy', 0.0)):.1%}\n"
            )
            baseline_rmse = _find_leaderboard_baseline(engine.leaderboard, "last_value")
            if baseline_rmse is not None and float(m.get("rmse", 0.0)) > 0:
                ratio = baseline_rmse / float(m["rmse"])
                metrics_text += (
                    f"  基准对比: last_value RMSE={baseline_rmse:.2f}, "
                    f"模型/基准比值={ratio:.3f}\n"
                )
        else:
            metrics_text = "  （本次未运行回测，无可用评估指标）"

        # --- 月度指标 ---
        ml_path = data_dir / "monthly_local_features_real.csv"
        ind_lines = []
        if not ml_path.exists():
            self.logger.warning("Missing data file: %s", ml_path)
        else:
            ml = pd.read_csv(ml_path)
            ml["date"] = pd.to_datetime(ml["date"])
            ml_recent = ml[ml["date"] >= "2025-01-01"]
            for ind in sorted(ml_recent["indicator_name"].unique()):
                sub = ml_recent[ml_recent["indicator_name"] == ind].sort_values("date")
                if len(sub) > 0:
                    last = sub.iloc[-1]
                    ind_lines.append(
                        f"  {ind}: {last['date'].strftime('%Y-%m')} "
                        f"值 {last['indicator_value']:.1f}%"
                    )

        return {
            "as_of_quarter": as_of_quarter,
            "actual_latest": actual_latest,
            "pred_quarter": pred_quarter,
            "pred_value": pred_value,
            "pred_error": round(float(pred_value) - float(actual_latest), 2),
            "pred_model": pred_model,
            "ci_lower": ci.get("lower", "N/A"),
            "ci_upper": ci.get("upper", "N/A"),
            "recent_data": "\n".join(recent_lines),
            "metrics": metrics_text,
            "indicators": "\n".join(ind_lines[:12]),
        }

    def generate(self, inputs: Dict[str, Any]) -> str:
        """生成简报并做数字校验。"""
        from ..prompts.registry import render

        prompt = render("briefing",
            as_of=inputs["as_of_quarter"],
            actual_latest=inputs["actual_latest"],
            pred_quarter=inputs["pred_quarter"],
            pred_value=inputs["pred_value"],
            error=inputs["pred_error"],
            pred_model=inputs["pred_model"],
            ci_lower=inputs["ci_lower"],
            ci_upper=inputs["ci_upper"],
            recent_data=inputs["recent_data"],
            metrics=inputs["metrics"],
            indicators=inputs["indicators"],
        )

        answer = self.llm.chat(
            system=prompt["system"], user=prompt["user"],
            temperature=prompt["temperature"], max_tokens=prompt["max_tokens"],
            prompt_id=prompt["id"], prompt_version=prompt["version"],
            caller="briefing",
        )

        # 数字校验
        if not self._validate_numbers(answer, prompt["user"]):
            self.logger.warning("数字校验失败，重试一次")
            answer = self.llm.chat(
                system=prompt["system"], user=prompt["user"],
                temperature=0.2, max_tokens=prompt["max_tokens"],
                prompt_id=prompt["id"], prompt_version=prompt["version"],
                caller="briefing",
            )

        return answer

    def _validate_numbers(self, briefing: str, input_text: str) -> bool:
        """校验简报中的数字是否存在于输入数据中。"""
        briefing_nums = set()
        for m in re.finditer(r'(\d+\.?\d*)\s*%?', briefing):
            v = float(m.group(1))
            if v > 0.5:
                briefing_nums.add(round(v, 1))
        input_nums = set()
        for m in re.finditer(r'(\d+\.?\d*)', input_text):
            v = float(m.group(1))
            if v > 0.5:
                input_nums.add(round(v, 1))
        extra = briefing_nums - input_nums
        if extra:
            self.logger.warning("数字校验: 以下数字不在输入中: %s", extra)
            return False
        return True
