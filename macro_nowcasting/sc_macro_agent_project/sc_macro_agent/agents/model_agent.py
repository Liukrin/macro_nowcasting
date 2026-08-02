"""ModelAgent: 调用已冻结的训练与预测流程，不调 LLM。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import AppConfig
from ..prediction_engine import PredictionEngine
from ..logging_utils import get_logger


class ModelAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger("sc_macro_agent.model_agent")

    def run(self, engine: PredictionEngine) -> Dict[str, Any]:
        # Train
        engine.run_agent(goal="audit_build_train", save_artifacts=False)

        # Predict next quarter
        pred = engine.predict_next()

        # Backtest metrics from engine.backtest_result（不依赖 artifacts/final，无则不填充 0）
        if engine.backtest_result is None:
            try:
                engine.backtest()
            except Exception as exc:
                self.logger.warning("backtest failed: %s", exc)

        backtest_rmse = None
        vs_baseline = None
        dir_acc = None
        if engine.backtest_result and engine.backtest_result.get("metrics"):
            m = engine.backtest_result["metrics"]
            backtest_rmse = float(m["rmse"])
            dir_acc = float(m.get("direction_accuracy", 0.0))
            baseline_rmse = _find_leaderboard_baseline(engine.leaderboard, "last_value")
            if baseline_rmse is not None and backtest_rmse > 0:
                vs_baseline = backtest_rmse / baseline_rmse
        else:
            self.logger.warning("No backtest_result available for ModelAgent metrics")

        # Caveats from known_limitations.md（缺失时返回明确中文占位说明）
        caveats = _load_caveats(self.config.data.resolve_artifact_dir(create=False))

        return {
            "prediction_quarter": pred.get("prediction_quarter", pred.get("nowcast_quarter", "N/A")),
            "nowcast_quarter": pred.get("nowcast_quarter", pred.get("prediction_quarter", "N/A")),
            "prediction_value": pred.get("prediction_value", 0.0),
            "model_name": pred.get("model_name", "unknown"),
            "confidence_interval": pred.get("confidence_interval") or {},
            "actual_value": pred.get("actual_value"),
            "nowcast_error": pred.get("nowcast_error"),
            "backtest_rmse": backtest_rmse,
            "vs_baseline_ratio": round(vs_baseline, 3) if vs_baseline is not None else None,
            "direction_accuracy": round(dir_acc, 3) if dir_acc is not None else None,
            "caveats": caveats,
        }


def _find_leaderboard_baseline(leaderboard: list, baseline_model: str = "last_value") -> Optional[float]:
    """从 leaderboard 中找基线的 RMSE。找不到返回 None（不伪造 0）。"""
    for entry in leaderboard or []:
        if isinstance(entry, dict) and entry.get("model_name") == baseline_model:
            rmse = entry.get("rmse")
            if rmse is not None:
                return float(rmse)
    return None


def _load_caveats(artifacts_dir: Path) -> list:
    path = artifacts_dir / "final" / "known_limitations.md"
    if not path.exists():
        logger = get_logger("sc_macro_agent.model_agent")
        logger.warning("Missing artifact: %s", path)
        return ["未生成《已知局限》说明文档；请以数据审计与回测结论为准，勿将本次预测视为官方口径"]
    text = path.read_text(encoding="utf-8")
    # Extract numbered limitation headers
    caveats = []
    for m in re.finditer(r'##\s*\d+\.\s*(.+)', text):
        caveats.append(m.group(1).strip())
    return caveats[:6]
