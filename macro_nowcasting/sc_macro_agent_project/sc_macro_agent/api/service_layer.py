"""
Service Layer
API 不直接碰 PredictionEngine 的内部细节，而是通过 Service 暴露较稳定的接口
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..prediction_engine import PredictionEngine


class ForecastService:
    def __init__(self, engine: PredictionEngine) -> None:
        self.engine = engine

    def health(self) -> Dict[str, Any]:
        return self.engine.get_status()

    def audit(self) -> Dict[str, Any]:
        return self.engine.audit_data(save_artifacts=True)

    def train(self) -> Dict[str, Any]:
        return self.engine.train()

    def backtest(self) -> Dict[str, Any]:
        return self.engine.backtest()

    def predict(self) -> Dict[str, Any]:
        return self.engine.predict_next()

    def agent_run(self, goal: str, force_refresh: bool = False, save_artifacts: bool = True) -> Dict[str, Any]:
        return self.engine.run_agent(goal=goal, force_refresh=force_refresh, save_artifacts=save_artifacts)
