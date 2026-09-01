"""
任务步骤追踪器（StepTracker）。

这个模块不碰大模型 API，也不是编排器本身，它只负责“分步执行留痕”：
为审计数据、构建特征、训练模型、回测、生成报告等步骤记录状态与产物，
供 Agent Orchestrator 使用。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import AgentConfig
from .logging_utils import get_logger


@dataclass
class AgentStep:
    name: str
    status: str
    started_at: str
    ended_at: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def close(self, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.status = status
        self.ended_at = datetime.now().isoformat(timespec="seconds")
        self.details = details or {}

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StepTracker:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.logger = get_logger("sc_macro_agent.step_tracker")
        self.steps: List[AgentStep] = []
        self.warnings: List[str] = []

    def start_step(self, name: str) -> AgentStep:
        step = AgentStep(name=name, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
        self.steps.append(step)
        return step

    def record_warning(self, message: str) -> None:
        self.warnings.append(message)
        self.logger.warning(message)

    def summary(self) -> Dict[str, Any]:
        return {
            "steps": [s.as_dict() for s in self.steps],
            "warnings": self.warnings,
        }
