"""
调度器。

真要上线时你可以把它替换成 Airflow / Prefect / cron。
这里先给一个本地轻量版。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import time

from ..logging_utils import get_logger


@dataclass
class ScheduledRunRecord:
    name: str
    started_at: str
    ended_at: Optional[str]
    status: str
    details: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SimpleScheduler:
    def __init__(self) -> None:
        self.logger = get_logger("sc_macro_agent.scheduler")
        self.records: List[ScheduledRunRecord] = []

    def run_once(self, name: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        record = ScheduledRunRecord(
            name=name,
            started_at=datetime.now().isoformat(timespec="seconds"),
            ended_at=None,
            status="running",
            details={},
        )
        self.records.append(record)
        try:
            result = fn()
            record.ended_at = datetime.now().isoformat(timespec="seconds")
            record.status = "completed"
            record.details = {"result_keys": list(result.keys())}
            return result
        except Exception as exc:
            record.ended_at = datetime.now().isoformat(timespec="seconds")
            record.status = "failed"
            record.details = {"error": str(exc)}
            raise

    def every_seconds(self, seconds: int, iterations: int, name: str, fn: Callable[[], Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for _ in range(iterations):
            outputs.append(self.run_once(name, fn))
            time.sleep(seconds)
        return outputs

    def summary(self) -> List[Dict[str, Any]]:
        return [r.as_dict() for r in self.records]
