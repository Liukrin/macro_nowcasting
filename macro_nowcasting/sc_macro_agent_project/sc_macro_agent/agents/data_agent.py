"""DataAgent: 数据审计，不调 LLM。"""
from __future__ import annotations

import json
from typing import Any, Dict

from ..prediction_engine import PredictionEngine


class DataAgent:
    def run(self, engine: PredictionEngine) -> Dict[str, Any]:
        result = engine.audit_data(save_artifacts=False)
        blocking = [
            c["details"] for c in result.get("checks", [])
            if not c["passed"] and c.get("severity") == "error"
            and "required_columns" not in c.get("check_name", "")  # optional metadata columns
        ]
        warnings = [
            c["details"] for c in result.get("checks", [])
            if not c["passed"] and c.get("severity") != "error"
        ]
        indicators = len(result.get("summary", []))
        return {
            "data_ok": len(blocking) == 0,
            "blocking_issues": blocking,
            "warnings": warnings,
            "latest_quarter": _extract_latest_quarter(result),
            "usable_indicators": indicators,
        }


def _extract_latest_quarter(audit_result: Dict[str, Any]) -> str:
    for s in audit_result.get("summary", []):
        if s.get("name") == "quarterly_target":
            end = s.get("end_date", "")
            if end:
                from datetime import datetime
                try:
                    dt = datetime.strptime(end[:10], "%Y-%m-%d")
                    return f"{dt.year}Q{(dt.month-1)//3+1}"
                except:
                    pass
    return "N/A"
