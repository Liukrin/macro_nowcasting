"""
报告生成模块。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from ..utils import json_dumps, pretty_quarter, save_json, save_text


class ReportBuilder:
    def build_markdown_summary(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(f"# {payload.get('project_name', '预测项目报告')}")
        lines.append("")
        lines.append("## 1. 运行摘要")
        lines.append("")
        lines.append(f"- 数据模式：{payload.get('dataset_mode')}")
        lines.append(f"- 目标指标：{payload.get('target_indicator')}")
        lines.append(f"- 选中模型：{payload.get('selected_model')}")
        lines.append(f"- 样本数：{payload.get('n_rows')}")
        lines.append(f"- 特征数：{payload.get('n_features')}")
        lines.append("")

        metrics = payload.get("metrics", {})
        if metrics:
            lines.append("## 2. 关键指标")
            lines.append("")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    lines.append(f"- {k}: {v:.6f}")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")

        prediction = payload.get("prediction", {})
        if prediction:
            lines.append("## 3. 最新预测")
            lines.append("")
            for k, v in prediction.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        factors = payload.get("factor_summary", {})
        if factors:
            lines.append("## 4. 因子摘要")
            lines.append("")
            lines.append(f"- 因子数：{factors.get('n_factors')}")
            lines.append(f"- 累计解释方差：{factors.get('cumulative_variance')}")
            lines.append("")

        feature_importance = payload.get("top_features", [])
        if feature_importance:
            lines.append("## 5. Top Features")
            lines.append("")
            for item in feature_importance[:15]:
                name = item.get("feature")
                score = item.get("combined_score") or item.get("abs_coefficient") or item.get("normalized_importance")
                lines.append(f"- {name}: {score}")
            lines.append("")
        return "\n".join(lines)

    def export_bundle(self, out_dir: str | Path, payload: Dict[str, Any]) -> Dict[str, str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = save_json(out_dir / "summary.json", payload)
        md_path = save_text(out_dir / "summary.md", self.build_markdown_summary(payload))
        return {
            "summary_json": str(json_path),
            "summary_md": str(md_path),
        }
