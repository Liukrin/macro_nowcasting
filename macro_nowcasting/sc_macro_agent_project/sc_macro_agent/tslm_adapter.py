"""
tslm Adapter / Text-Supervised Layer Mock.

这里不是接真实大模型，而是给未来扩展留接口：
- 自动解读数据质量报告
- 自动生成指标解释文本
- 自动拼装面试可讲述总结

wjm后面想接 OpenAI / 本地模型，
直接在这个 adapter 里替换实现就行
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class TSLMAdapter:
    def summarize_audit(self, audit_result: Dict[str, Any]) -> str:
        status = audit_result.get("status", "unknown")
        checks = audit_result.get("checks", [])
        failed = [c for c in checks if not c.get("passed")]
        if not failed:
            return "数据审计整体通过，字段与时间对齐基本正常，可进入特征工程和模型训练阶段。"
        return (
            f"数据审计状态为 {status}。"
            f"当前共有 {len(failed)} 项未通过或存在警告，"
            "建议优先补齐历史序列与缺失较高的局部指标。"
        )

    def summarize_prediction(self, prediction: Dict[str, Any]) -> str:
        return (
            f"模型 {prediction.get('model_name')} 给出的下一季度 "
            f"{prediction.get('target_indicator')} 预测值为 "
            f"{prediction.get('prediction_value'):.2f}。"
        )

    def build_interview_script(
        self,
        summary: Dict[str, Any],
        max_points: int = 6,
    ) -> List[str]:
        points = [
            "我把原来的 demo 改成了标准化长表输入 + 季度面板输出的工程流水线。",
            "数据层支持 real / demo / hybrid 三种模式，便于真实数据不足时做开发和回测。",
            "特征层除了现成季度聚合面板，也支持从月度长表重建 mean/last/std/trend 等季度特征。",
            "模型层不是单一模型，而是 baseline、正则化线性模型和残差混合模型一起比较。",
            "回测采用 expanding-window，避免普通随机切分带来的时序泄漏。",
            "我另外补了 Agent Orchestrator、ArtifactStore 和 Service Layer，让项目更像完整应用。"
        ]
        return points[:max_points]
