"""
OpenAI 格式的 function calling 工具定义与分发器。

4 个只读查询函数挂载在 RAGService 上作为方法（需要访问 self.documents / self.engine），
本模块只负责导出 TOOL_SCHEMAS 常量和 dispatch() 分发器。
"""

from __future__ import annotations

import copy
from typing import Any

# ================================================================
# OpenAI Function Calling 工具定义（4 个 JSON Schema）
# ================================================================

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_indicator",
            "description": (
                "查询指定经济指标在特定时间的数值。"
                "year 为 4 位整数年份（如 2025），quarter 为 1-4 的季度（如 2 表示第二季度），"
                "month 为 1-12 的月份（如 6 表示 6 月）。"
                "quarter 和 month 互斥，只能指定其中一个，不可同时提供。"
                "region 默认为'四川省'。"
                "indicator 支持模糊匹配，可写简称。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "description": "指标名称，支持模糊匹配。如'GDP累计同比增速'、'工业增加值同比增速'、'社零同比增速'等。",
                    },
                    "year": {
                        "type": "integer",
                        "description": "4 位整数年份，如 2025。",
                    },
                    "quarter": {
                        "type": "integer",
                        "description": "季度，取值为 1、2、3 或 4。与 month 互斥，二选一。",
                    },
                    "month": {
                        "type": "integer",
                        "description": "月份，取值为 1-12。与 quarter 互斥，二选一。",
                    },
                    "region": {
                        "type": "string",
                        "description": "地区名称，默认为'四川省'。",
                    },
                },
                "required": ["indicator", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_indicators",
            "description": (
                "列出所有可查询的经济指标及其数据覆盖范围。"
                "返回每个指标的名称、地区、频率（monthly/quarterly）、"
                "数据起止日期和观测数量。无需参数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_leaderboard",
            "description": (
                "获取模型排行榜，按验证集差分口径 RMSE 升序排列。"
                "包含每个模型的排名、RMSE、MAE、MAPE、SMAPE、R²、是否为系统选用模型、是否为基线模型。"
                "mape/smape 在差分口径下分母接近零，数值可能极大（>10000），"
                "解读模型优劣时请只用 rmse 和 mae。"
                "无需参数。预测引擎未初始化时返回 available=false。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prediction",
            "description": (
                "获取当前 nowcast 目标季度的 GDP 预测值、90% 置信区间、"
                "及所用模型名称。无需参数。预测引擎未初始化时返回 available=false。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ================================================================
# 分发器：根据 tool name 路由到 RAGService 对应方法
# ================================================================

def dispatch(rag: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """将 function calling 的 tool call 分发到 RAGService 的对应方法。

    Args:
        rag: RAGService 实例（需要已构建好 self.documents / self.engine）。
        name: 工具函数名，为 "query_indicator" / "list_indicators" /
              "get_model_leaderboard" / "get_prediction" 之一。
        arguments: LLM 传入的工具参数字典。

    Returns:
        dict: 工具执行结果。
    """
    if name == "query_indicator":
        return rag.query_indicator(
            indicator=arguments.get("indicator", ""),
            year=arguments.get("year"),
            quarter=arguments.get("quarter"),
            month=arguments.get("month"),
            region=arguments.get("region", "四川省"),
        )
    elif name == "list_indicators":
        return rag.list_indicators()
    elif name == "get_model_leaderboard":
        return rag.get_model_leaderboard()
    elif name == "get_prediction":
        return rag.get_prediction()
    else:
        return {"error": f"未知工具: {name}"}


# ================================================================
# 工具 schema 构建：从语料注入真实指标名，与语料自动对齐
# ================================================================

def build_tool_schemas(rag: Any | None) -> list[dict]:
    """深拷贝 TOOL_SCHEMAS，将真实指标名注入 query_indicator 的 description。

    rag 为 None 或取不到指标时，返回未注入的深拷贝（不抛异常）。
    调用方应缓存结果；模块常量 TOOL_SCHEMAS 永远不会被修改。
    """
    schemas = copy.deepcopy(TOOL_SCHEMAS)

    # 尝试取真实指标名
    indicator_names: list[str] = []
    if rag is not None:
        try:
            result = rag.list_indicators()
            names = sorted(set(
                e["name"] for e in result.get("indicators", [])
            ))
            indicator_names = names[:25]
        except Exception:
            pass

    if not indicator_names:
        return schemas

    names_str = "、".join(indicator_names)
    suffix = f"\n可用指标名（请优先使用其中之一）：{names_str}"

    for schema in schemas:
        if schema.get("function", {}).get("name") == "query_indicator":
            schema["function"]["description"] += suffix
            break

    return schemas
