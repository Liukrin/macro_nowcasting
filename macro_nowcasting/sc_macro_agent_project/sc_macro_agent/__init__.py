"""
sc_macro_agent
==============

一个围绕“四川省 GDP 混频预测”重构出的工程化项目骨架
强调几个点：

1. 数据以“标准化长表”为中心，方便接真实统计局数据
2. 特征工程与建模解耦，便于以后替换模型
3. 模型层支持小样本友好回退策略，避免真实数据太短时直接报错
4. 既能脚本运行，也能挂 FastAPI / Streamlit
5. 额外提供轻量 Agent Orchestrator，让项目在叙事上更接近 AI Agent 工程应用

目标变量是四川省季度 GDP 同比增速
"""
from .config import AppConfig, DataConfig, FeatureConfig, ModelConfig, BacktestConfig, AgentConfig
from .prediction_engine import PredictionEngine

__all__ = [
    "AppConfig",
    "DataConfig",
    "FeatureConfig",
    "ModelConfig",
    "BacktestConfig",
    "AgentConfig",
    "PredictionEngine",
]

__version__ = "2.0.0"
