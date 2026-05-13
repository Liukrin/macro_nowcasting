"""
FastAPI 入口。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sc_macro_agent import PredictionEngine, AppConfig
from sc_macro_agent.api.schemas import (
    AgentTaskRequest,
    AgentTaskResponse,
    BacktestResponse,
    DataAuditResponse,
    DataAvailabilityResponse,
    FactorSummaryResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    TrainResponse,
)


@lru_cache(maxsize=1)
def get_engine() -> PredictionEngine:
    config = AppConfig.from_env()
    return PredictionEngine(config=config)


app = FastAPI(
    title="四川省GDP混频预测 Agent Pipeline",
    version="2.0.0",
    description="围绕真实/演示混频数据构建的工程化预测服务",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, Any])
def root():
    return {
        "name": "四川省GDP混频预测 Agent Pipeline",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    engine = get_engine()
    status = engine.get_status()
    return HealthResponse(**status)


@app.get("/data/availability", response_model=DataAvailabilityResponse)
def data_availability():
    engine = get_engine()
    engine.initialize()
    payload = engine.data_manager.get_data_availability()
    return DataAvailabilityResponse(**payload)


@app.get("/data/audit", response_model=DataAuditResponse)
def data_audit():
    engine = get_engine()
    result = engine.audit_data(save_artifacts=True)
    return DataAuditResponse(**result)


@app.post("/train", response_model=TrainResponse)
def train():
    engine = get_engine()
    result = engine.train()
    return TrainResponse(**result)


@app.get("/backtest", response_model=BacktestResponse)
def backtest():
    engine = get_engine()
    try:
        result = engine.backtest()
        return BacktestResponse(
            status="ok",
            selected_model=engine.selected_model_name or "unknown",
            n_windows=result["n_windows"],
            metrics=result["metrics"],
            by_window=result["window_results"],
            top_features=engine.selected_model.get_feature_importance(10) if engine.selected_model else [],
            warnings=engine.warnings,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    engine = get_engine()
    if request.force_retrain:
        engine.train(force_rebuild=True)
    result = engine.predict_next(use_blend=request.use_blend)
    return PredictionResponse(**result)


@app.get("/factors", response_model=Dict[str, Any])
def factors():
    engine = get_engine()
    engine.build_features()
    return engine.get_factor_summary()


@app.post("/agent/run", response_model=AgentTaskResponse)
def run_agent(request: AgentTaskRequest):
    engine = get_engine()
    result = engine.run_agent(
        goal=request.goal,
        save_artifacts=request.save_artifacts,
        force_refresh=request.force_refresh,
    )
    return AgentTaskResponse(**result)
