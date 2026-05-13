"""
Pydantic schema 定义。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    initialized: bool
    dataset_mode: Optional[str] = None
    last_run_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class DataAvailabilityItem(BaseModel):
    name: str
    rows: int
    columns: int
    start: Optional[str] = None
    end: Optional[str] = None
    missing_ratio: float = 0.0
    indicators: int = 0
    regions: int = 0


class DataAvailabilityResponse(BaseModel):
    dataset_mode: str
    items: List[DataAvailabilityItem]
    notes: List[str] = Field(default_factory=list)


class PredictionRequest(BaseModel):
    target_indicator: str = Field(default="GDP_同比增速")
    horizon: int = Field(default=1, ge=1, le=4)
    use_blend: bool = True
    force_retrain: bool = False


class PredictionResponse(BaseModel):
    target_indicator: str
    prediction_quarter: str
    model_name: str
    prediction_value: float
    benchmark_value: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None
    components: Dict[str, float] = Field(default_factory=dict)
    top_features: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class TrainResponse(BaseModel):
    status: str
    selected_model: str
    n_rows: int
    n_features: int
    metrics: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class BacktestResponse(BaseModel):
    status: str
    selected_model: str
    n_windows: int
    metrics: Dict[str, float]
    by_window: List[Dict[str, Any]] = Field(default_factory=list)
    top_features: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class FactorSummaryItem(BaseModel):
    factor_name: str
    explained_variance: float
    top_loadings: List[Dict[str, Any]] = Field(default_factory=list)


class FactorSummaryResponse(BaseModel):
    n_factors: int
    cumulative_variance: float
    factors: List[FactorSummaryItem] = Field(default_factory=list)


class DataAuditResponse(BaseModel):
    status: str
    dataset_mode: str
    summary: Dict[str, Any]
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    artifacts: Dict[str, str] = Field(default_factory=dict)


class AgentTaskRequest(BaseModel):
    goal: str = "audit_build_train_backtest_report"
    force_refresh: bool = False
    save_artifacts: bool = True


class AgentTaskResponse(BaseModel):
    goal: str
    status: str
    steps: List[Dict[str, Any]]
    summary: Dict[str, Any]
    artifacts: Dict[str, str] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
