"""
模型基类。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..exceptions import ModelTrainingError

# 月度滞后特征的列名前缀
MIDAS_PREFIX = "midas__"


def select_model_features(model: "BaseForecastModel", X: pd.DataFrame) -> pd.DataFrame:
    """按模型的 uses_midas_lags 标记过滤 midas 列。

    若模型不需要 MIDAS 滞后特征，则从 X 中移除所有以 midas__ 开头的列；
    若需要则保留全量。

    Args:
        model: 预测模型实例，需有 uses_midas_lags 属性
        X: 特征 DataFrame

    Returns:
        过滤后的 DataFrame（不复制整个 frame，仅做列切片）

    Raises:
        ModelTrainingError: 若过滤后列数为 0
    """
    if getattr(model, "uses_midas_lags", False):
        return X
    cols = [c for c in X.columns if not c.startswith(MIDAS_PREFIX)]
    if not cols:
        raise ModelTrainingError(
            f"模型 {model.model_name} 过滤 midas 列后无可用特征"
        )
    return X[cols]


@dataclass
class ModelResult:
    model_name: str
    predictions: np.ndarray
    components: Dict[str, np.ndarray]
    feature_importance: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["predictions"] = self.predictions.tolist()
        payload["components"] = {k: v.tolist() for k, v in self.components.items()}
        return payload


class BaseForecastModel:
    model_name: str = "base"
    uses_midas_lags: bool = False  # 子类重写为 True 以接收 midas__ 前缀列

    def __init__(self) -> None:
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.train_predictions_: Optional[np.ndarray] = None
        self.train_residuals_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecastModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_result(self, X: pd.DataFrame) -> ModelResult:
        pred = self.predict(X)
        return ModelResult(
            model_name=self.model_name,
            predictions=np.asarray(pred, dtype=float),
            components={"prediction": np.asarray(pred, dtype=float)},
            feature_importance=self.get_feature_importance(),
            metadata=self.get_summary(),
        )

    def get_feature_importance(self, top_n: int = 20) -> List[Dict[str, Any]]:
        return []

    def get_summary(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "is_fitted": self.is_fitted}
