"""
Chronos-Bolt 残差修正适配器。
职责：输入一维残差序列，输出点预测 + 分位数区间。
全程离线，不做多步预测（horizon=1）。
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple
import numpy as np

from .logging_utils import get_logger


class ChronosResidualCorrector:
    """用 Chronos-Bolt 模型对残差序列做一步外推。

    使用模式：
        corrector = ChronosResidualCorrector("amazon/chronos-bolt-tiny")
        point, (lower, upper) = corrector.correct(residuals)

    若加载失败，所有 predict 调用返回 (0.0, (0.0, 0.0)) 并置 failed=True。
    """

    def __init__(self, model_name: str = "amazon/chronos-bolt-tiny") -> None:
        self.logger = get_logger("sc_macro_agent.chronos")
        self.model_name = model_name
        self.pipe = None
        self.failed = False
        self.failure_reason: Optional[str] = None
        self._torch = None
        self._total_calls = 0
        self._total_time = 0.0
        self._load()

    def _load(self) -> None:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            import torch
            self._torch = torch
        except ImportError as exc:
            self.failed = True
            self.failure_reason = "依赖未安装"
            self.logger.warning("Chronos loading failed (依赖未安装): %s", exc)
            return
        try:
            from chronos import ChronosBoltPipeline
        except ImportError as exc:
            self.failed = True
            self.failure_reason = "依赖未安装"
            self.logger.warning("Chronos loading failed (依赖未安装): %s", exc)
            return
        try:
            self.pipe = ChronosBoltPipeline.from_pretrained(
                self.model_name,
                device_map="cpu",
                dtype=torch.float32,
            )
            self.logger.info("Chronos model loaded: %s", self.model_name)
        except (OSError, FileNotFoundError) as exc:
            self.failed = True
            self.failure_reason = "模型权重缺失"
            self.pipe = None
            self.logger.warning("Chronos loading failed (模型权重缺失): %s", exc)
        except Exception as exc:
            self.failed = True
            self.failure_reason = "推理失败"
            self.pipe = None
            self.logger.warning("Chronos loading failed (推理失败): %s", exc)

    def correct(self, residuals: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """对残差序列做一步预测，返回 (点预测, (下分位, 上分位))。

        Args:
            residuals: 一维残差序列，shape (n,)。

        Returns:
            (point_prediction, (lower_bound, upper_bound))
            失败时返回 (0.0, (0.0, 0.0))。
        """
        if self.pipe is None or self.failed or self._torch is None:
            return 0.0, (0.0, 0.0)

        self._total_calls += 1
        t0 = time.perf_counter()

        try:
            # context: shape (batch=1, length=n)
            arr = np.asarray(residuals, dtype=np.float32)
            context = self._torch.tensor(arr).unsqueeze(0)

            with self._torch.no_grad():
                pred = self.pipe.predict(context, prediction_length=1)
                # pred shape: (1, num_samples, 1)
                samples = pred[0, :, 0].numpy()  # (num_samples,)

            elapsed = time.perf_counter() - t0
            self._total_time += elapsed

            point = float(np.median(samples))
            lower = float(np.percentile(samples, 10))
            upper = float(np.percentile(samples, 90))
            return point, (lower, upper)

        except Exception as exc:
            self.logger.warning("Chronos predict failed: %s", exc)
            self.failed = True
            return 0.0, (0.0, 0.0)

    @property
    def stats(self) -> dict:
        avg_time = self._total_time / max(self._total_calls, 1)
        return {
            "model_name": self.model_name,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "total_calls": self._total_calls,
            "avg_inference_ms": round(avg_time * 1000, 1),
        }
