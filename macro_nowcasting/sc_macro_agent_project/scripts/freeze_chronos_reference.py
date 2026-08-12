"""
冻结 Chronos 参考结果到仓库内静态文件。

本地运行（需 torch + chronos），生成 assets/chronos_reference.json，
该文件随代码提交，供线上无 torch 环境读取展示。

用法：
    cd sc_macro_agent_project
    python scripts/freeze_chronos_reference.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

# 确保可以从项目根导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sc_macro_agent import AppConfig, PredictionEngine
from sc_macro_agent.models.backtesting import ExpandingWindowBacktester

OUTPUT_PATH = _PROJECT_ROOT / "assets" / "chronos_reference.json"

def _get_chronos_version() -> str:
    try:
        import chronos
        return getattr(chronos, "__version__", "unknown")
    except Exception:
        return "unknown"


def main() -> None:
    config = AppConfig.from_env()
    engine = PredictionEngine(config=config)
    engine.initialize()
    engine.build_features()

    panel = engine.feature_artifacts.training_panel
    panel_t, base_series = engine._apply_target_transform(panel)
    feat_cols = engine.feature_artifacts.feature_columns
    target_col = engine.feature_artifacts.target_column

    # 数据 vintage：目标序列最后一个季度
    target_df = engine.bundle.quarterly_target
    target_f = target_df[
        target_df["indicator_name"].astype(str).str.contains("GDP", na=False)
    ]
    last_q = str(target_f["date"].max())[:10] if not target_f.empty else "unknown"

    backtester = ExpandingWindowBacktester(config.backtest, config.model)
    models = ["ridge_midas", "hybrid_residual", "hybrid_chronos"]

    results = {}
    window_details = {}
    for mname in models:
        bt = backtester.run(
            panel=panel_t,
            feature_cols=feat_cols,
            target_col=target_col,
            selected_model_name=mname,
            base_series=base_series,
        )
        results[mname] = {
            "rmse": bt["metrics"]["rmse"],
            "mae": bt["metrics"]["mae"],
            "r2": bt["metrics"]["r2"],
            "direction_accuracy": bt["metrics"].get("direction_accuracy", 0.0),
            "n_windows": bt["n_windows"],
        }
        window_details[mname] = bt["window_results"]

    import torch as _torch
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chronos_model_name": "amazon/chronos-bolt-tiny",
        "torch_version": _torch.__version__,
        "chronos_version": _get_chronos_version(),
        "data_vintage": last_q,
        "n_windows": bt["n_windows"],
        "results": results,
        "window_details": window_details,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Chronos reference frozen → {OUTPUT_PATH}")
    print(f"  vintage: {last_q}, n_windows: {bt['n_windows']}")
    for mname in models:
        r = results[mname]
        print(f"  {mname}: RMSE={r['rmse']:.4f}, DirAcc={r['direction_accuracy']:.1%}")


if __name__ == "__main__":
    main()
