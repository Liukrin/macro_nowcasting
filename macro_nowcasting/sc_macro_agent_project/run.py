"""
CLI 入口。

示例：
python run.py audit --data-dir /path/to/data
python run.py train --data-dir /path/to/data
python run.py backtest --data-dir /path/to/data
python run.py predict --data-dir /path/to/data
python run.py agent --data-dir /path/to/data
"""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv
load_dotenv()

from sc_macro_agent import AppConfig, PredictionEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="四川省GDP混频预测 Agent Pipeline")
    parser.add_argument("command", choices=["audit", "train", "backtest", "predict", "agent", "status"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--artifact-dir", default="./artifacts")
    parser.add_argument("--dataset-mode", default="auto", choices=["auto", "real", "demo", "hybrid"])
    parser.add_argument("--debug", action="store_true")
    return parser


def build_config(args) -> AppConfig:
    config = AppConfig()
    config.debug = args.debug
    config.data.data_dir = args.data_dir
    config.data.artifact_dir = args.artifact_dir
    config.data.dataset_mode = args.dataset_mode
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    engine = PredictionEngine(config=config)

    from sc_macro_agent.llm.client import LLMClient
    LLMClient.set_artifact_dir(config.data.resolve_artifact_dir(create=False))

    if args.command == "audit":
        result = engine.audit_data(save_artifacts=True)
    elif args.command == "train":
        engine.audit_data(save_artifacts=False)
        engine.build_features()
        result = engine.train()
    elif args.command == "backtest":
        # 只用 audit_build_train，避免 goal 里含 backtest/report 导致重复执行
        engine.run_agent(goal="audit_build_train", save_artifacts=False)
        result = engine.backtest()
    elif args.command == "predict":
        # 只用 audit_build_train，避免 goal 里含 predict 导致 predict_next 跑两遍
        engine.run_agent(goal="audit_build_train", save_artifacts=False)
        result = engine.predict_next()
    elif args.command == "agent":
        from sc_macro_agent.agents import AgentOrchestrator
        orch = AgentOrchestrator(config=engine.config)
        result = orch.run(engine)
    else:
        engine.initialize()
        result = engine.get_status()

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
