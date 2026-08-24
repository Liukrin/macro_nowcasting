"""
RAG 检索回归测试套件。

遍历 tests/rag_cases/*.json 逐个运行 rag.search()，按用例期望判定 PASS/FAIL，
并保存到 artifacts/prompt_evals/。

用例 JSON 字段：
- case_id（必需）
- description（必需）
- question（必需）
- expect_has_answer（必需，bool）：true=应检索到答案；false=应拒答
- expect_top1_contains（可选，str|null）：top1 文本应包含的子串
- min_score（可选，float）：top1 分数阈值，低于视为「无答案」
- known_fail（可选，str）：标注为已知失败的用例，结果不计入通过率统计之外的告警

判定规则：
- actual_has_answer = 检索非空 且 top1 分数 >= min_score（若指定）
- expect_has_answer=true  → 通过当且仅当 actual_has_answer 且
  （expect_top1_contains 为空，或 top1 文本含该子串）
- expect_has_answer=false → 通过当且仅当 not actual_has_answer

检索为 TF-IDF 分池检索，不依赖 LLM；引擎构建（数据 + 训练）为确定性流程。
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from sc_macro_agent.config import AppConfig
from sc_macro_agent.prediction_engine import PredictionEngine
from sc_macro_agent.rag_service import RAGService
from sc_macro_agent.llm.client import LLMClient


def load_cases(cases_dir: Path) -> list[dict]:
    cases = []
    for fp in sorted(cases_dir.glob("*.json")):
        cases.append(json.loads(fp.read_text(encoding="utf-8")))
    return cases


def judge(case: dict, sources: list) -> dict:
    """按用例期望判定单条结果，返回判定详情。"""
    expect_has_answer = bool(case["expect_has_answer"])
    min_score = case.get("min_score")
    expect_top1 = case.get("expect_top1_contains")

    actual_has_answer = len(sources) > 0
    top1_text = ""
    top1_score = 0.0
    if actual_has_answer:
        top1_text = sources[0][0]
        top1_score = float(sources[0][1])
        if min_score is not None and top1_score < float(min_score):
            actual_has_answer = False

    if expect_has_answer:
        contains_ok = True
        if expect_top1:
            contains_ok = expect_top1 in top1_text
        passed = actual_has_answer and contains_ok
        fail_reason = "" if passed else (
            "无答案" if not actual_has_answer else f"top1 不含期望子串「{expect_top1}」"
        )
    else:
        passed = not actual_has_answer
        fail_reason = "" if passed else "应拒答但检索到了结果"

    return {
        "expect_has_answer": expect_has_answer,
        "actual_has_answer": actual_has_answer,
        "expect_top1_contains": expect_top1,
        "top1_text_preview": top1_text[:100].replace("\n", " "),
        "top1_score": round(top1_score, 4),
        "n_results": len(sources),
        "passed": passed,
        "fail_reason": fail_reason,
    }


def main() -> int:
    cases_dir = _project_root / "tests" / "rag_cases"
    cases = load_cases(cases_dir)

    # --- 构建引擎 + RAG（确定性，一次） ---
    config = AppConfig()
    config.data.dataset_mode = "real"
    LLMClient.set_artifact_dir(config.data.resolve_artifact_dir(create=True))

    t0 = _time.perf_counter()
    print("Initializing engine ...")
    engine = PredictionEngine(config=config)
    engine.run_agent(goal="audit_build_train_backtest_report", save_artifacts=True)
    print(f"Engine ready: {engine.selected_model_name} ({_time.perf_counter() - t0:.1f}s)")

    rag = RAGService(config, engine)
    print(f"RAGService: {len(rag.documents)} docs "
          f"(card={len(rag.idx_card)}, doc={len(rag.idx_doc)})")

    header = (f"{'Case ID':<26} {'Expect':>6} {'Actual':>6} "
              f"{'Top1 score':>10} {'Contains':>8} {'Result':>7}")
    print("\n" + header)
    print("-" * len(header))

    results = []
    n_pass = 0
    n_known_fail = 0
    for case in cases:
        cid = case["case_id"]
        sources = rag.search(case["question"], top_k=5, min_score=0.15)
        j = judge(case, sources)
        known_fail = case.get("known_fail")

        if j["passed"]:
            n_pass += 1
        elif known_fail:
            n_known_fail += 1

        contains_flag = ""
        if case.get("expect_top1_contains"):
            contains_flag = "OK" if (j["expect_top1_contains"] in j["top1_text_preview"]) else "MISS"

        row = (f"{cid:<26} {str(j['expect_has_answer']):>6} "
               f"{str(j['actual_has_answer']):>6} {j['top1_score']:>10} "
               f"{contains_flag:>8} {'PASS' if j['passed'] else 'FAIL':>7}")
        print(row)
        if j["fail_reason"]:
            print(f"    ↳ {j['fail_reason']} | top1: {j['top1_text_preview'][:60]}")

        results.append({
            "case_id": cid,
            "description": case.get("description", ""),
            "question": case["question"],
            "tier": case.get("tier", ""),
            "known_fail": known_fail,
            **j,
        })

    total = len(cases)
    print("-" * len(header))
    print(f"\nTotal: {total} cases | PASS {n_pass} | FAIL {total - n_pass - n_known_fail} "
          f"| KNOWN_FAIL {n_known_fail} | pass_rate={n_pass / max(total, 1):.1%}")

    # --- 保存产物 ---
    artifact_dir = config.data.resolve_artifact_dir(create=True) / "prompt_evals"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ts = _time.strftime("%Y%m%dT%H%M%SZ")
    out_path = artifact_dir / f"rag_regression_{ts}.json"
    out_path.write_text(json.dumps({
        "eval_type": "rag_retrieval_regression",
        "timestamp": ts,
        "total_docs": len(rag.documents),
        "total_cases": total,
        "n_pass": n_pass,
        "n_fail": total - n_pass - n_known_fail,
        "n_known_fail": n_known_fail,
        "pass_rate": round(n_pass / max(total, 1), 4),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")

    return 0 if (n_pass + n_known_fail == total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
