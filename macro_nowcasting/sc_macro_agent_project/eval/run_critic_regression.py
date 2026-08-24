"""
Critic 回归测试套件。
遍历 tests/critic_cases/*.json 逐个运行，输出结果表格，
并保存到 artifacts/prompt_evals/。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sc_macro_agent.agents.critic_agent import CriticAgent
from sc_macro_agent.llm.client import LLMClient
from sc_macro_agent.prompts.registry import load_prompt


def load_cases(cases_dir: Path) -> list[dict]:
    cases = []
    for fp in sorted(cases_dir.glob("*.json")):
        case = json.loads(fp.read_text(encoding="utf-8"))
        cases.append(case)
    return cases


def fmt_bool(v: bool) -> str:
    return "PASS" if v else "FAIL"


def match_issue_types(actual_issues: list[dict], expected_types: list[str]) -> bool:
    """所有期望类型至少各命中一次。"""
    actual_types = {i.get("type", "") for i in actual_issues}
    return all(t in actual_types for t in expected_types)


def main() -> None:
    # --- setup ---
    cases_dir = Path(__file__).parent.parent / "tests" / "critic_cases"
    cases = load_cases(cases_dir)

    # --version 参数：指定提示词版本用于回归测试，不指定则用当前版本
    target_version = None
    if "--version" in sys.argv:
        idx = sys.argv.index("--version")
        if idx + 1 < len(sys.argv):
            target_version = sys.argv[idx + 1]

    prompt = load_prompt("critic_review", version=target_version)
    version = prompt.get("version", "0.0.0")

    critic = CriticAgent(prompt_version=target_version)

    # --- header ---
    header = f"{'Case ID':<32} {'Expect passed':>14} {'Got passed':>11} {'Issues hit':>11} {'Result':>7}"
    print(header)
    print("-" * len(header))

    results = []
    all_pass = True

    for case in cases:
        cid = case["case_id"]
        desc = case["description"]
        inputs = case["inputs"]
        expect_passed = case["expect_passed"]
        expected_types = case.get("expect_issue_types", [])

        result = critic.review(briefing=case["briefing"], structured_inputs=inputs)
        actual_passed = result.get("passed", False)
        actual_issues = result.get("issues", [])

        passed_check = (actual_passed == expect_passed)
        types_check = match_issue_types(actual_issues, expected_types)
        case_ok = passed_check and types_check

        if not case_ok:
            all_pass = False

        actual_types_str = ", ".join(i.get("type", "?") for i in actual_issues[:6])
        issues_hit = f"{len(actual_issues)} [{actual_types_str}]"

        row = (
            f"{cid:<32} "
            f"{str(expect_passed):>14} "
            f"{str(actual_passed):>11} "
            f"{issues_hit:<11} "
            f"{fmt_bool(case_ok):>7}"
        )
        print(row)

        results.append({
            "case_id": cid,
            "description": desc,
            "expect_passed": expect_passed,
            "actual_passed": actual_passed,
            "expect_issue_types": expected_types,
            "actual_issue_types": [i.get("type") for i in actual_issues],
            "issues_matched": types_check,
            "result": "PASS" if case_ok else "FAIL",
            "full_issues": actual_issues,
        })

    print("-" * len(header))
    print(f"\nTotal: {len(cases)} cases | {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    # --- cost / token 统计（仅本轮测试的调用） ---
    client = LLMClient.get_instance()
    usage = client.get_usage_stats()
    total_cost = sum(
        (t.get("prompt_tokens", 0) or 0) * 0.001 / 1000
        + (t.get("completion_tokens", 0) or 0) * 0.002 / 1000
        for t in usage.get("_traces_for_eval", [])
    ) if "_traces_for_eval" in usage else 0.0  # fallback: use cumulative stats
    # Use cumulative usage stats as a practical approximation
    est_cost = usage.get("est_cost_cny", 0)

    artifact_dir = (
        Path(__file__).parent.parent
        / "artifacts"
        / "prompt_evals"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "_backfill" if target_version else ""
    out_name = f"{version}{suffix}_{ts}.json"
    out_path = artifact_dir / out_name

    eval_record = {
        "prompt_id": "critic_review",
        "prompt_version": version,
        "backfill": target_version is not None,
        "eval_timestamp": ts,
        "total_cases": len(cases),
        "all_passed": all_pass,
        "results": results,
        "token_cost": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "avg_latency_s": usage.get("avg_latency_s", 0),
            "est_cost_cny": est_cost,
        },
    }
    out_path.write_text(json.dumps(eval_record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Eval saved: {out_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
