"""
Diebold-Mariano 检验（含 Harvey-Leybourne-Newbold 1997 小样本修正）。

用途：比较两组逐窗口预测误差序列是否在统计上可区分。

- 输入：两组对齐的逐窗口预测误差序列（e = prediction - actual）。
- 损失函数：平方误差（L(e)=e²）与绝对误差（L(e)=|e|）各跑一次。
- 小样本修正：本回测仅 32 窗口，属小样本。未修正的 DM 统计量在小样本下
  会高估显著性，故采用 Harvey-Leybourne-Newbold (1997) 修正：
      DM_HLN = DM × sqrt((T + 1 - 2h + h(h-1)/T) / T)
  并以 Student-t(T-1) 分布取双侧 p 值。
- 输出：DM 统计量、修正后 DM_HLN、p 值、样本量。

符号约定：d_t = L(e1_t) - L(e2_t)。d̄ > 0 表示模型 1 的损失更大（更差）。
拒绝原假设（p < 0.05）时，可断言两者预测精度不同。

阶段 2 回测产物（artifacts/vintage_comparison.json）只保存了聚合指标
（RMSE/MAE/…），未保存逐窗口预测。因此逐窗口误差需先补存到
artifacts/vintage_window_errors.json（见仓库内补存步骤），再由本脚本读取。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def diebold_mariano(
    e1: np.ndarray | list,
    e2: np.ndarray | list,
    loss: str = "squared",
    h: int = 1,
) -> Dict[str, float]:
    """对两组对齐的预测误差序列做 Diebold-Mariano 检验（含 HLN 修正）。

    Args:
        e1: 模型 1 的逐窗口误差序列（prediction - actual）。
        e2: 模型 2 的逐窗口误差序列，与 e1 长度相同、逐期对齐。
        loss: "squared"（平方误差）或 "absolute"（绝对误差）。
        h: 预测步长。本系统 horizon=1（one-step-ahead nowcast），故默认 1。

    Returns:
        含 dm / dm_hl / p_value / n / h / d_bar / hln_factor 的字典。
    """
    e1 = np.asarray(e1, dtype=float).ravel()
    e2 = np.asarray(e2, dtype=float).ravel()
    if e1.shape != e2.shape:
        raise ValueError(f"误差序列长度不一致: {e1.shape} vs {e2.shape}")
    T = int(e1.shape[0])
    if T < 2:
        raise ValueError(f"样本量过小: T={T} < 2")
    if h < 1:
        raise ValueError(f"预测步长 h 必须 >= 1，得到 {h}")
    if T <= h:
        raise ValueError(f"样本量 T={T} 不足以支撑 h={h} 步预测的 DM 检验")

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"未知损失函数: {loss!r}，合法值为 'squared' / 'absolute'")

    d_bar = float(np.mean(d))

    # Diebold-Mariano (1995) 的长期方差估计：γ̂_0 + 2·Σ_{k=1}^{h-1} γ̂_k，1/T 归一。
    gamma0 = float(np.sum((d - d_bar) ** 2) / T)
    var_hat = gamma0
    for k in range(1, h):
        var_hat += 2.0 * float(np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / T)

    if var_hat <= 0.0:
        raise ValueError("损失差分 d 的方差为 0，DM 统计量无定义（两组误差的损失恒相等）")

    dm = d_bar / np.sqrt(var_hat / T)

    # Harvey-Leybourne-Newbold (1997) 小样本修正。h=1 时退化为 sqrt((T-1)/T)。
    hln_factor = float(np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T))
    dm_hl = dm * hln_factor

    # 双侧 Student-t(T-1) p 值。
    p_value = float(2 * stats.t.sf(abs(dm_hl), df=T - 1))

    return {
        "loss": loss,
        "n": T,
        "h": h,
        "d_bar": d_bar,
        "dm": dm,
        "dm_hl": dm_hl,
        "hln_factor": hln_factor,
        "p_value": p_value,
    }


def _load_window_errors(path: Path) -> Dict[str, Dict[str, Dict[str, list]]]:
    """读取逐窗口产物，返回 {vintage: {model: {test_quarters, actual, prediction}}}。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _check_alignment(
    data: Dict[str, Dict[str, Dict[str, list]]],
) -> Tuple[bool, List[str]]:
    """检验三模式逐窗口预测是否对齐同一批测试期。

    返回 (aligned, messages)。未对齐时 messages 记录差异原因，供打印后中止。
    """
    msgs: List[str] = []

    # 收集所有 (vintage, model) 的 test_quarters 与 actual
    ref_key = None
    ref_quarters: list | None = None
    ref_actual: list | None = None
    for vintage, models in data.items():
        for model, rec in models.items():
            q = list(rec.get("test_quarters", []))
            a = list(rec.get("actual", []))
            p = list(rec.get("prediction", []))
            key = f"{vintage}/{model}"
            if len(q) != len(a) or len(q) != len(p):
                msgs.append(f"{key}: 期数/实际值/预测值长度不一致 "
                            f"(quarters={len(q)}, actual={len(a)}, pred={len(p)})")
                continue
            if ref_key is None:
                ref_key, ref_quarters, ref_actual = key, q, a
                continue
            if q != ref_quarters:
                msgs.append(f"{ref_key} 与 {key} 的 test_quarters 序列不一致")
            if a != ref_actual:
                msgs.append(f"{ref_key} 与 {key} 的 actual 序列不一致（目标值应不随 vintage/model 变化）")

    if msgs:
        return False, msgs

    # 汇总样本量信息
    n_set = {len(rec.get("test_quarters", []))
             for models in data.values() for rec in models.values()}
    msgs.append(f"对齐通过：所有 (vintage, model) 共 {len(data) * (len(next(iter(data.values()))) if data else 0)} 组，"
                f"窗口数一致 = {sorted(n_set)}，测试期与 actual 序列逐期一致。")
    return True, msgs


# 阶段 2 要求的四组两两比较。
PAIRS: List[Tuple[str, str, str, str, str]] = [
    ("full_quarter vs two_month", "full_quarter", "elastic_midas", "two_month", "elastic_midas"),
    ("full_quarter vs one_month", "full_quarter", "elastic_midas", "one_month", "elastic_midas"),
    ("two_month vs one_month", "two_month", "elastic_midas", "one_month", "elastic_midas"),
    ("two_month vs arima", "two_month", "elastic_midas", "two_month", "arima"),
]


def run_pairwise_tests(artifact_path: Path) -> Dict[str, list]:
    """从逐窗口产物读取误差，做对齐检查后跑全部两两 DM 检验。

    返回 {pair_label: [ {loss, dm, dm_hl, p_value, n, d_bar}, ... ]}。
    若对齐检查失败，抛出 RuntimeError（不强行比较）。
    """
    data = _load_window_errors(artifact_path)

    aligned, msgs = _check_alignment(data)
    for m in msgs:
        print(f"[对齐检查] {m}")
    if not aligned:
        raise RuntimeError("三模式逐窗口预测未对齐同一批测试期，中止检验。差异如下：\n" + "\n".join(msgs))

    errors: Dict[str, Dict[str, np.ndarray]] = {}
    for vintage, models in data.items():
        errors[vintage] = {}
        for model, rec in models.items():
            pred = np.asarray(rec["prediction"], dtype=float)
            actual = np.asarray(rec["actual"], dtype=float)
            errors[vintage][model] = pred - actual

    results: Dict[str, list] = {}
    for label, va, ma, vb, mb in PAIRS:
        e1 = errors[va][ma]
        e2 = errors[vb][mb]
        row = []
        for loss in ("squared", "absolute"):
            row.append(diebold_mariano(e1, e2, loss=loss))
        results[label] = row
    return results


def _fmt(r: Dict[str, float]) -> str:
    return (f"DM={r['dm']:+.4f}  DM_HLN={r['dm_hl']:+.4f}  "
            f"p={r['p_value']:.4f}  n={int(r['n'])}  d_bar={r['d_bar']:+.4f}")


def main(argv: list[str]) -> int:
    default = Path(__file__).resolve().parent.parent / "artifacts" / "vintage_window_errors.json"
    path = Path(argv[1]) if len(argv) > 1 else default
    if not path.exists():
        print(f"未找到逐窗口产物：{path}")
        print("请先补存逐窗口误差（确定性重跑阶段 2 回测并保存 window_results），再运行本检验。")
        return 1

    print(f"读取逐窗口产物：{path}")
    results = run_pairwise_tests(path)

    for label, rows in results.items():
        print(f"\n=== {label} ===")
        for r in rows:
            print(f"  [{r['loss']:<8}] {_fmt(r)}")
        sq = rows[0]
        absr = rows[1]
        print(f"  判读：平方误差 p={sq['p_value']:.4f}，绝对误差 p={absr['p_value']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
