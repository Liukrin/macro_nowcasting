# Bad Cases 归因（评测集扩容后）

生成时间：2026-08-23
评测套件：
- RAG 检索回归：`eval/run_rag_regression.py`（TF-IDF 分池检索，无需 LLM）
- Critic 回归：`eval/run_critic_regression.py`（真实 LLM，DeepSeek deepseek-v4-flash，prompt v1.3.0）

## 总览

| 套件 | 用例数 | 通过 | 失败 | 已知失败 | 通过率 |
|---|---|---|---|---|---|
| rag_cases | 20 | 16 | 3 | 1 | 80% |
| critic_cases | 20 | 16 | 4 | 0 | 80% |

说明：通过率如实记录，未为提升通过率改动任何判定标准或删除用例。

## 归因分类（5 类）

1. **语义鸿沟**：TF-IDF 词汇匹配无法跨越语义等价（如「生产」→「规模以上工业增加值」、「已知局限」→「chronos-2-small」），需 embedding 检索。
2. **文档池未按时间/指标约束过滤**：doc 池（forward_prediction / 数据覆盖等文档）不参与 card 池的时间/指标过滤，导致无关文档逃过约束。
3. **pred_value 舍入口径不一致**：`CriticAgent.review()` 把结构化输入 `pred_value=5.37` 舍入为 `5.4%`，而干净简报写 `5.37%`，被误报为「数字-语境匹配」。
4. **时间语义口径不一致**：结构化输入 `as_of_quarter=2025Q4` 与 `pred_quarter=2026Q1` 不一致，叠加「nowcast 复现」措辞，导致干净简报被误报「时间一致性」。
5. **critic 输出结构异常**：`passed=false` 但 `issues` 为空，无有效问题却被判不通过。

## RAG 失败明细

| 用例 | 期望 | 实际 | 归因 |
|---|---|---|---|
| q_known_limits | top1 含「chronos」 | top1 = 已知局限文档标题「# 已知局限…」（score 0.526） | ① 语义鸿沟 |
| q_no_data_future | 应拒答（2030 年为未来） | 检索到 forward_prediction 文档（score 0.428） | ② 文档池未按时间过滤 |
| q_sichuan_2025m6 | top1 含「2025年6月」 | top1 = 「数据覆盖」文档（score 0.456） | ② 文档池未按指标过滤 |
| q_production_semantic（known_fail） | top1 含「工业增加值」 | top1 = 「指标口径混杂」文档（score 0.126） | ① 语义鸿沟 |

## Critic 失败明细（均为「应通过」的干净用例被误判）

| 用例 | 期望 | 实际（issue type + quote 摘要） | 归因 |
|---|---|---|---|
| clean_01 | 通过 | 数字-语境匹配「…5.37%…」；时间一致性「…实际值为5.5%…」 | ③④ |
| clean_02 | 通过 | passed=false，issues 为空 | ⑤ |
| clean_03 | 通过 | 数字-语境匹配 ×2「…5.37%…」「…5.5%…」 | ③ |
| clean_04 | 通过 | 时间一致性「…5.37%…」；数字-语境匹配「5.37%」 | ③④ |

## 结论

- RAG：16/20 通过。3 个失败中 2 个源于「文档池不参与约束过滤」（可修复的工程缺陷），1 个源于「语义鸿沟」（已在 known_limitations #11 记录的架构局限）。
- Critic：16/20 通过。4 个失败全部是「干净用例被误判」，根因是 `pred_value` 舍入（③）与 nowcast 时间语义口径（④），以及 1 例输出结构异常（⑤）。这些是 critic 配置/prompt 口径问题，非用例本身错误。
- 本轮只记录，不修 bad case、不改判定标准、不做 prompt A/B。
