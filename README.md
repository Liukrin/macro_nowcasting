# 四川省 GDP 混频现时预测系统

基于 MIDAS 混频回归与多智能体协作的省级 GDP 现时预测（nowcasting）演示系统。
输入月度高频指标，输出季度 GDP 累计同比增速点预测、置信区间、
自动生成的经济简报，以及基于检索的数据问答。

> **定位说明**：本项目为学术竞赛与个人实践产物，运行于开发环境，
> 未做生产级部署。已知局限见 [docs/known_limitations.md](macro_nowcasting/sc_macro_agent_project/docs/known_limitations.md)。

## 快速开始

```bash
git clone [仓库地址]
cd [仓库目录]
pip install -r requirements.txt
streamlit run macro_nowcasting/sc_macro_agent_project/app.py
```

需在 `.env` 或 Streamlit Secrets 中配置 `DEEPSEEK_API_KEY`。
未配置时 LLM 相关功能自动降级为 mock，主预测流程不受影响。

可选依赖（时序基础模型残差修正）：`pip install -r requirements-optional.txt`

## 数据

全部来自国家统计局与四川省统计局公开发布，无随机生成数据。

| 数据集 | 区间 | 规模 |
|---|---|---|
| 季度 GDP 目标变量 | 2010Q1–2025Q4 | 64 季度 |
| 四川月度指标 | 2010-02–2025-12 | 3 个指标 |
| 全国月度指标 | 2010-01–2025-12 | 18 个指标 |
| PMI 分项 | 2015-01–2025-12 | 14 个分项 |

2025Q4 GDP 为估算值；全国月度指标截止 2025-09，其后存在发布滞后缺口。
数据溯源见 [docs/data_lineage.md](macro_nowcasting/sc_macro_agent_project/docs/data_lineage.md)。

## 方法

- **特征工程**：月度→季度聚合（mean/last/std/trend/delta 等）、PCA 因子降维
- **发布时滞**：feature_vintage 三模式（full_quarter / two_month / one_month），默认 two_month
- **MIDAS 回归**：RidgeCV / ElasticNetCV 正则化；Exponential Almon Lag
  权重函数 B(k;θ)=exp(θ₁k+θ₂k²)/Σexp(θ₁j+θ₂j²)，对 θ 施加 L2 收缩
  以适应 64 季度小样本，参数触界时前端告警
- **残差修正**：Chronos-Bolt 时序基础模型（可选依赖，缺失时自动降级置零）
- **基线对比**：last_value / mean_recent / seasonal_naive / ARIMA
- **评测**：expanding-window 回测，初始训练 24 季度，32 个测试窗口

详见 [docs/methodology.md](macro_nowcasting/sc_macro_agent_project/docs/methodology.md)。

## 结果

默认 feature_vintage=two_month，模型选择器实际选中 ridge_midas（level 空间 32 窗口回测 RMSE=3.210）。
以下为固定 elastic_midas 的三模式对照（隔离 vintage 效应），level 空间、32 窗口：

| 配置（elastic_midas 固定） | RMSE | MAE | R² | 方向准确率 |
|---|---|---|---|---|
| full_quarter（当季 3 个月，泄漏上界） | 3.139 | 1.588 | 0.112 | 0.613 |
| two_month（前 2 个月，默认） | 3.115 | 1.574 | 0.126 | 0.548 |
| one_month（仅第 1 个月） | 2.989 | 1.455 | 0.195 | 0.548 |
| last_value（基线） | 4.609 | 2.175 | -0.914 | 0.581 |
| seasonal_naive（基线） | 4.586 | 2.800 | -0.895 | 0.581 |
| arima（基线） | 3.357 | 1.916 | -0.015 | 0.548 |

**结论**：三模式下 elastic_midas 均显著优于所有基线，且是唯一 R² 为正的模型。
但 vintage 影响方向与「泄漏假设」相反——可用月数越少、点预测 RMSE 越低
（3.139→3.115→2.989），而方向准确率反向（0.613→0.548）。说明季度内后两个月的
高频信息在当前累计同比口径下带来的是噪声而非信号，full_quarter 泄漏的是噪声而非信号。
这是**诚实信息集下的真实表现**，本次对比未调任何超参、未换模型、未改特征集、未改回测窗口数。

> **口径说明**：上表为 level 空间回测 RMSE（单位百分点）。模型选择器按 delta 空间
> 验证集 RMSE 排序，与 level 空间回测不可直接比较。完整对比与解读见
> [docs/vintage_comparison.md](macro_nowcasting/sc_macro_agent_project/docs/vintage_comparison.md)。

## 多智能体流水线

DataAgent → ModelAgent → AnalystAgent → CriticAgent，
Critic 五维审查不通过时回退 Analyst 重写（最多 1 轮）。

LLM 输出可靠性处理：Pydantic 宽松校验 + JSON 三级降级提取 +
截断自适应重试 + prompt 版本化与回归用例。

## 项目结构

```
sc_macro_agent_project/
├── app.py / main.py / run.py          # Streamlit / FastAPI / CLI 三个入口
├── requirements.txt                   # -r 指针 → 仓库根 requirements.txt
├── sc_macro_agent/                    # 核心包
│   ├── config.py                      # 分层配置（含 feature_vintage）
│   ├── prediction_engine.py           # 统一预测引擎（编排层）
│   ├── agent.py                       # 步骤留痕
│   ├── rag_service.py / tools.py      # RAG 问答 + function calling
│   ├── chronos_adapter.py / tslm_adapter.py  # 时序模型适配
│   ├── data/  features/  models/      # 数据 / 特征 / 建模
│   ├── agents/  llm/  prompts/        # 多智能体 / LLM 客户端 / 提示词
│   └── api/                           # 服务层 / 调度 / 上报
├── data/                              # 真实数据（CSV / Excel）
├── docs/                              # methodology / data_lineage / known_limitations / vintage_comparison
├── scripts/                           # 诊断 / 评测 / ETL 脚本
├── eval/                              # 评测入口（run_critic_regression.py）
├── tests/                             # 单元测试 + JSON 夹具
└── assets/                            # 静态参考（chronos_reference.json）
```

## 已知局限

本项目主动记录了 13 条局限，其中最重要的一条（feature_vintage）现已实现：

**feature_vintage 三模式已实现（默认 two_month）**——预测当季时通常仅前两月数据已发布，
故默认 two_month；full_quarter 为泄漏上界参照（含尚未发布的第 3 个月）；
指标级 ragged edge（按各指标真实发布时滞截断）为后续方向。

完整清单见 [docs/known_limitations.md](macro_nowcasting/sc_macro_agent_project/docs/known_limitations.md)。
