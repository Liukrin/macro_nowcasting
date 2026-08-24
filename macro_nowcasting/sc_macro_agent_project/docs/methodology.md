> 数据局限与估算说明以 docs/known_limitations.md 为准。

# 方法论

本文档描述系统实际采用的技术方法。每节内容基于代码实现编写，与立项方案不一致处已在节末注明。

## 1. MIDAS 混频数据抽样

本系统采用 MIDAS（Mixed Data Sampling）框架处理混频数据：将月度高频指标（工业增加值、社零、PMI 分项等）按季度聚合后，与季度 GDP 目标变量对齐到同一频率，再用线性回归建模。

月度指标的季度聚合方式为：对每个季度内的月度观测值计算多种统计量，包括均值（mean）、末值（last）、标准差（std）、最小值（min）、最大值（max）、线性趋势（trend）、季环比变化（delta）和取值范围（range），以及该季度内可用月数（available_months）。DFM 因子也被聚合为均值、末值、标准差和趋势四组统计量后进入特征矩阵。

建模阶段使用 scikit-learn 的 RidgeCV 或 ElasticNetCV 进行正则化线性回归，特征经 StandardScaler 标准化。RidgeCV 的 alpha 候选集为 [0.01, 0.1, 1.0, 10.0, 100.0]；ElasticNetCV 的 alpha 候选集为 [0.001, 0.01, 0.1, 1.0]，l1_ratio 候选集为 [0.1, 0.3, 0.5, 0.7, 0.9]，最大迭代次数 10000，使用 5 折交叉验证（样本不足时自动降至 min(3, n-1) 折）。

注意：立项方案设计为 Almon 多项式权重或其他经典 MIDAS 加权函数，当前实现采用 sklearn 的 Ridge/ElasticNet 交叉验证，未实现分布滞后多项式。原因是训练面板共 63 个可用季度（2010Q1 至 2025Q4，delta 差分丢弃首行），样本量不足以稳定估计多项式参数，直接正则化回归在实践中更稳健。

实现位置：models/midas_model.py, features/feature_engineering.py

## 2. DFM 动态因子模型

DFM（Dynamic Factor Model）用于从大量月度指标中提取少量共同因子，以降低 MIDAS 特征矩阵的维度。当前实现为 PCA-based 工程化因子抽取器，而非经典的状态空间 DFM。

流程：首先将 local 和 national 两张月度长表分别 pivot 为 wide 格式（行=日期，列=指标名），合并后做前向/后向填充和缺失值中位数填补。经 StandardScaler 标准化后，用 sklearn.decomposition.PCA 提取 n_factors=3 个主成分（实际数量受样本行数和列数约束，取 min(3, n_rows, n_cols)）。

提取的月度因子随后按季度聚合：计算每个季度内三个月的因子均值（mean）、末值（last）、标准差（std）和线性趋势（trend），产生 3×4=12 个季度因子特征。因子载荷矩阵（components_）按绝对值排序后可查询每个因子最重要的原始指标。

注意：立项方案设计为包含状态空间方程和卡尔曼滤波的经典 DFM，当前实现为 PCA 因子提取。选用 PCA 而非 Kalman 滤波的原因：(1) 状态空间模型需要同时估计转移矩阵、载荷矩阵、观测噪声协方差和状态噪声协方差，参数量相对 40 个季度的训练样本过大，迭代优化容易不收敛；(2) PCA 是闭式解（SVD），无迭代收敛风险，在小样本下数值稳定。代价是放弃了卡尔曼滤波对 ragged edge（季度内月度数据不完整）的递归处理能力——当前改为在特征工程阶段通过月度到季度的分组聚合来处理不完整月份，缺失的月份自然不计入聚合统计量。

实现位置：models/dfm_model.py, features/feature_engineering.py

## 3. Chronos 残差修正

模型训练后，用训练集残差序列作为 Chronos-Bolt 预训练时序模型的上下文，预测残差的下一步取值，并将其加到 delta 预测值上作为修正项。

技术细节：使用 amazon/chronos-bolt-tiny 模型，通过 HuggingFace 缓存本地加载（HF_HUB_OFFLINE=1，离线模式）。加载失败时自动降级，修正值置零不影响主流程。预测时以训练残差序列为 context（shape (1, n)），prediction_length=1，生成 num_samples 个样本后取中位数作为点预测、10th/90th 分位数作为区间。修正值以加法方式叠加到 delta 空间的预测值上。

依赖：需要安装 torch 和 chronos 包，均为可选依赖。未安装或模型权重缺失时，Chronos 自动置 failed=True，所有 predict 调用返回 (0.0, (0.0, 0.0))，不影响主预测链路。

实现位置：chronos_adapter.py, prediction_engine.py（predict_next 方法）

## 4. delta 差分参数化

目标变量 GDP 累计同比增速在建模前做一阶差分变换：Δy_t = y_t - y_{t-1}（target_transform="delta"）。模型在差分空间训练和预测，避免非平稳序列直接建模的统计风险，同时使训练目标近似零均值，改善正则化效果。

预测时执行 delta add-back 回加：y_hat_t = y_{t-1} + Δŷ_t，其中 y_{t-1} 取训练面板倒数第二行的 target_value（因为 nowcast 映射下 X_t 与 y_t 同属第 t 季度，差分基准应为 y_{t-1} 而非 y_t）。回测中的 metrics（RMSE、MAE 等）也在 level 空间计算，单位为"百分点"。若 Chronos 修正生效，修正值在 add-back 之后叠加：y_hat_t = y_{t-1} + Δŷ_t + e_hat。

当样本量不足以执行差分（面板仅有 1 行）时，不进行差分变换和 add-back，直接输出模型原始预测值并附加告警。

实现位置：prediction_engine.py（_apply_target_transform, predict_next）

## 5. expanding-window 回测

采用 ExpandingWindowBacktester 做滚动窗口回测，scheme="expanding"。参数配置：initial_train_quarters=24（前 24 个季度为初始训练集），max_test_windows=32（最多回测 32 个窗口）。

每个窗口的流程：用当前窗口的全部历史数据训练候选模型，在模型选择器（ModelSelector）上选出验证集 RMSE 最低的模型，用该模型预测窗口后的下一个季度的目标值。delta 模式下，窗口内训练在差分空间进行，预测值经 add-back 回到 level 空间后再与真实值计算 metrics。每个窗口的结果（train_end、test_quarter、actual、prediction、模型名）汇入 BacktestWindowResult 列表。

最终聚合指标包括 RMSE、MAE、MAPE、SMAPE、R² 和方向准确率（direction_accuracy），所有指标在 level 空间计算。若测试窗口数不足 min_required_test_windows=2，回测终止。

实现位置：models/backtesting.py

## 6. 模型选择与基线对比

候选模型池包含 5 个模型，由 ModelConfig.candidate_models 配置：2 个基线模型 —— last_value（上一期值）、mean_recent（近 4 期均值）；3 个 MIDAS 变体 —— ridge_midas（RidgeCV 正则化）、elastic_midas（ElasticNetCV 正则化）、hybrid_residual（RidgeCV 线性部分 + RandomForestRegressor 残差修正，树模型在训练行数 ≥12 时启用，参数为 n_estimators=200、max_depth=3、min_samples_leaf=2）。

注意：ModelFactory.create() 中还定义了 seasonal_naive 和 drift 两个基线的构造方法，但它们不在默认 candidate_models 列表中，不会被模型选择器自动评估。如需启用需修改配置。

模型选择由 ModelSelector 执行：对每个候选模型在训练/验证切分上做 fit + evaluate，选择验证集 RMSE 最低的模型作为 selected_model。所有候选模型的评估结果汇入 leaderboard，包含 RMSE、MAE、MAPE、SMAPE、R² 和方向准确率。MAPE/SMAPE 在差分口径下分母接近零，数值可能极大，解读模型优劣时应以 RMSE 和 MAE 为主。

注意：立项方案中提到 GBRT（Gradient Boosted Regression Trees），但当前实现中 hybrid_residual 的残差修正组件使用的是 RandomForestRegressor（bagging 集成）而非 GBRT（boosting 集成），两者是不同的集成学习方法。默认 candidate_models 列表中也不包含独立的 GBRT 模型。

实现位置：models/model_selection.py, config.py（ModelConfig.candidate_models）, models/hybrid_model.py

## 7. 发布时滞与信息集设定

特征侧通过 FeatureConfig.feature_vintage 控制「预测时点当季有多少个月的数据被视为已发布」，取值 full_quarter / two_month / one_month，默认 two_month。实现方式：在月度→季度聚合（_build_monthly_aggregated_panel）按 quarter_end 分组后，仅保留当季前 N 个月（N=3/2/1）再计算 mean/last/std/trend/delta/range 等聚合特征。

full_quarter 为泄漏上界参照（当季 3 个月全用，含尚未发布的第 3 个月）；two_month 为默认（预测当季时通常仅前两月数据已发布）；one_month 为最保守下限。指标级 ragged edge（按各指标真实发布时滞逐月截断）为后续方向，尚未实现。

三模式回测对比（elastic_midas 固定，32 窗口，level 空间）见 docs/vintage_comparison.md：full_quarter RMSE 3.139、two_month 3.115、one_month 2.989。实测结论是可用月数越少、点预测 RMSE 越低但方向准确率越低，详见该文档。

实现位置：config.py（FeatureConfig.feature_vintage）, features/feature_engineering.py（_build_monthly_aggregated_panel）
