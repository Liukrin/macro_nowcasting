# feature_vintage 三模式回测对比

生成时间：2026-08-23（阶段 2 实测）
模型：elastic_midas（固定），expanding-window 回测（initial_train_quarters=24，max_test_windows=32，32 窗口）
口径：level 空间，RMSE / MAE 单位均为「百分点」。

## 对比表

| 模型 | RMSE | MAE | R² | 方向准确率 |
|---|---|---|---|---|
| elastic_midas — full_quarter（3 个月） | 3.139 | 1.588 | 0.112 | 0.613 |
| elastic_midas — two_month（前 2 个月） | 3.115 | 1.574 | 0.126 | 0.548 |
| elastic_midas — one_month（仅第 1 个月） | 2.989 | 1.455 | 0.195 | 0.548 |
| last_value（基线） | 4.609 | 2.175 | -0.914 | 0.581 |
| seasonal_naive（基线） | 4.586 | 2.800 | -0.895 | 0.581 |
| arima（基线） | 3.357 | 1.916 | -0.015 | 0.548 |

注：三个基线（last_value / seasonal_naive / arima）只用目标序列、不读月度特征，因此对 vintage 不敏感，三种模式下数值完全相同。

## 结论

三模式下 elastic_midas 均显著优于所有基线（RMSE 2.99–3.14，对比 last_value 4.61、seasonal_naive 4.59、arima 3.36），且是唯一 R² 为正的模型。

但 vintage 的影响方向与「泄漏假设」相反：**可用月数越少，点预测的 RMSE 反而越低**（full_quarter 3.139 → two_month 3.115 → one_month 2.989，one_month 较 full 改善约 0.15 个百分点），而**方向准确率朝反方向走**（full_quarter 0.613 最高，two_month / one_month 均跌到 0.548，与 arima 持平）。

这说明三点：

1. 季度内第 2、3 个月的高频信息在当前口径下**没有带来点预测收益，反而引入噪声**——可能因为四川月度指标均为累计同比（ytd_yoy），季度内三个月高度共线，后两个月叠加了未修订的初值噪声；full_quarter 所「泄漏」的是噪声而非信号。
2. 方向判断对「更多月份」更敏感：只有 full_quarter 的方向准确率（0.613）能超过 last_value / seasonal_naive（0.581），砍掉月份后方向判断退化为与 arima（0.548）相当。
3. 这是在**诚实信息集下的真实表现**——本次对比未调任何超参、未换模型、未改特征集、未改回测窗口数，只切换了 feature_vintage 一个开关；上述数字就是实验原样结果。
