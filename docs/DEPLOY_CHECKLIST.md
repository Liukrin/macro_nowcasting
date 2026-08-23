# Streamlit Community Cloud 部署操作清单

> ⚠️ 每次 push 前先跑 `python scripts/precheck.py`（脚本位于
> `macro_nowcasting/sc_macro_agent_project/scripts/`，任意目录执行均可）

> 目标仓库：`Liukrin/macro_nowcasting`（master 分支）
> 入口文件：`macro_nowcasting/sc_macro_agent_project/app.py`
> 生成时间：2026-08-23（本地模拟云端冷启动排查后）

---

## 0. 部署前须知（本地已处理，无需网页操作）

以下问题已在本地修复并随本次提交进入 master，Cloud 端无需额外处理：

1. **入口脚本语法错误已修复**：`app.py` 的 TSLM 参考结果渲染里，
   `f'{r[\"rmse\"]:.4f}'` 这种 f-string 表达式内带反斜杠转义引号的写法
   （4 处）在 Python 解析期直接 `SyntaxError`，应用无法启动。已改为
   `f'{r["rmse"]:.4f}'`。
2. **LLM Key 读取已加固**：`sc_macro_agent/llm/client.py` 新增
   `_resolve_api_key()`，优先级「环境变量 > st.secrets > mock」，本地
   无 `secrets.toml` 时安全兜底、不抛异常。
3. **`.idea/` 已移出版本控制**并加入 `.gitignore`。

---

## 1. 删除现有 App（关键：Python 版本无法事后修改）

- 打开 https://share.streamlit.io/，找到现有 App。
- 进入 App 设置 → **Delete app**（删除后重建，因为已选的 Python 版本
  一旦部署就无法修改；若之前误选了 3.14，只能删除重建）。
- 确认删除。

---

## 2. 重新 Deploy

在 share.streamlit.io 点击 **New app / Deploy**，填写：

| 项 | 值 |
|---|---|
| Repository | `Liukrin/macro_nowcasting` |
| Branch | `master` |
| Main file path | `macro_nowcasting/sc_macro_agent_project/app.py` |

> Main file path 必须是嵌套路径。若只填 `app.py` 会报
> 「Main module not found」，因为入口不在仓库根目录。

---

## 3. Advanced settings → Python 版本

- Deploy 页面展开 **Advanced settings…**。
- **Python version 选 3.12**（不要选 3.14 / 3.13）：
  - 3.14 下 `pandas` / `numpy` / `scipy` / `statsmodels` 尚无预编译 wheel，
    `pip install` 必然失败（Cloud 的依赖安装阶段直接红字报错）。
- 其余保持默认。

---

## 4. Secrets 填写（TOML 格式，用占位符，勿填真实 Key 到仓库）

App 内 **Settings → Secrets** 填入：

```toml
DEEPSEEK_API_KEY = "sk-xxxx"
```

> - 用你自己的真实 Key 替换 `sk-xxxx`，**只写在 Cloud 的 Secrets 面板**，
>   绝不写进代码或提交到仓库。
> - Cloud 会把该 Secret 同时注入为环境变量与 `st.secrets`；应用已同时
>   兼容两种读取路径，任一可用即可。

---

## 5. 部署后查看日志

- App 页面右上角 **⋮（More）→ Manage app → ⋮ → 查看 Terminal / Logs**。
- 首次部署会经历：克隆仓库 → 安装依赖（`requirements.txt`）→ 启动
  `streamlit run`。观察终端即可定位卡在哪一步。

---

## 6. 三类常见报错及对应含义

| 报错现象（终端/日志关键字） | 含义 | 常见处置 |
|---|---|---|
| **依赖安装失败**：`No matching distribution found` / `Could not find a version` / `Building wheel ... failed` | pip 解析不到满足版本的包，通常是 Python 版本过新（3.14）导致 numpy/pandas 无 wheel，或网络受限 | 回 3.1 把 Python 版本改为 **3.12** 后重建 App |
| **Main module not found** / `FileNotFoundError: app.py` | Main file path 填错，入口文件不在仓库根目录 | 改为 `macro_nowcasting/sc_macro_agent_project/app.py`（见 2） |
| **运行时异常**：`Traceback (most recent call last)` 且出现在 `sc_macro_agent/...` | 依赖已装好、入口已找到，但脚本执行期抛异常（如数据缺失、import 失败） | 看 Traceback 最后一行定位；常见如缺 `statsmodels`（确认根目录 `requirements.txt` 未被删改） |

---

## 7. 冷启动预期行为（无 torch / 无 Key 时的降级表现）

- 无 `torch` / `chronos`：TSLM 残差修正进入 `failed` 态，前端显示
  「当前部署环境未启用（无 torch）」，主预测流程不受影响。
- 未配 `DEEPSEEK_API_KEY`：LLM 走 **mock** 占位，简报页/问答页可正常打开，
  页面顶部有「降级模式」提示。
- 无 `artifacts/` 历史产物：数据质量、简报、RAG 问答页均安全降级
  （读取历史产物处有内置占位文案），不崩溃。

> 以上三点已在本地通过模拟云端约束（无 torch stub 拦截、无 Key、无 HF 缓存、
> 无 artifacts、根目录启动）实测通过。
