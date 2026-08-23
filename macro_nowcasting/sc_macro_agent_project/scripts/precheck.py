"""push 前检查：编译 + 导入 + 密钥扫描。

用法（脚本与执行目录解耦，任意目录下均可）：
    python macro_nowcasting/sc_macro_agent_project/scripts/precheck.py
或在 sc_macro_agent_project/ 目录下：
    python scripts/precheck.py

三项检查，任一失败以非零码退出：
  (a) 编译检查：对 sc_macro_agent 包与 app.py / run.py / main.py 全量
      编译（等价于 py_compile，用 compile() 以拿到精确行号），
      失败打印文件与行号。
  (b) 导入检查：from sc_macro_agent import AppConfig, PredictionEngine。
  (c) 密钥扫描：仓库内被跟踪 + 未跟踪（非忽略）文件里，sk- 开头且长度
      >20 的连续串，排除 sk-test / sk-xxxx 占位符前缀。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# sc_macro_agent_project/ 目录（与脚本所在位置解耦）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 仓库根目录（git ls-files 的 CWD）
REPO_ROOT = PROJECT_ROOT.parent.parent

# 需编译的入口脚本（相对项目根）
ENTRY_SCRIPTS = ["app.py", "run.py", "main.py"]

# 密钥占位符前缀（命中后跳过，不视为真实 Key）
KEY_PLACEHOLDER_PREFIXES = ("sk-test", "sk-xxxx")
# "sk-" + 18 字符 = 总长 21（>20），匹配真实 Key（sk- + 32 位十六进制）
KEY_RE = re.compile(r"sk-[A-Za-z0-9_-]{18,}")


def _tracked_and_untracked_files() -> list[str]:
    """返回仓库内「被跟踪 + 未跟踪(非忽略)」文件的相对路径列表。"""
    files: list[str] = []
    for args in (["ls-files"], ["ls-files", "--others", "--exclude-standard"]):
        try:
            out = subprocess.check_output(
                ["git", *args], cwd=str(REPO_ROOT), text=True
            )
            files.extend(ln for ln in out.splitlines() if ln.strip())
        except Exception:
            continue
    return files


def check_compile() -> int:
    """编译 sc_macro_agent 包与入口脚本，返回失败数。"""
    targets: list[Path] = []
    pkg = PROJECT_ROOT / "sc_macro_agent"
    if pkg.is_dir():
        targets.extend(sorted(pkg.rglob("*.py")))
    for name in ENTRY_SCRIPTS:
        p = PROJECT_ROOT / name
        if p.exists():
            targets.append(p)

    failures = 0
    for p in targets:
        try:
            source = p.read_text(encoding="utf-8")
            compile(source, str(p), "exec")
        except SyntaxError as exc:
            failures += 1
            print(f"  [COMPILE FAIL] {p.relative_to(PROJECT_ROOT)}: "
                  f"line {exc.lineno} -> {exc.msg}")
        except Exception as exc:
            failures += 1
            print(f"  [COMPILE FAIL] {p.relative_to(PROJECT_ROOT)}: {exc}")
    return failures


def check_import() -> int:
    """快速导入检查，失败返回 1。"""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from sc_macro_agent import AppConfig, PredictionEngine  # noqa: F401
        return 0
    except Exception as exc:
        print(f"  [IMPORT FAIL] from sc_macro_agent import ... -> "
              f"{type(exc).__name__}: {exc}")
        return 1


def check_keys() -> int:
    """扫描仓库文件中的真实 Key 模式，返回命中数。"""
    failures = 0
    for rel in _tracked_and_untracked_files():
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in KEY_RE.finditer(text):
            token = m.group(0)
            if token.startswith(KEY_PLACEHOLDER_PREFIXES):
                continue
            failures += 1
            print(f"  [KEY] {rel}: {token}")
    return failures


def main() -> int:
    print("precheck: (a) 编译检查 ...")
    compile_fail = check_compile()
    print("precheck: (b) 导入检查 ...")
    import_fail = check_import()
    print("precheck: (c) 密钥扫描 ...")
    key_fail = check_keys()

    total = compile_fail + import_fail + key_fail
    if total:
        print(f"\nprecheck FAILED: 编译失败 {compile_fail} / "
              f"导入失败 {import_fail} / 密钥命中 {key_fail}")
        return 1
    print("\nprecheck OK: 编译 / 导入 / 密钥 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
