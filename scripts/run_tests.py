#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按范围运行测试，避免改前端却跑全量引擎。

用法:
  engine\\.venv\\Scripts\\python.exe scripts\\run_tests.py
  engine\\.venv\\Scripts\\python.exe scripts\\run_tests.py auto
  engine\\.venv\\Scripts\\python.exe scripts\\run_tests.py frontend
  engine\\.venv\\Scripts\\python.exe scripts\\run_tests.py engine-fast
  engine\\.venv\\Scripts\\python.exe scripts\\run_tests.py engine-api
  engine\\.venv\\Scripts\\python.exe scripts\\run_tests.py engine-core
  engine\\.venv\\Scripts\\python.exe scripts\\run_tests.py engine
  engine\\.venv\\Scripts\\python.exe scripts\\run_tests.py all
  engine\\.venv\\Scripts\\python.exe scripts\\run_tests.py --list
  engine\\.venv\\Scripts\\python.exe scripts\\run_tests.py auto --base origin/main
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from test_suites import (  # noqa: E402
    PY, ROOT as SUITE_ROOT, SUITE_HELP, detect_suites_from_paths, paths_for_suite,
)

assert ROOT == SUITE_ROOT


def _python() -> str:
    if PY.exists():
        return str(PY)
    return sys.executable


def git_changed_files(base: str | None) -> list[str]:
    """工作区 + 暂存 + 相对 base 的提交改动。"""
    files: set[str] = set()
    cmds = [
        ["git", "diff", "--name-only", "--diff-filter=ACMR"],
        ["git", "diff", "--name-only", "--cached", "--diff-filter=ACMR"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]
    if base:
        cmds.append(["git", "diff", "--name-only", f"{base}...HEAD", "--diff-filter=ACMR"])
    else:
        # 相对上游或 HEAD~1（若存在）
        upstream = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if upstream.returncode == 0 and upstream.stdout.strip():
            cmds.append(
                ["git", "diff", "--name-only", "@{u}...HEAD", "--diff-filter=ACMR"]
            )

    for cmd in cmds:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            continue
        for line in r.stdout.splitlines():
            line = line.strip()
            if line:
                files.add(line.replace("\\", "/"))
    return sorted(files)


def run_frontend() -> int:
    print("\n>>> [frontend] node frontend/tests/run.mjs", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(
        ["node", str(ROOT / "frontend" / "tests" / "run.mjs")],
        cwd=ROOT,
    )
    dt = time.perf_counter() - t0
    print(f"<<< [frontend] exit={r.returncode}  ({dt:.1f}s)", flush=True)
    return r.returncode


def run_pytest(paths: list[Path], extra: list[str] | None = None) -> int:
    if not paths:
        print(">>> [engine] 无路径，跳过")
        return 0
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("缺少测试文件:", ", ".join(str(m) for m in missing))
        return 2
    rels = [str(p.relative_to(ROOT)) if p.is_absolute() else str(p) for p in paths]
    cmd = [
        _python(), "-X", "utf8", "-m", "pytest",
        *rels,
        "-q",
        "--tb=line",
    ]
    if extra:
        cmd.extend(extra)
    print("\n>>> [engine]", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    dt = time.perf_counter() - t0
    print(f"<<< [engine] exit={r.returncode}  ({dt:.1f}s)", flush=True)
    return r.returncode


def expand_suites(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        n = n.lower().replace("_", "-")
        if n in ("all", "full"):
            out.extend(["frontend", "engine"])
        else:
            out.append(n)
    # 去重保序
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def run_suites(suites: list[str], pytest_extra: list[str] | None = None) -> int:
    if not suites:
        print("无需要运行的测试套件（例如仅改了文档）。跳过。")
        return 0

    print("将运行套件:", ", ".join(suites), flush=True)
    code = 0
    for s in suites:
        if s in ("frontend", "fe", "js"):
            rc = run_frontend()
        elif s in ("engine", "engine-all", "backend"):
            rc = run_pytest(paths_for_suite("engine"), pytest_extra)
        else:
            try:
                paths = paths_for_suite(s)
            except KeyError as e:
                print(e)
                return 2
            if s.startswith("engine") or s in ("fast", "api", "core"):
                rc = run_pytest(paths, pytest_extra)
            else:
                print(f"未知套件: {s}")
                return 2
        if rc != 0:
            code = rc
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Xunlong 分范围测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=SUITE_HELP,
    )
    parser.add_argument(
        "scope",
        nargs="*",
        default=["auto"],
        help="套件名：auto / frontend / engine-fast / engine-api / engine-core / engine / all",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="auto 模式下 git diff 的基线（如 origin/main）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出套件说明并退出",
    )
    parser.add_argument(
        "--print-files",
        action="store_true",
        help="auto 时打印检测到的全部改动文件",
    )
    # 用 parse_known_args：未知参数（及 -- 之后）原样传给 pytest，避免可选旗标被 REMAINDER 吃掉
    args, unknown = parser.parse_known_args(argv)

    if args.list:
        print(SUITE_HELP)
        return 0

    scopes = [s for s in args.scope if s != "--"]
    if not scopes:
        scopes = ["auto"]

    pytest_extra = list(unknown)
    if pytest_extra and pytest_extra[0] == "--":
        pytest_extra = pytest_extra[1:]

    if len(scopes) == 1 and scopes[0].lower() in ("auto", "detect"):
        changed = git_changed_files(args.base)
        print(f"检测到 {len(changed)} 个改动文件"
              + (f"（base={args.base}）" if args.base else ""))
        if args.print_files:
            for f in changed[:80]:
                print(f"  - {f}")
            if len(changed) > 80:
                print(f"  ... 另有 {len(changed) - 80} 个")
        elif changed:
            # 默认只列前 15 个，避免刷屏
            for f in changed[:15]:
                print(f"  - {f}")
            if len(changed) > 15:
                print(f"  ... 另有 {len(changed) - 15} 个（--print-files 看全量）")
        suites = detect_suites_from_paths(changed)
        print("auto →", ", ".join(suites) if suites else "(无)")
        return run_suites(suites, pytest_extra)

    suites = expand_suites(scopes)
    return run_suites(suites, pytest_extra)


if __name__ == "__main__":
    raise SystemExit(main())
