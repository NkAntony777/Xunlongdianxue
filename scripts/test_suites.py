"""测试套件定义：按改动范围选择子集，避免无脑全量。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_TESTS = ROOT / "engine" / "tests"
PY = ROOT / "engine" / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = ROOT / "engine" / ".venv" / "bin" / "python"

# 引擎快测：纯逻辑 / 轻量集成（日常改 core 公式优先）
ENGINE_FAST = [
    "test_aoi_limits.py",
    "test_water_model.py",
    "test_water_sha_influence.py",
    "test_scoring_gate.py",
    "test_cross_water_dragon.py",
    "test_traditional_audit.py",
]

# 算法核心（含较慢四象/龙脉/排序）
ENGINE_CORE = ENGINE_FAST + [
    "test_engine.py",
    "test_four_beasts_detect.py",
    "test_dragon_quality.py",
    "test_incoming_vein.py",
    "test_p1_p2_form_liqi.py",
]

# API / 渲染 / 在线位置
ENGINE_API = [
    "test_render_api.py",
    "test_location_apis.py",
    "test_aoi_limits.py",
]

# 路径前缀 → 触发的 suite 名（auto 模式）
PATH_RULES: list[tuple[str, list[str]]] = [
    ("frontend/", ["frontend"]),
    ("engine/api/", ["engine-api"]),
    ("engine/llm/", ["engine-api"]),
    ("engine/run_server.py", ["engine-api"]),
    ("engine/core/render", ["engine-api", "engine-fast"]),
    ("engine/core/rendering/", ["engine-api", "engine-fast"]),
    ("engine/core/", ["engine-fast"]),
    ("engine/io/", ["engine-fast"]),
    ("engine/pipeline/", ["engine-core"]),
    ("engine/tests/test_render", ["engine-api"]),
    ("engine/tests/test_location", ["engine-api"]),
    ("engine/tests/", ["engine-fast"]),
    ("engine/", ["engine-fast"]),
    ("scripts/", []),  # 测试编排脚本本身不强制跑套件
    ("pytest.ini", []),
    ("AGENTS.md", []),  # 文档不触发
    ("HANDOFF.md", []),
    ("research/", []),
    ("docs/", []),
    ("data/", []),
]

SUITE_HELP = """
可用套件:
  auto          根据 git 改动自动选择（默认）
  frontend      前端 Node 轻测（秒级）
  engine-fast   引擎快测（公式/水法/玄空等，约数十秒内）
  engine-api    API / 图层渲染 / 位置接口
  engine-core   核心算法（含四象/龙脉等较慢用例）
  engine        全部 engine/tests
  all           frontend + engine 全量
""".strip()


def paths_for_suite(name: str) -> list[Path]:
    """返回 pytest 路径列表；frontend 返回空（由 runner 特殊处理）。"""
    name = name.lower().replace("_", "-")
    if name in ("frontend", "fe", "js"):
        return []
    if name in ("engine-fast", "fast"):
        return [ENGINE_TESTS / f for f in dict.fromkeys(ENGINE_FAST)]
    if name in ("engine-api", "api"):
        return [ENGINE_TESTS / f for f in dict.fromkeys(ENGINE_API)]
    if name in ("engine-core", "core"):
        return [ENGINE_TESTS / f for f in dict.fromkeys(ENGINE_CORE)]
    if name in ("engine", "engine-all", "backend"):
        return [ENGINE_TESTS]
    if name in ("all", "full"):
        return [ENGINE_TESTS]
    raise KeyError(f"未知套件: {name}\n{SUITE_HELP}")


def detect_suites_from_paths(changed: list[str]) -> list[str]:
    """根据改动文件路径推断应跑的套件列表。"""
    if not changed:
        return ["engine-fast"]  # 无改动时给个轻量默认

    suites: set[str] = set()
    matched_any = False
    for raw in changed:
        p = raw.replace("\\", "/").lstrip("./")
        for prefix, suite_list in PATH_RULES:
            if p.startswith(prefix) or p == prefix.rstrip("/"):
                matched_any = True
                suites.update(suite_list)
                break
        else:
            # 未匹配规则：保守跑快测
            if p.endswith((".py", ".js", ".html", ".css")):
                matched_any = True
                if p.startswith("frontend"):
                    suites.add("frontend")
                else:
                    suites.add("engine-fast")

    if not matched_any:
        return ["engine-fast"]
    if not suites:
        return []  # 纯文档改动
    # 若已选 engine-core 或 engine，去掉被包含的 fast
    if "engine" in suites:
        suites.discard("engine-fast")
        suites.discard("engine-core")
        suites.discard("engine-api")
    elif "engine-core" in suites:
        suites.discard("engine-fast")
    return sorted(suites)
