"""계층 규칙을 코드로 강제한다. 문서만으로는 지켜지지 않는다.

v0.5 의 의존 방향은 하나다.

    experiments -> solvers.mappo -> solvers.milp -> theory -> models(도메인)

``solvers.mappo`` 가 ``solvers.milp`` 를 읽는 것은 **의도된** 것이다 — MAPPO 가
해석하는 대상이 milp 가 만든 Stone 인스턴스다. 반대 방향은 금지다. milp 가
mappo 를 읽기 시작하면 "계획법이 학습을 안다"가 되어, 두 계획법을
같은 인스턴스로 비교한다는 주장이 무너진다.

2026-09-16 구조 개편: ``research/`` 한 단을 없애고 도메인 모형을 모았다.
2026-10-28 구조 개편: core→models, planning→solvers/milp, learning→solvers/mappo,
chapters→experiments.
2026-10-29 구조 개편: 패키지 최상위의 낱개 모듈을 전부 하위 패키지로 —
aoi·truth→models, config·options→common, kpi·reliability·figures→reporting,
runner→experiments. 최상위에는 ``__init__``·``__main__`` 만 남는다.
계층 규칙 자체는 그대로다 (내부 계층명 planning/learning 은 유지).
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "cpp_search"

#: 각 계층이 import 해도 되는 상위 계층 (자기 계층과 아래는 항상 허용).
LAYER_ORDER = ("domain", "theory", "planning", "learning", "chapters")

#: ``python -m cpp_search`` 진입점. 챕터를 디스패치해야 하므로 러너(chapters)를 읽는다.
#: 계층 규칙의 유일한 예외이고, 예외인 이유를 여기 적어 둔다.
ENTRY_POINT_MODULES = {"__main__"}


def _layer(module: str) -> str:
    parts = module.split(".")
    head = parts[0]
    if head == "models":
        return "domain"
    if head == "solvers":
        sub = parts[1] if len(parts) > 1 else ""
        if sub == "milp":
            return "planning"
        if sub == "mappo":
            return "learning"
        return "support"  # solvers/__init__
    if head == "theory":
        return head
    if head in {"experiments", "pipeline"}:
        return "chapters"  # 계층 라벨은 유지 — 둘 다 최상단 실행 계층
    # common(설정·옵션)·reporting(지표·그림)과 최상위 __init__/__main__ —
    # 사다리에 놓지 않고 따로 다룬다.
    return "support"


def _internal_imports(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module.split(".")
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package[: -node.level] if node.level > 1 else package[:-1]
            target = ".".join(base + ([node.module] if node.module else []))
            found.add(target)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # 절대 import. 2026-09-16 구조 개편에서 상대를 전부 절대로 바꿨는데
            # 이 가지가 없어서 그래프가 통째로 비었고, 계층 검사가 아무것도
            # 검사하지 않으면서 '통과'했다.
            if node.module.startswith("cpp_search"):
                found.add(node.module.removeprefix("cpp_search."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("cpp_search"):
                    found.add(alias.name.removeprefix("cpp_search."))
    return found


def _modules() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        module = str(path.relative_to(SOURCE_ROOT))[:-3].replace("/", ".")
        graph[module] = _internal_imports(path, module)
    return graph


class LayerRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = _modules()

    def test_the_domain_layer_never_imports_research(self) -> None:
        for module, imports in self.graph.items():
            if _layer(module) != "domain" or module in ENTRY_POINT_MODULES:
                continue
            offending = sorted(
                name
                for name in imports
                if _layer(name) in {"theory", "planning", "learning", "chapters"}
            )
            self.assertEqual(
                offending,
                [],
                f"{module} imports an upper layer: {offending}",
            )

    def test_planning_never_imports_learning_or_experiments(self) -> None:
        for module, imports in self.graph.items():
            if _layer(module) != "planning":
                continue
            offending = sorted(
                name
                for name in imports
                if _layer(name) in {"learning", "chapters"}
            )
            self.assertEqual(
                offending,
                [],
                f"{module} must not know about learning or chapters: {offending}",
            )

    def test_theory_only_looks_downward(self) -> None:
        for module, imports in self.graph.items():
            if _layer(module) != "theory":
                continue
            offending = sorted(
                name
                for name in imports
                if _layer(name) in {"planning", "learning", "chapters"}
            )
            self.assertEqual(offending, [], f"{module} looks upward: {offending}")

    def test_learning_never_imports_experiments(self) -> None:
        for module, imports in self.graph.items():
            if _layer(module) != "learning":
                continue
            offending = sorted(
                name for name in imports if _layer(name) == "chapters"
            )
            self.assertEqual(offending, [], f"{module} looks upward: {offending}")

    def test_learning_is_allowed_to_read_planning(self) -> None:
        """이 규칙은 금지가 아니라 **확인**이다.

        MAPPO 가 Stone 인스턴스를 읽지 않으면 "같은 문제를 푼다"는 주장이
        성립하지 않는다. 그래서 이 import 가 사라지면 테스트가 실패해야 한다.
        """

        self.assertIn(
            "solvers.milp.stone_spx",
            self.graph["solvers.mappo.spx_env"],
        )
        self.assertIn(
            "solvers.milp.stone_spx",
            self.graph["solvers.mappo.spx_policy"],
        )

    def test_support_modules_do_not_depend_on_chapters(self) -> None:
        for module, imports in self.graph.items():
            if _layer(module) != "support" or module in ENTRY_POINT_MODULES:
                continue
            offending = sorted(
                name for name in imports if _layer(name) == "chapters"
            )
            self.assertEqual(
                offending,
                [],
                f"{module} is shared support code and must not import chapters: "
                f"{offending}",
            )


if __name__ == "__main__":
    unittest.main()
