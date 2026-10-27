"""CLI — ``python -m cpp_search.pipeline <scenario.toml> [--out results/runs]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cpp_search.pipeline.run import run_scenario
from cpp_search.pipeline.scenario import ScenarioError, load_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cpp_search.pipeline",
        description="시나리오 TOML 하나를 Input→Model→Solver→Output 으로 실행한다",
    )
    parser.add_argument("scenario", help="시나리오 TOML 경로")
    parser.add_argument(
        "--out", default="results/runs", help="실행 디렉터리 루트 (기본 results/runs)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="입력 검증까지만 하고 실행하지 않는다",
    )
    args = parser.parse_args(argv)

    try:
        scenario = load_scenario(args.scenario)
    except ScenarioError as error:
        print(f"시나리오 거부: {error}", file=sys.stderr)
        return 2
    for warning in scenario.warnings:
        print(f"경고: {warning}", file=sys.stderr)
    if args.validate_only:
        print(f"OK — {scenario.name} (케이스 {scenario.target.case}, "
              f"{scenario.grid.kind} {scenario.grid.n}×{scenario.grid.n})")
        return 0

    metrics = run_scenario(scenario, out_root=args.out)
    out_dir = Path(args.out) / scenario.name
    solver = metrics["solver"]
    flight = metrics["flight"]
    print(json.dumps(
        {
            "run_dir": str(out_dir),
            "model_PD_worst": round(solver["model_detection_probability_worst"], 4),
            "optimality_gap": solver["relative_optimality_gap"],
            "certified": solver["certified"],
            "flown_PD_worst": round(flight["detection_probability_worst"], 4),
            "model_minus_flown": round(flight["model_minus_flown"], 4),
        },
        ensure_ascii=False, indent=1,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
