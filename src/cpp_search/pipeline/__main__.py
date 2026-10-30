"""CLI — ``python -m cpp_search.pipeline <scenario.toml> [--out results/runs]``."""

from __future__ import annotations

import argparse
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
    _print_summary(metrics, out_dir)
    return 0


def _behaviour_token(name: str) -> str:
    """모형 표적 이름('Tank-nominal/STOP_HEAVY')과 가설 라벨
    ('STOP_HEAVY@(1200,-900)')을 같은 거동 토큰으로 접는다."""

    token = name.split("/", 1)[-1].split("@", 1)[0]
    return "custom" if token.startswith("CUSTOM(") else token


def _print_summary(metrics: dict, out_dir: Path) -> None:
    """프로파일별 표 한 장 + 공통 줄. 전체 기록은 metrics.json 에 있다.

    optimality gap 과 AOI 커버리지는 프로파일별 값이 아니다 — gap 은 minimax
    계획 전체의 인증폭이고, 커버리지는 경로 기하라 모든 가설에 같다. 그래서
    표가 아니라 공통 줄에 적는다.
    """

    solver = metrics["solver"]
    flight = metrics["flight"]
    model_by_target: dict[str, float] = solver["model_detection_probability_by_target"]
    model_by_token = {_behaviour_token(k): v for k, v in model_by_target.items()}

    rows = []
    for label, block in flight["by_target"].items():
        model_pd = model_by_target.get(label)
        if model_pd is None:                 # 명시 가설 — 거동 토큰으로 대응
            model_pd = model_by_token.get(_behaviour_token(label))
        mc_pd = float(block["detection_probability"])
        time_s = block.get("conditional_mean_detection_time_s")
        rows.append((label, model_pd, mc_pd, float(block["failure_rate"]), time_s))

    width = max(len("profile"), *(len(r[0]) for r in rows))

    def fmt(value, pattern: str, blank: str = "      —") -> str:
        return pattern % value if value is not None else blank

    print(f"{'profile':<{width}}  model_PD   MC_PD  model-MC  fail_rate  t_detect[s]")
    for label, model_pd, mc_pd, failure, time_s in rows:
        delta = (model_pd - mc_pd) if model_pd is not None else None
        print(
            f"{label:<{width}}  {fmt(model_pd, '%8.4f')}  {mc_pd:6.4f}"
            f"  {fmt(delta, '%+8.4f')}  {failure:9.4f}  {fmt(time_s, '%11.1f')}"
        )

    blocks = list(flight["by_target"].values())
    coverage = next(
        (b["unique_area_coverage_ratio"] for b in blocks
         if b.get("unique_area_coverage_ratio") is not None),
        None,
    )
    detection_times = [
        b["conditional_mean_detection_time_s"] for b in blocks
        if b.get("conditional_mean_detection_time_s") is not None
    ]
    gap = solver["relative_optimality_gap"]
    print(f"\nrun_dir: {out_dir}")
    print(
        f"공통  optimality gap {gap:.3%} · 인증 {'O' if solver['certified'] else 'X'}"
        + (f" · AOI 커버리지 {100.0 * coverage:.1f}%" if coverage is not None else "")
    )
    mean_time = (
        f"{sum(detection_times) / len(detection_times):.0f} s" if detection_times else "—"
    )
    print(
        f"요약  flown 평균 {flight['detection_probability_mean']:.4f}"
        f" (실패율 {1.0 - flight['detection_probability_mean']:.4f})"
        f" · 평균 탐지시간 {mean_time}"
        f" · model−flown {flight['model_minus_flown']:+.4f}"
    )
    print(
        f"최악  model: {solver['model_detection_probability_worst_target']}"
        f" ({solver['model_detection_probability_worst']:.4f})"
        f" · flown: {flight['detection_probability_worst_target']}"
        f" ({flight['detection_probability_worst']:.4f})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
