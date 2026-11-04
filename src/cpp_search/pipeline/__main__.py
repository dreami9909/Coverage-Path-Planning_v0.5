"""CLI — ``python -m cpp_search.pipeline <scenario.toml>``.

플래그: ``--out`` 결과 루트(기본 results/runs), ``--validate-only`` 입력
검증만(해소·좌표·거동 이름까지, 실행 없음), ``--resolve`` 저장된
plan.json 을 무시하고 다시 solve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cpp_search.pipeline.run import run_scenario, validate_inputs
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
    parser.add_argument(
        "--resolve", action="store_true",
        help="저장된 계획(plan.json)을 무시하고 다시 푼다 — solve 는 실행마다 "
             "다른 계획을 낼 수 있다 (기본은 같은 입력이면 재사용)",
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
        # 로딩 한도 검사에 더해 config 해소·AOI 좌표·가설 거동 이름까지 —
        # run_scenario 가 solver 전에 할 검사를 전부 미리 돈다.
        try:
            resolved = validate_inputs(scenario)
        except ScenarioError as error:
            print(f"시나리오 거부: {error}", file=sys.stderr)
            return 2
        print(f"OK — {scenario.name} (케이스 {scenario.target.case}, "
              f"{scenario.grid.kind} {scenario.grid.n}×{scenario.grid.n}, "
              f"AOI {resolved['aoi_radius_m']:,.0f} m, "
              f"표적 {len(resolved['targets'])}종)")
        return 0

    metrics = run_scenario(
        scenario, out_root=args.out, reuse_plan=not args.resolve
    )
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
    mc = metrics["mc"]
    flightlike = mc["kind"] == "flight-hypotheses"
    model_by_target: dict[str, float] = solver["model_detection_probability_by_target"]
    model_by_token = {_behaviour_token(k): v for k, v in model_by_target.items()}

    rows = []
    for label, block in mc["by_target"].items():
        model_pd = model_by_target.get(label)
        if model_pd is None:                 # 명시 가설 — 거동 토큰으로 대응
            model_pd = model_by_token.get(_behaviour_token(label))
        mc_pd = float(block["detection_probability"])
        time_s = block.get("conditional_mean_detection_time_s")
        rows.append((label, model_pd, mc_pd, float(block["failure_rate"]), time_s))

    width = max(len("profile"), *(len(r[0]) for r in rows))

    def fmt(value, pattern: str, blank: str = "      —") -> str:
        return pattern % value if value is not None else blank

    # t_detect 는 실비행 채점에만 있다 — 규약 맞춤 연속 MC 는 시간 분해가 없다.
    tail = "  t_detect[s]" if flightlike else ""
    print(f"{'profile':<{width}}  model_PD   MC_PD  model-MC  fail_rate{tail}")
    for label, model_pd, mc_pd, failure, time_s in rows:
        delta = (model_pd - mc_pd) if model_pd is not None else None
        line = (
            f"{label:<{width}}  {fmt(model_pd, '%8.4f')}  {mc_pd:6.4f}"
            f"  {fmt(delta, '%+8.4f')}  {failure:9.4f}"
        )
        if flightlike:
            line += f"  {fmt(time_s, '%11.1f')}"
        print(line)

    coverage = metrics.get("geometry", {}).get("unique_area_coverage_ratio")
    gap = solver["relative_optimality_gap"]
    device = (
        f"실비행 (표본 {mc['monte_carlo_samples']})" if flightlike
        else f"규약맞춤 연속 (centre-sync, 입자 {mc['particles']:,})"
    )
    print(f"\nrun_dir: {out_dir}")
    print(
        f"공통  optimality gap {gap:.3%} · 인증 {'O' if solver['certified'] else 'X'}"
        + (f" · AOI 커버리지 {100.0 * coverage:.1f}%" if coverage is not None else "")
        + f" · MC 장치 {device}"
        + (" · 계획 plan.json 재사용" if solver.get("plan_reused") else "")
    )
    summary = (
        f"요약  MC 평균 {mc['detection_probability_mean']:.4f}"
        f" (실패율 {1.0 - mc['detection_probability_mean']:.4f})"
    )
    if flightlike:
        detection_times = [
            b["conditional_mean_detection_time_s"] for b in mc["by_target"].values()
            if b.get("conditional_mean_detection_time_s") is not None
        ]
        if detection_times:
            summary += (
                f" · 평균 탐지시간 {sum(detection_times) / len(detection_times):.0f} s"
            )
    summary += f" · model−MC {mc['model_minus_mc']:+.4f}"
    print(summary)
    print(
        f"최악  model: {solver['model_detection_probability_worst_target']}"
        f" ({solver['model_detection_probability_worst']:.4f})"
        f" · MC: {mc['detection_probability_worst_target']}"
        f" ({mc['detection_probability_worst']:.4f})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
