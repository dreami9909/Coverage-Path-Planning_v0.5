"""조건 하나를 절단평면으로 풀어 **anchor 형식으로 저장**한다.

``results/studies/anchor-*.json`` 은 원래 임시 스크립트로 만들어졌다. 새 조건을
추가할 때마다 그 스크립트를 다시 쓰면 조건이 갈리므로, 인스턴스 생성부터 저장까지
한 곳에 둔다. 인스턴스는 ``ch2_studies.published_instance`` 를 지나므로 표 1·2 의
다른 조건들과 같은 방식으로 만들어진다.

저장하는 것은 계획(``assignments``)과 인증에 필요한 값 전부다 — 쌍대 하한, 상계,
간격, 반복 수, 그리고 같은 인스턴스임을 증명할 ``fingerprint``.

실행:  python3 tools/runs/solve_plan.py TEL 12            (기본 예산 master 300 s x 20)
       SOLVE_SEED (기본 20260201), MASTER_S, MAX_ITERATIONS, OUT (파일 접두어)
"""
import json, os, pathlib, sys
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cpp_search.common.config import load_chapter_config
from cpp_search.experiments.ch2_stone_spx import case_specs
from cpp_search.experiments.ch2_studies import published_instance
from cpp_search.solvers.milp.stone_spx import solve_stone_grid_paths

SEED = int(os.environ.get("SOLVE_SEED", 20260201))
MASTER_S = float(os.environ.get("MASTER_S", 300.0))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 20))


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit("사용법: python3 tools/runs/solve_plan.py <케이스> <격자 한 변>")
    case, width = str(argv[0]), int(argv[1])
    cfg = load_chapter_config("2")
    block = next(c for c in case_specs(cfg) if c["name"] == case)
    started = perf_counter()
    instance = published_instance(
        cfg, case=block, width=width, seed=SEED,
        grid_overrides={"master_time_limit_s": MASTER_S,
                        "max_iterations": MAX_ITERATIONS})
    build_s = perf_counter() - started
    print(f"[{case} {width}×{width}]  셀 {instance.cell_count}, 단계 "
          f"{instance.time_slice_count}, master {MASTER_S:.0f} s × {MAX_ITERATIONS} 회, "
          f"인스턴스 {build_s:.0f} s", flush=True)

    solution = solve_stone_grid_paths(instance, method="stone-spx")
    worst = float(min(solution.target_detection_probabilities))
    gap = float(solution.relative_optimality_gap)
    print(f"  PD {worst:.4f} | 하한 {solution.lower_bound_nondetection:.6f} | "
          f"상계 {solution.upper_bound_nondetection:.6f} | 간격 {gap*100:.3f} % | "
          f"반복 {solution.iterations} | {perf_counter() - started:.0f} s", flush=True)

    prefix = os.environ.get("OUT", "anchor")
    dst = ROOT / f"results/studies/{prefix}-{case}-{width}-{SEED}.json"
    json.dump({
        "case": case, "n": width, "seed": SEED,
        "cells": int(instance.cell_count),
        "time_slices": int(instance.time_slice_count),
        "fingerprint": instance.fingerprint,
        "budget": {"master_time_limit_s": MASTER_S, "max_iterations": MAX_ITERATIONS},
        "start_cells": [int(c) for c in instance.starts],
        "worst_pd": worst,
        "per_target": {name.split("/")[-1]: float(value) for name, value
                       in zip(instance.target_names,
                              solution.target_detection_probabilities)},
        "lower_bound_nondetection": float(solution.lower_bound_nondetection),
        "upper_bound_nondetection": float(solution.upper_bound_nondetection),
        "gap": gap,
        "certified": bool(solution.converged),
        "iterations": int(solution.iterations),
        "incumbent_source": solution.incumbent_source,
        "warm_start_nondetection": (None if solution.warm_start_nondetection is None
                                    else float(solution.warm_start_nondetection)),
        "status_history": list(solution.master_status_history or []),
        "bound_history": [list(row) for row in (solution.bound_history or [])],
        "assignments": [list(map(int, path)) for path in solution.paths],
        "build_s": build_s,
        "runtime_s": float(solution.runtime_s),
    }, open(dst, "w"), ensure_ascii=False, indent=2)
    print("  저장:", dst.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
