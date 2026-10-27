"""같은 인스턴스의 계획 다섯 개를 **공통 난수로 순위** 짓는다 (4.8절, 그림 7 나).

계산 조건만 다르게 하여 얻은 Tank 10 × 10 의 계획 다섯을 모아, 셀 모형 순위와
모형 밖 MC 순위가 같은지 본다. 셀 모형이 최선이라 지목한 계획이 연속 공간에서도
최선인지가 질문이다.

계획은 저장된 연구 결과에서 그대로 가져온다 (새로 풀지 않는다).

    300x20        anchor-Tank-10-20260201.json          master 300 s x 20 회
    60x60         budget-Tank-10-60x60.json             master 60 s x 60 회
    300x60 인증   scale-Tank-10-300x60.json             master 300 s x 60 회 (인증)
    T3600 k=6     budget-alloc-…-laneA.json 의 k=6 행    총예산 3,600 s 를 6 회로
    T3600 k=20    budget-alloc-…-laneB.json 의 k=20 행   총예산 3,600 s 를 20 회로

실행:  python3 tools/rank_probe.py      MC_PARTICLES (기본 120000)
"""
import json, os, pathlib, sys
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.experiments.ch2_stone_spx import (
    case_specs, case_target_specs, case_mission, cue_prior)
from cpp_search.experiments.ch2_studies import published_instance
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, evaluate
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL

SEED, DT_S, CASE, WIDTH = 20260201, 5.0, "Tank", 10
PARTICLES = int(os.environ.get("MC_PARTICLES", 120_000))


def _paths(assignments) -> tuple:
    return tuple(tuple(int(c) for c in p) for p in assignments)


def collect_plans() -> list[tuple[str, tuple]]:
    """다섯 계획을 저장된 결과에서 모은다. 순서는 셀 모형 값과 무관하게 고정한다."""

    studies = ROOT / "results/studies"
    plans = []
    for label, name in (("T3600 k=6", "budget-alloc-Tank-10-T3600-laneA.json"),
                        ("T3600 k=20", "budget-alloc-Tank-10-T3600-laneB.json")):
        rows = json.load(open(studies / name))["rows"]
        target = 6 if "k=6" in label else 20
        row = next(r for r in rows if int(r["iterations"]) == target)
        plans.append((label, _paths(row["assignments"])))
    for label, name in (("300x20", "anchor-Tank-10-20260201.json"),
                        ("60x60", "budget-Tank-10-60x60.json"),
                        ("300x60 인증", "scale-Tank-10-300x60.json")):
        plans.append((label, _paths(json.load(open(studies / name))["assignments"])))
    return plans


def main() -> None:
    cfg = load_chapter_config("2")
    block = next(c for c in case_specs(cfg) if c["name"] == CASE)
    targets = case_target_specs(cfg, block)
    mission = case_mission(cfg.mission(), block)
    instance = published_instance(cfg, case=block, width=WIDTH, seed=SEED)
    prior = cue_prior(cfg, mission.search_radius_m)[0]
    delay = float(cfg.get("stone_spx.initial_belief_delay_s", 0.0))
    plans = collect_plans()

    cell = [min(instance.score_paths(paths)) for _, paths in plans]
    print(f"[{CASE} {WIDTH}×{WIDTH}]  계획 {len(plans)}개, 입자 {PARTICLES:,}", flush=True)
    started = perf_counter()
    exact = evaluate(instance, targets,
                     [(name, paths, SUPPORT_HALF_M) for name, paths in plans],
                     mission=mission, prior=prior, terrain_coupling=BELIEF_TERRAIN_FULL,
                     particle_count=PARTICLES, dt_s=DT_S, seed=SEED,
                     belief_delay_s=delay, progress=lambda *_: None)
    print(f"  평가기 {perf_counter() - started:.0f} s", flush=True)

    pd = np.array([t["pd"] for t in exact["targets"]])          # [표적, 계획]
    worst = pd.min(axis=0)
    cell_rank = np.argsort(np.argsort(-np.array(cell))) + 1
    mc_rank = np.argsort(np.argsort(-worst)) + 1
    print(f"  {'계획':<14}{'셀 모형':>9}{'순위':>5}{'연속':>9}{'순위':>5}")
    for i, (name, _) in enumerate(plans):
        print(f"  {name:<14}{cell[i]:>9.4f}{cell_rank[i]:>5}{worst[i]:>9.4f}{mc_rank[i]:>5}",
              flush=True)
    spearman = float(np.corrcoef(cell_rank, mc_rank)[0, 1])
    print(f"  순위상관 {spearman:+.3f} | 셀 최선의 연속 순위 {mc_rank[int(np.argmax(cell))]} "
          f"| 연속 최선의 셀 순위 {cell_rank[int(np.argmax(worst))]}")

    dst = ROOT / "results/studies/mc2-rank-Tank-10.json"
    result = dict(exact)
    result.update({"case": CASE, "n": WIDTH, "seed": SEED,
                   "plans": [name for name, _ in plans],
                   "cell_model_pd": [float(v) for v in cell],
                   "rank_spearman": spearman})
    json.dump(result, open(dst, "w"), ensure_ascii=False, indent=2)
    print("  저장:", dst.name)


if __name__ == "__main__":
    main()
