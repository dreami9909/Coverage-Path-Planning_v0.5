"""이산화의 비가 **표적 운동모형에 달린 것인지** 본다 (본문 4.8절).

같은 계획(Tank 8 × 8 의 인증 계획)을 두 운동모형 위에서 각각 세운다.

    imm5        본 논문의 표적 모형. HALT·저속CV·고속CV·CTRV·기동 다섯 모드
    isotropic   위치만 가진 무작위 방향 모형 (v0.3 의 모형)

운동모형을 바꾸면 전이행렬이 바뀌므로 **셀 모형 값도 같이 바뀐다.** 그래서 비교는
절대값이 아니라 **비(연속/셀)** 로 한다. 비가 두 모형에서 같으면, 그 비는 표적이
어떻게 움직이는가가 아니라 셀 이산화 자체에서 오는 것이다.

계획은 새로 풀지 않는다 — 저장된 인증 계획을 두 인스턴스에서 그대로 채점한다.
운동모형이 다르면 그 계획이 그 인스턴스의 최적해는 아니지만, 여기서 묻는 것은
최적성이 아니라 **두 평가의 비**다.

실행:  python3 tools/iso_probe.py      MC_PARTICLES (기본 80000)
"""
import json, os, pathlib, sys
from dataclasses import replace
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpp_search.config import load_chapter_config
from cpp_search.experiments.ch2_stone_spx import (
    StoneSPXTargetSpec, case_specs, case_target_specs, case_mission, cue_prior)
from cpp_search.experiments.ch2_studies import published_instance
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, evaluate
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL

SEED, DT_S, CASE, WIDTH = 20260201, 5.0, "Tank", 8
PLAN_FILE = "occupancy-1.json"
PARTICLES = int(os.environ.get("MC_PARTICLES", 80_000))


def main() -> None:
    cfg = load_chapter_config("2")
    block = next(c for c in case_specs(cfg) if c["name"] == CASE)
    mission = case_mission(cfg.mission(), block)
    prior = cue_prior(cfg, mission.search_radius_m)[0]
    delay = float(cfg.get("stone_spx.initial_belief_delay_s", 0.0))
    stored = json.load(open(ROOT / "results/studies" / PLAN_FILE))
    paths = tuple(tuple(int(c) for c in p) for p in stored["assignments"])
    base_targets = case_target_specs(cfg, block)

    summary = []
    for model in ("imm5", "isotropic"):
        targets = tuple(
            StoneSPXTargetSpec(replace(spec.profile, motion_model=model),
                               spec.hazard_multiplier)
            for spec in base_targets
        )
        instance = published_instance(cfg, case=block, width=WIDTH, seed=SEED,
                                      targets=targets)
        cell = instance.score_paths(paths)
        print(f"\n[{model}]  셀 모형 최저 {min(cell):.4f}, 입자 {PARTICLES:,}", flush=True)
        started = perf_counter()
        exact = evaluate(instance, targets, [("plan", paths, SUPPORT_HALF_M)],
                         mission=mission, prior=prior,
                         terrain_coupling=BELIEF_TERRAIN_FULL,
                         particle_count=PARTICLES, dt_s=DT_S, seed=SEED,
                         belief_delay_s=delay, progress=lambda *_: None)
        worst_cell = float(min(cell))
        worst_mc = min(float(t["pd"][0]) for t in exact["targets"])
        print(f"  평가기 {perf_counter() - started:.0f} s | 연속 최저 {worst_mc:.4f} | "
              f"비 {worst_mc / worst_cell:.3f}", flush=True)
        for t, value in zip(exact["targets"], cell):
            print(f"   {t['target']:<17} 셀 {value:.4f} | 연속 {t['pd'][0]:.4f} | "
                  f"비 {t['pd'][0] / value:.3f}")
        result = dict(exact)
        result.update({"case": CASE, "n": WIDTH, "seed": SEED, "motion_model": model,
                       "plan_file": PLAN_FILE, "fingerprint": instance.fingerprint})
        dst = ROOT / f"results/studies/mc2-iso-{model}.json"
        json.dump(result, open(dst, "w"), ensure_ascii=False, indent=2)
        summary.append((model, worst_cell, worst_mc, worst_mc / worst_cell))
        print("  저장:", dst.name, flush=True)

    print(f"\n{'운동모형':<12}{'셀 모형':>9}{'연속':>9}{'비':>8}")
    for model, cell, mc, ratio in summary:
        print(f"{model:<12}{cell:>9.4f}{mc:>9.4f}{ratio:>8.3f}")
    print(f"비의 차 {abs(summary[0][3] - summary[1][3]):.3f}")


if __name__ == "__main__":
    main()
