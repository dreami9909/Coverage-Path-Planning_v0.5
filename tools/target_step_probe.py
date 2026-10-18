"""표적을 **어떤 시간 간격으로 전개하는가**가 모형 밖 MC 값을 얼마나 바꾸는가.

무엇이 문제인가
---------------
셀 모형은 표적 입자를 슬라이스마다 ``predict(슬라이스)`` 로 전개해 전이행렬을 만든다.
운동모형이 그 안에서 적분 간격(30 초)씩 끊어 전진한다. 모형 밖 MC 는 같은 운동모형을
약 5 초마다 불렀다. 운동모형에는 **한 번 부를 때마다** 적용되는 항이 있어서 — 은폐
지형이 정지 모드로 끌어당기는 확률, 선회율의 감쇠 — 두 전개는 같은 과정이 아니다.
잘게 부른 쪽에서 정지가 늘고 순변위가 15~20 % 달라진다.

그러면 평가기는 계획이 상대한 표적과 **다른 표적**으로 채점한 것이 된다. 이 도구는
같은 인증 계획을 두 전개로 채점해 그 크기를 재고, 셀 모형과 같은 간격으로 전개했을 때
표적의 셀 점유가 셀 모형과 일치하는지도 확인한다.

실행:  python3 tools/target_step_probe.py [Tank-8 ...]     MC_PARTICLES (기본 40000)
저장:  results/studies/target-step-{케이스}-{N}.json
"""
import json, os, pathlib, sys
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.chapters.ch2_stone_spx import (
    case_specs, case_target_specs, case_mission, cue_prior)
from cpp_search.chapters.ch2_studies import published_instance
from cpp_search.planning.continuous_mc import (
    SUPPORT_HALF_M, evaluate, particle_trace, _cell_index)
from cpp_search.planning.terrain_belief import BELIEF_TERRAIN_FULL
from cpp_search.theory import transitions as transition_ops

from conditions import available_conditions                 # noqa: E402

SEED, DT_S = 20260201, 5.0
MC_PARTICLES = int(os.environ.get("MC_PARTICLES", 40_000))


def occupancy_gap(instance, targets, *, mission, prior, delay, stepping, cache):
    """표적의 셀 점유가 셀 모형과 얼마나 다른가 — (총변동거리 최대, 표본만의 몫)."""
    slice_s = float(instance.slice_s)
    sub = max(1, int(round(slice_s / DT_S)))
    steps = instance.time_slice_count * sub
    cells = instance.cell_count
    rng = np.random.default_rng(3)
    worst, noise_at_worst = 0.0, 0.0
    for k, target in enumerate(targets):
        weights, positions, escaped = particle_trace(
            instance, target, k, mission=mission, prior=prior,
            terrain_coupling=BELIEF_TERRAIN_FULL, particle_count=MC_PARTICLES, seed=SEED,
            steps=steps, dt=slice_s / sub, belief_delay_s=delay, cache_dir=cache,
            stepping=stepping, slice_s=slice_s)
        model = instance.target_models[k]
        mass = np.asarray(model.initial_mass, dtype=float).copy()
        chain = transition_ops.as_sequence(model.transitions)
        for t in range(instance.time_slice_count):
            index = np.where(escaped[t * sub], cells,
                             _cell_index(instance.grid, positions[t * sub, :, 0], positions[t * sub, :, 1]))
            occupancy = np.bincount(index, weights=weights, minlength=cells + 1)
            tv = 0.5 * float(np.abs(occupancy - mass).sum())
            a = rng.multinomial(5000, occupancy / occupancy.sum()) / 5000
            b = rng.multinomial(MC_PARTICLES, occupancy / occupancy.sum()) / MC_PARTICLES
            if tv > worst:
                worst, noise_at_worst = tv, 0.5 * float(np.abs(a - b).sum())
            if t < instance.time_slice_count - 1:
                mass = mass @ chain[t]
    return worst, noise_at_worst


def main(argv: list[str]) -> None:
    wanted = set(argv) or {"Tank-8"}
    cfg = load_chapter_config("2")
    delay = float(cfg.get("stone_spx.initial_belief_delay_s", 0.0))
    cache = os.environ.get("CPP_SEARCH_TRACE_CACHE") or None
    for case, n, plan_file in available_conditions():
        tag = f"{case}-{n}"
        if tag not in wanted:
            continue
        block = next(c for c in case_specs(cfg) if c["name"] == case)
        targets = case_target_specs(cfg, block)
        mission = case_mission(cfg.mission(), block)
        instance = published_instance(cfg, case=block, width=n, seed=SEED)
        paths = tuple(tuple(int(c) for c in p)
                      for p in json.load(open(ROOT / "results/studies" / plan_file))["assignments"])
        prior = cue_prior(cfg, mission.search_radius_m)[0]
        print(f"\n[{case} {n}×{n}]  입자 {MC_PARTICLES:,}, 슬라이스 {instance.slice_s:.1f} s", flush=True)
        block_out = {"case": case, "n": n, "particles": MC_PARTICLES, "rows": []}
        for stepping, label in (("fine", "5 초마다 전개 (지금까지)"), ("model", "셀 모형과 같은 간격")):
            started = perf_counter()
            exact = evaluate(instance, targets,
                             [("최근접 진입", paths, SUPPORT_HALF_M), ("중심 경유", paths, SUPPORT_HALF_M)],
                             mission=mission, prior=prior, terrain_coupling=BELIEF_TERRAIN_FULL,
                             particle_count=MC_PARTICLES, dt_s=DT_S, seed=SEED, belief_delay_s=delay,
                             entry=("nearest", "centre-sync"), target_stepping=stepping,
                             trace_cache=cache, progress=lambda *_: None)
            pd = np.array([t["pd"] for t in exact["targets"]])      # [표적, 규약]
            tv, noise = occupancy_gap(instance, targets, mission=mission, prior=prior,
                                      delay=delay, stepping=stepping,
                                      cache=cache)
            names = [t["target"] for t in exact["targets"]]
            cell = exact["cell_model_pd_per_target"][0]
            print(f"  {label:<24} 최근접 " + " ".join(f"{v:.4f}" for v in pd[:, 0])
                  + " | 중심 " + " ".join(f"{v:.4f}" for v in pd[:, 1])
                  + f" | 셀 점유 총변동거리 {tv:.3f} (표본만의 몫 {noise:.3f}) | "
                  f"{perf_counter() - started:.0f} s", flush=True)
            block_out["rows"].append({
                "stepping": stepping, "targets": names, "cell_model_pd_per_target": cell,
                "pd_nearest_per_target": pd[:, 0].tolist(), "pd_centre_per_target": pd[:, 1].tolist(),
                "occupancy_tv_max": tv, "occupancy_tv_sampling": noise})
        a, b = block_out["rows"]
        for key, name in (("pd_nearest_per_target", "최근접 진입"), ("pd_centre_per_target", "중심 경유")):
            diff = np.array(b[key]) - np.array(a[key])
            print(f"  {name}: 전개를 바꾼 차 " + " ".join(f"{v:+.4f}" for v in diff)
                  + f" | 최저 PD {min(a[key]):.4f} -> {min(b[key]):.4f} | 셀 모형 대비 비 "
                  + " ".join(f"{m / c:.3f}" for m, c in zip(b[key], b["cell_model_pd_per_target"])))
        print("  거동: " + ", ".join(a["targets"]))
        dst = ROOT / f"results/studies/target-step-{tag}.json"
        json.dump(block_out, open(dst, "w"), ensure_ascii=False, indent=2)
        print("  저장:", dst.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
