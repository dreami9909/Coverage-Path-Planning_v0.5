"""수치 수렴 — 시간 간격과 난수를 바꿔도 모형 밖 MC 값이 유지되는가.

모형 밖 MC 는 표적 입자를 5 초 간격으로 전개하고, 창 하나(5 초 비행)가 남기는 hazard 를
그 창이 시작할 때의 표적 위치에서 잰다. 간격이 너무 크면 그 사이 표적이 움직인 만큼
어긋난다. 간격을 절반으로 줄여 값이 유지되는지 보고, 난수 seed 를 바꿔 표본오차의
크기도 같이 잰다. 입자 수 의존은 4만(``erase-*``) 대 12만(``mc2-*``)으로 따로 본다.

실행:  python3 tools/convergence_probe.py [Tank-8 ...]     MC_PARTICLES (기본 40000)
저장:  results/studies/convergence-{케이스}-{N}.json
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
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, evaluate
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL

from conditions import available_conditions                 # noqa: E402

SEED = 20260201
MC_PARTICLES = int(os.environ.get("MC_PARTICLES", 40_000))
#: (이름, 시간 간격 [s], seed)
VARIANTS = (("기준 (5 s)", 5.0, SEED), ("간격 절반 (2.5 s)", 2.5, SEED), ("다른 난수 (5 s)", 5.0, SEED + 1))


def main(argv: list[str]) -> None:
    wanted = set(argv) or {"Tank-8"}
    cfg = load_chapter_config("2")
    delay = float(cfg.get("stone_spx.initial_belief_delay_s", 0.0))
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
        rows = []
        print(f"\n[{case} {n}×{n}]  입자 {MC_PARTICLES:,}", flush=True)
        for name, dt, seed in VARIANTS:
            started = perf_counter()
            exact = evaluate(instance, targets,
                             [("최근접 진입", paths, SUPPORT_HALF_M), ("중심 경유", paths, SUPPORT_HALF_M)],
                             mission=mission, prior=prior, terrain_coupling=BELIEF_TERRAIN_FULL,
                             particle_count=MC_PARTICLES, dt_s=dt, seed=seed, belief_delay_s=delay,
                             entry=("nearest", "centre-sync"), progress=lambda *_: None)
            pd = np.array([t["pd"] for t in exact["targets"]])      # [표적, 규약]
            rows.append({"variant": name, "dt_s": float(exact["dt_s"]), "seed": seed,
                         "targets": [t["target"] for t in exact["targets"]],
                         "pd_nearest_per_target": pd[:, 0].tolist(),
                         "pd_centre_per_target": pd[:, 1].tolist()})
            print(f"  {name:<18} 간격 {exact['dt_s']:.2f} s | 최근접 "
                  + " ".join(f"{v:.4f}" for v in pd[:, 0]) + " | 중심 "
                  + " ".join(f"{v:.4f}" for v in pd[:, 1])
                  + f" | {perf_counter() - started:.0f} s", flush=True)
        ref = rows[0]
        for row in rows[1:]:
            d = max(abs(a - b) for key in ("pd_nearest_per_target", "pd_centre_per_target")
                    for a, b in zip(row[key], ref[key]))
            print(f"  {row['variant']}: 기준과의 최대 차 {d:.4f}")
        dst = ROOT / f"results/studies/convergence-{tag}.json"
        json.dump({"case": case, "n": n, "particles": MC_PARTICLES, "rows": rows},
                  open(dst, "w"), ensure_ascii=False, indent=2)
        print("  저장:", dst.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
