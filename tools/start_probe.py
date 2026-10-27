"""출발 위치를 **셀 중심으로 반올림**한 것이 결과를 얼마나 바꾸는가.

무엇이 문제인가
---------------
탐색자의 실제 출발점은 발사 고리 위의 한 점이다. 셀 모형은 그 점을 가장 가까운
셀의 중심으로 옮겨 놓고 계획을 세우고, 모형 밖 MC 도 같은 셀 중심에서 항적을
시작한다. 옮겨지는 거리는 격자가 거칠수록 크다 (Tank 4x4 에서 최대 2.8 km,
16x16 에서 0.6 km). 표적·센서·시간은 격자와 무관한데 **출발 위치만 격자에 의존**
하므로, 격자 사이에 MC 값을 견줄 때 이 반올림이 섞여 들어간다.

같은 인증 계획을 두 출발 위치로 **같은 난수에서** 채점해 그 크기를 잰다. 두 항적
규약 모두에서 잰다.

출발점 하나를 바꾼 결과는 표본 하나일 뿐이다. 그래서 출발점을 **출발 셀 안에서
무작위로** 여러 번 흔들어, 한 계획의 탐지확률이 출발 위치에 얼마나 흔들리는지를
표준편차로도 잰다. 계획은 셀 안 위치를 정하지 않으므로 이 흔들림은 계획의 성질이
아니라 **평가 규약이 만든 잡음**이다.

실행:  python3 tools/start_probe.py [Tank-8 ...]
       MC_PARTICLES (기본 40000), START_JITTERS (흔드는 횟수, 기본 12)
저장:  results/studies/start-{케이스}-{N}.json
"""
import json, os, pathlib, sys
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.experiments.ch2_stone_spx import (
    case_specs, case_target_specs, case_mission, cue_prior)
from cpp_search.experiments.ch2_studies import published_instance
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, evaluate, selftest
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL

from conditions import available_conditions, convention_label       # noqa: E402

SEED, DT_S = 20260201, 5.0
MC_PARTICLES = int(os.environ.get("MC_PARTICLES", 40_000))
JITTERS = int(os.environ.get("START_JITTERS", 12))
VARIANTS = (("nearest", "cell"), ("nearest", "true"), ("centre-sync", "cell"), ("centre-sync", "true"))


def main(argv: list[str]) -> None:
    if not selftest():
        raise SystemExit("selftest 실패 — 소인 규약이 어긋났다")
    wanted = set(argv) or None
    cfg = load_chapter_config("2")
    delay = float(cfg.get("stone_spx.initial_belief_delay_s", 0.0))
    for case, n, plan_file in available_conditions():
        tag = f"{case}-{n}"
        if wanted and tag not in wanted:
            continue
        block = next(c for c in case_specs(cfg) if c["name"] == case)
        targets = case_target_specs(cfg, block)
        mission = case_mission(cfg.mission(), block)
        instance = published_instance(cfg, case=block, width=n, seed=SEED)
        stored = json.load(open(ROOT / "results/studies" / plan_file))
        if stored.get("fingerprint") not in (None, instance.fingerprint):
            raise SystemExit(f"{tag}: fingerprint 불일치 — 저장된 계획과 다른 인스턴스다")
        paths = tuple(tuple(int(c) for c in p) for p in stored["assignments"])
        shift = [float(np.hypot(point.x - instance.grid.center_of(cell).x,
                                point.y - instance.grid.center_of(cell).y))
                 for point, cell in zip(instance.initial_positions, instance.starts)]
        # 출발 셀 안에서 균등하게 뽑은 출발점 묶음 (탐색자마다 하나씩, 묶음마다 다르게)
        side = float(instance.grid.cell_area_m2) ** 0.5
        rng = np.random.default_rng(SEED + 17 * n)
        jitters = []
        for _ in range(JITTERS):
            jitters.append([(instance.grid.center_of(cell).x + rng.uniform(-side / 2, side / 2),
                             instance.grid.center_of(cell).y + rng.uniform(-side / 2, side / 2))
                            for cell in instance.starts])
        variants = list(VARIANTS) + [(entry, origin) for entry in ("nearest", "centre-sync")
                                     for origin in jitters]
        started = perf_counter()
        exact = evaluate(
            instance, targets,
            [(f"v{i}", paths, SUPPORT_HALF_M) for i in range(len(variants))],
            mission=mission, prior=cue_prior(cfg, mission.search_radius_m)[0],
            terrain_coupling=BELIEF_TERRAIN_FULL, particle_count=MC_PARTICLES, dt_s=DT_S,
            seed=SEED, belief_delay_s=delay, entry=[e for e, _ in variants],
            start=[s for _, s in variants], progress=lambda *_: None)
        pd = np.array([t["pd"] for t in exact["targets"]])          # [표적, 변형]
        names = [t["target"] for t in exact["targets"]]
        print(f"\n[{case} {n}×{n}]  출발 위치 반올림 평균 {np.mean(shift):.0f} m, 최대 "
              f"{max(shift):.0f} m, 평가기 {perf_counter() - started:.0f} s", flush=True)
        print(f"  {'항적 규약':<12}{'셀 중심 출발':>12}{'실제 위치 출발':>14}{'차(최저)':>10}   "
              + "".join(f"{name[:10]:>12}" for name in names))
        rows = []
        for k, entry in enumerate(("nearest", "centre-sync")):
            cell_start, true_start = pd[:, 2 * k], pd[:, 2 * k + 1]
            print(f"  {convention_label(entry):<12}{cell_start.min():>12.4f}{true_start.min():>14.4f}"
                  f"{true_start.min() - cell_start.min():>+10.4f}   "
                  + "".join(f"{b - a:>+12.4f}" for a, b in zip(cell_start, true_start)), flush=True)
            block = pd[:, 4 + k * JITTERS: 4 + (k + 1) * JITTERS]      # [표적, 흔든 횟수]
            lowest = block.min(axis=0)
            print(f"    출발점을 {JITTERS}번 흔들면: 최저 PD 평균 {lowest.mean():.4f}, 표준편차 "
                  f"{lowest.std(ddof=1):.4f}, 범위 {lowest.min():.4f}~{lowest.max():.4f} | "
                  f"거동별 표준편차 " + " ".join(f"{v:.4f}" for v in block.std(axis=1, ddof=1)),
                  flush=True)
            rows.append({"entry": entry, "pd_cell_start_per_target": cell_start.tolist(),
                         "pd_true_start_per_target": true_start.tolist(),
                         "delta_worst": float(true_start.min() - cell_start.min()),
                         "jitter_pd_per_target": block.tolist(),
                         "jitter_worst_mean": float(lowest.mean()),
                         "jitter_worst_std": float(lowest.std(ddof=1)),
                         "jitter_std_per_target": block.std(axis=1, ddof=1).tolist()})
        dst = ROOT / f"results/studies/start-{tag}.json"
        json.dump({"case": case, "n": n, "particles": MC_PARTICLES, "seed": SEED,
                   "fingerprint": instance.fingerprint, "targets": names,
                   "start_shift_m": shift, "rows": rows},
                  open(dst, "w"), ensure_ascii=False, indent=2)
        print("  저장:", dst.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
