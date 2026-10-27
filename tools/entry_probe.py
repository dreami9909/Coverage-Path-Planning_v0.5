"""새 셀에 **어떻게 들어가는가**가 결과를 얼마나 바꾸는가.

무엇이 문제였나
---------------
2026-10-05 이전의 항적은 셀을 옮길 때 지금 위치에서 다음 셀의 **중심**까지 날아간
뒤 회로를 시작했다. 셀 모형은 이동을 "중심에서 중심으로, 셀 한 변만큼"이라고
적지만, 기체는 회로 위 어딘가에 있으므로 실제 거리는 그보다 길 수 있다. 한
슬라이스에 날 수 있는 거리(도달거리)는 셀 한 변의 1.09~1.20 배뿐이라 그 거리가
넘치는 일이 흔했고, 넘친 만큼을 다음 슬라이스에서 갚지 않아 항적이 계획보다 계속
뒤처졌다 (``tools/track_audit.py``).

새 규약은 다음 셀의 가장 가까운 차선 끝으로 들어가 거기서 회로를 시작한다.
인접 셀이면 항상 제시간에 닿는다.

바뀐 것은 둘이다 — **시간 계산**(넘친 만큼을 갚는가)과 **진입점**(중심인가 가까운
차선 끝인가). 둘을 가르려고 가운데 규약을 하나 더 잰다.

    중심            이전 규약. 중심 경유, 넘친 시간을 갚지 않는다 (결함)
    중심·시간맞춤   중심 경유는 그대로, 넘친 시간은 다음 슬라이스가 갚는다
    최근접          가장 가까운 차선 끝으로 진입 (새 규약)

    중심 → 중심·시간맞춤   결함 수정만의 몫
    중심·시간맞춤 → 최근접  진입점을 바꾼 것만의 몫

저장된 계획들을 세 규약으로 **같은 난수에서** 다시 채점한다. 계획은 조건마다 열이다.

    절단평면            저장된 인증 계획
    셀 학습 s1~s3       셀 모형에서 보상을 받은 정책 (1단계)
    연속 학습·최근접    최근접 진입 규약의 연속 보상으로 학습한 정책 (2단계)
    연속 학습·중심      중심 경유 규약의 연속 보상으로 학습한 정책 (2단계)

연속 학습 정책은 자기가 보상을 받은 규약이 따로 있으므로, 이 표가 곧 **교차 채점**
이다 — 한 규약에서 배운 정책을 다른 규약으로 날리면 얼마나 잃는가. 본문 표 4 가
이 파일에서 나온다.

실행:  python3 tools/entry_probe.py [Tank-8 ...]
       MC_PARTICLES (기본 40000)
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
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, evaluate, selftest
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL

from mappo2_rescore import combined_plans                  # noqa: E402
from conditions import CONDITIONS                           # noqa: E402

PLANNING_SEED, DT_S = 20260201, 5.0
MC_PARTICLES = int(os.environ.get("MC_PARTICLES", 40_000))
CONVENTIONS = (("중심", "centre"), ("중심·시간맞춤", "centre-sync"), ("최근접", "nearest"))


def main(argv: list[str]) -> None:
    if not selftest():
        raise SystemExit("selftest 실패 — 소인 규약이 어긋났다")
    wanted = set(argv) or None
    cfg = load_chapter_config("3")
    delay = float(cfg.get("stone_spx.initial_belief_delay_s", 0.0))

    for case, n, _ in CONDITIONS:
        tag = f"{case}-{n}"
        if wanted and tag not in wanted:
            continue
        saved, combined = combined_plans(tag)
        plans = [(name, paths) for name, _, paths in combined]
        groups = [group for _, group, _ in combined]
        block = next(c for c in case_specs(cfg) if c["name"] == case)
        targets = case_target_specs(cfg, block)
        mission = case_mission(cfg.mission(), block)
        instance = published_instance(cfg, case=block, width=n, seed=PLANNING_SEED)
        if instance.fingerprint != saved["fingerprint"]:
            raise SystemExit(f"{tag}: fingerprint 불일치 — 다른 인스턴스다")
        prior = cue_prior(cfg, mission.search_radius_m)[0]

        entries, modes = [], []
        for label, mode in CONVENTIONS:
            for name, paths in plans:
                entries.append((f"{name} · {label}", paths, SUPPORT_HALF_M))
                modes.append(mode)
        started = perf_counter()
        exact = evaluate(instance, targets, entries, mission=mission, prior=prior,
                         terrain_coupling=BELIEF_TERRAIN_FULL,
                         particle_count=MC_PARTICLES, dt_s=DT_S, seed=PLANNING_SEED,
                         belief_delay_s=delay, entry=modes, progress=lambda *_: None)
        pd = np.array([t["pd"] for t in exact["targets"]])          # [표적, 계획]
        names = [t["target"] for t in exact["targets"]]
        cell = np.array(exact["cell_model_pd_per_target"])          # [계획, 표적]
        count = len(plans)
        print(f"\n[{case} {n}×{n}]  계획 {count}개, 입자 {MC_PARTICLES:,}, "
              f"평가기 {perf_counter() - started:.0f} s", flush=True)
        print(f"  {'계획':<18}{'셀 모형':>9}{'중심':>9}{'시간맞춤':>9}{'최근접':>9}"
              f"{'결함 몫':>9}{'진입점 몫':>10}")
        rows = []
        for index, (name, _) in enumerate(plans):
            old, sync, new = (pd[:, index + k * count] for k in range(3))
            print(f"  {name:<18}{cell[index].min():>9.4f}{old.min():>9.4f}{sync.min():>9.4f}"
                  f"{new.min():>9.4f}{sync.min() - old.min():>+9.4f}"
                  f"{new.min() - sync.min():>+10.4f}", flush=True)
            rows.append({
                "plan": name, "group": groups[index],
                "cell_model_pd_per_target": cell[index].tolist(),
                "pd_centre_per_target": old.tolist(),
                "pd_centre_sync_per_target": sync.tolist(),
                "pd_nearest_per_target": new.tolist(),
                "worst_pd_centre": float(old.min()),
                "worst_pd_centre_sync": float(sync.min()),
                "worst_pd_nearest": float(new.min()),
                "delta_worst": float(new.min() - old.min()),
                "delta_worst_timing": float(sync.min() - old.min()),
                "delta_worst_entry": float(new.min() - sync.min()),
            })
        dst = ROOT / f"results/studies/entry-{tag}.json"
        json.dump({"case": case, "n": n, "particles": MC_PARTICLES, "seed": PLANNING_SEED,
                   "fingerprint": instance.fingerprint, "targets": names, "rows": rows},
                  open(dst, "w"), ensure_ascii=False, indent=2)
        print("  저장:", dst.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
