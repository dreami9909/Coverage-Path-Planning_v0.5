"""인증된 절단평면 계획에서 체류 복귀 규약이 MC 값을 얼마나 바꾸는가.

``tools/probes/dwell_probe.py`` 는 MAPPO 2단계 계획 묶음을 재채점하는 도구라
2단계 결과 파일이 없으면 빈 보고서가 나온다. 이 도구는 같은 질문을 **인증된
계획**(``anchor-*`` 류, ``conditions.available_conditions``)에 적용한다 —
체류 슬라이스마다 셀 중심으로 복귀하는 현행 규약과 그 자리에서 이어 쓰는
규약을 같은 난수로 나란히 채점한다. 진입 규약은 dwell_probe 와 같은 ``centre``
다 (중심 복귀는 중심 진입 규약의 선택지라서다).

실행:  python3 tools/probes/dwell_certified.py [TEL-10 Tank-8 ...]
       MC_PARTICLES (기본 20000)
저장:  results/studies/dwell-certified.json
"""
import json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from cpp_search.common.config import load_chapter_config
from cpp_search.common.options import RunOptions
from cpp_search.experiments.ch2_stone_spx import (
    build_instance, case_mission, case_specs, case_target_specs, cue_prior)
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL
from cpp_search.solvers.milp.continuous_mc import evaluate, SUPPORT_HALF_M

from conditions import available_conditions                 # noqa: E402

SEED, H = 20260201, 600.0
PARTICLES = int(os.environ.get("MC_PARTICLES", 20_000))


def main(argv: list[str]) -> None:
    wanted = set(argv) or None
    cfg = load_chapter_config("2")
    m0, sensor = cfg.mission(), cfg.sensor()
    base = next(g for g in cfg.get("stone_spx.grid_conditions", [])
                if g.get("grid_kind") == "square")
    out_dir = ROOT / "results/studies"
    report = []

    for case, n, plan_file in available_conditions():
        tag = f"{case}-{n}"
        if wanted and tag not in wanted:
            continue
        block = next(c for c in case_specs(cfg) if c["name"] == case)
        targets = case_target_specs(cfg, block)
        mission = case_mission(m0, block)
        opts = RunOptions(sample_count=10, particle_count=5000, episode_count=1,
                          target_profile="both", seed=1, map_error=False,
                          communication_loss_probability=0.0,
                          communication_latency_slices=0, mission_time_s=H)
        inst = build_instance(cfg, opts, mission, sensor,
                              grid={**base, "grid_width": n, "grid_height": n},
                              seed=SEED, terrain_weighting=True, targets=targets)
        stored = json.load(open(out_dir / plan_file))
        if stored.get("fingerprint") not in (None, inst.fingerprint):
            raise SystemExit(f"{tag}: fingerprint 불일치 — 저장된 계획과 다른 인스턴스다")
        paths = tuple(tuple(int(c) for c in p) for p in stored["assignments"])
        plans = [("중심복귀", paths, SUPPORT_HALF_M), ("이어서", paths, SUPPORT_HALF_M)]
        print(f"\n[{tag}]  입자 {PARTICLES:,}", flush=True)
        res = evaluate(inst, targets, plans, mission=mission,
                       prior=cue_prior(cfg, mission.search_radius_m)[0],
                       terrain_coupling=BELIEF_TERRAIN_FULL,
                       particle_count=PARTICLES, dt_s=5.0, seed=SEED,
                       entry=("centre", "centre"), start=("cell", "cell"),
                       dwell=(True, False),
                       belief_delay_s=float(cfg.get("stone_spx.initial_belief_delay_s", 0.0)))
        pd = np.array([t["pd"] for t in res["targets"]])   # [표적, 계획]
        old, new = float(pd[:, 0].min()), float(pd[:, 1].min())
        print(f"  최악표적 PD  중심복귀 {old:.4f}  이어서 {new:.4f}  차 {new - old:+.4f}",
              flush=True)
        report.append({
            "case": case, "n": n, "particles": PARTICLES,
            "targets": [t["target"] for t in res["targets"]],
            "worst_pd_centre_return": old, "worst_pd_continue": new,
            "delta": new - old,
            "pd_centre_return_per_target": pd[:, 0].tolist(),
            "pd_continue_per_target": pd[:, 1].tolist(),
        })

    dst = out_dir / "dwell-certified.json"
    json.dump(report, open(dst, "w"), ensure_ascii=False, indent=2)
    print("\n저장:", dst.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
