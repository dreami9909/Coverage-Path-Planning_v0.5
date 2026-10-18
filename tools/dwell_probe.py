"""체류 슬라이스마다 **셀 중심으로 돌아가는 규약**이 결과를 얼마나 바꾸는가.

무엇이 문제였나
---------------
``core/searcher_track.build_windows`` 는 슬라이스마다 목적 셀의 중심을 항법
지점으로 쓴다. 같은 셀에 계속 머물 때도 일단 중심으로 돌아갔다가 회로를 이어
간다. 인증된 계획에서 그 복귀 비행이 체류 슬라이스의 18~30 % 를 먹었다.

비행으로 보면 돌아갈 이유가 없다. 이미 회로 위에 있으므로 그 자리에서 계속
쓸면 된다. 그래서 ``dwell_from_centre=False`` 를 만들고, 저장된 계획들을 두
규약으로 **같은 난수에서** 다시 채점한다.

바뀌면 체류가 긴 계획이 유리해진다. 절단평면 계획은 평균 2.7 단계, 학습 계획은
1.2 단계 머무르므로 2 단계 비교가 되돌아올 수 있다. 그 크기를 재는 것이 목적이다.

실행:  python3 tools/dwell_probe.py [Tank-8 ...]
       MC_PARTICLES (기본 20000)
"""
import json, os, pathlib, sys
from math import hypot
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.chapters.ch2_stone_spx import (
    case_specs, case_target_specs, case_mission, cue_prior)
from cpp_search.chapters.ch2_studies import published_instance
from cpp_search.core.searcher_track import cell_cover_polyline, _cum, _advance
from cpp_search.planning.continuous_mc import SUPPORT_HALF_M, evaluate
from cpp_search.planning.terrain_belief import BELIEF_TERRAIN_FULL

from mappo2_rescore import load_plans                      # noqa: E402
from conditions import stage2_results                       # noqa: E402

#: 어느 규약의 보상으로 학습한 2단계 계획 묶음을 쓸지 (tools/conditions.py)
PLANS = os.environ.get("PROBE_PLANS", "nearest")

PLANNING_SEED, DT_S = 20260201, 5.0
MC_PARTICLES = int(os.environ.get("MC_PARTICLES", 20_000))


def sweep_length(instance, mission, paths, *, lane_m: float,
                 dwell_from_centre: bool) -> tuple[float, float]:
    """계획 하나의 (소인 거리, 이동 거리). ``build_windows`` 의 장부와 같은 계산."""

    side = float(instance.grid.cell_area_m2) ** 0.5
    slice_s = float(instance.slice_s)
    transit_speed = float(mission.transit_speed_mps)
    search_speed = float(mission.search_speed_mps)
    swept = moved = 0.0
    for index, path in enumerate(paths):
        start = instance.problem.searchers[index].start_state
        centre = instance.grid.center_of(start)
        position = (float(centre.x), float(centre.y))
        current_cell, cover, consumed = None, None, 0.0
        for cell in path:
            target = instance.grid.center_of(int(cell))
            staying = int(cell) == current_cell and cover is not None
            distance = 0.0 if (staying and not dwell_from_centre) else hypot(
                target.x - position[0], target.y - position[1])
            transit_s = distance / transit_speed if transit_speed > 0 else 0.0
            dwell_len = max(0.0, search_speed * (slice_s - transit_s))
            if not staying:
                current_cell = int(cell)
                cover = _cum(cell_cover_polyline(
                    target.x, target.y, side, lane_m, (target.x, target.y)))
                consumed = 0.0
            points, cumulative = cover
            track = _advance(points, cumulative, consumed, dwell_len)
            consumed += dwell_len
            position = track[-1]
            swept += dwell_len
            moved += distance
    return swept, moved


def main(argv: list[str]) -> None:
    wanted = set(argv) or None
    cfg = load_chapter_config("3")
    delay = float(cfg.get("stone_spx.initial_belief_delay_s", 0.0))
    report = []

    for tag, saved in stage2_results(PLANS, wanted):
        case, n = saved["case"], saved["n"]
        block = next(c for c in case_specs(cfg) if c["name"] == case)
        targets = case_target_specs(cfg, block)
        mission = case_mission(cfg.mission(), block)
        instance = published_instance(cfg, case=block, width=n, seed=PLANNING_SEED)
        if instance.fingerprint != saved["fingerprint"]:
            raise SystemExit(f"{tag}: fingerprint 불일치 — 다른 인스턴스다")
        prior = cue_prior(cfg, mission.search_radius_m)[0]
        plans = load_plans(saved)

        print(f"\n[{case} {n}×{n}]  계획 {len(plans)}개, 입자 {MC_PARTICLES:,}", flush=True)
        entries, dwell_flags = [], []
        for name, paths in plans:
            entries.append((f"{name} · 중심복귀", paths, SUPPORT_HALF_M)); dwell_flags.append(True)
        for name, paths in plans:
            entries.append((f"{name} · 이어서", paths, SUPPORT_HALF_M)); dwell_flags.append(False)

        started = perf_counter()
        exact = evaluate(instance, targets, entries, mission=mission, prior=prior,
                         terrain_coupling=BELIEF_TERRAIN_FULL,
                         particle_count=MC_PARTICLES, dt_s=DT_S, seed=PLANNING_SEED,
                         belief_delay_s=delay, dwell=dwell_flags,
                         # 중심 복귀는 중심 진입 규약의 선택지다. 세 규약을 한꺼번에
                         # 견주는 것은 tools/entry_probe.py 가 한다.
                         entry=("centre",) * len(entries),
                         progress=lambda *_: None)
        pd = np.array([t["pd"] for t in exact["targets"]])          # [표적, 계획]
        names = [t["target"] for t in exact["targets"]]
        print(f"  평가기 {perf_counter() - started:.0f} s", flush=True)
        print(f"  {'계획':<16}{'중심복귀':>10}{'이어서':>10}{'차':>9}{'소인거리':>10}")
        rows = []
        count = len(plans)
        for index, (name, paths) in enumerate(plans):
            old, new = pd[:, index].min(), pd[:, index + count].min()
            swept_old, _ = sweep_length(instance, mission, paths,
                                        lane_m=SUPPORT_HALF_M, dwell_from_centre=True)
            swept_new, _ = sweep_length(instance, mission, paths,
                                        lane_m=SUPPORT_HALF_M, dwell_from_centre=False)
            gain = swept_new / swept_old - 1.0
            print(f"  {name:<16}{old:>10.4f}{new:>10.4f}{new - old:>+9.4f}{gain:>9.1%}",
                  flush=True)
            rows.append({
                "plan": name, "worst_pd_centre": float(old), "worst_pd_continue": float(new),
                "delta": float(new - old), "sweep_length_gain": float(gain),
                "pd_centre_per_target": pd[:, index].tolist(),
                "pd_continue_per_target": pd[:, index + count].tolist(),
            })
        report.append({"case": case, "n": n, "particles": MC_PARTICLES,
                       "targets": names, "rows": rows})

    dst = ROOT / "results/studies/dwell-probe.json"
    json.dump(report, open(dst, "w"), ensure_ascii=False, indent=2)
    print("\n저장:", dst.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
