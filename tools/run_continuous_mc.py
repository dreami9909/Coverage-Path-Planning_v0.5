"""모형 밖 MC 실행기 — 같은 계획을 여러 **셀 안 항로 간격**으로 날려 본다.

계획은 "단계 t 에 셀 j 에 있으라"까지만 말한다. 셀 안을 어떻게 날지는 정하지
않으므로, 평가기가 그것을 골라야 하고 그 선택이 결과를 좌우한다. 그래서 한
조건마다 두 가지를 함께 보고한다.

    규약  항로 간격 = 탐지폭        (교과서식 평행 소인)
    최선  같은 조건에서 PD 를 최대로 만드는 간격 (셀 하나짜리 대리문제로 탐색)

둘의 폭이 곧 "계획이 정하지 않은 자유도"의 크기다. 자유도는 하나 더 있다 — 새
셀에 **어디로 들어가는가**. 위 둘은 본 표 규약(셀 중심 경유)으로 날리고, 같은
계획을 참고 규약(가장 가까운 차선 끝)으로도 같은 난수에서 날려 나란히 싣는다
(``tools/conditions.py``). 출발점은 실제 발진 지점, 표적은 셀 모형과 같은 간격으로
전개한다 (평가기 기본값, 2026-10-08).

계획 순서는 [중심 경유(W), 최근접 진입(W), 최선 간격(중심 경유)] 이며 아래 도구가
이 순서를 읽는다: ``tools/gap_decomposition.py``, ``tools/grid_convergence.py``,
``paper/mc_tables.py``, ``paper/trendfig.py``.

실행:  python3 tools/run_continuous_mc.py [조건...]   (기본: 저장된 계획이 있는 조건 전부)
       MC_PARTICLES (기본 40000), MC_OUT (저장 접두어, 기본 mc3)
       본문 표 1·2 의 값: MC_PARTICLES=120000 MC_OUT=mc2
"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.options import RunOptions
from cpp_search.experiments.ch2_stone_spx import (
    build_instance, case_mission, case_specs, case_target_specs, cue_prior)
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL
from cpp_search.models.searcher_track import cell_cover_polyline, _cum, _advance
from cpp_search.solvers.milp.continuous_mc import (
    evaluate, selftest, accumulate_window, SUPPORT_HALF_M)

SEED, H = 20260201, 600.0
PARTICLES = int(__import__("os").environ.get("MC_PARTICLES", 40_000))
OUT_TAG = __import__("os").environ.get("MC_OUT", "mc3")
sys.path.insert(0, str(ROOT / "tools"))
from conditions import available_conditions, BASELINE, REFERENCE   # noqa: E402
LANES = (800.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 160.0, 120.0)


def best_lane(side_m: float, dwell_len_m: float, steps: int, points: int = 420):
    """셀 하나짜리 대리문제로 PD 를 최대로 만드는 항로 간격을 고른다.

    전 단계를 한 셀에 머문다고 보고 hazard 장을 만들어 E[1-exp(-h)] 를 비교한다.
    실제 계획은 이동도 하므로 근사지만, 어느 간격이 유리한지는 이것으로 갈린다.
    """
    g = (np.arange(points) + .5) / points * side_m - side_m / 2
    px, py = np.meshgrid(g, g); px, py = px.ravel(), py.ravel()
    table = []
    for lane in LANES:
        P, cum = _cum(cell_cover_polyline(0.0, 0.0, side_m, lane, (0.0, 0.0)))
        h = np.zeros(px.size); used = 0.0
        for _ in range(steps):
            h = accumulate_window(px, py, np.asarray(_advance(P, cum, used, dwell_len_m), float), 1.0, h)
            used += dwell_len_m
        table.append((float(np.mean(1 - np.exp(-h))), lane))
    table.sort(reverse=True)
    return table[0][1], table


def main(argv):
    if not selftest():
        raise SystemExit("selftest 실패 — 소인 규약이 어긋났다")
    wanted = set(argv) or None
    cfg = load_chapter_config("2")
    m0, sensor = cfg.mission(), cfg.sensor()
    base = next(g for g in cfg.get("stone_spx.grid_conditions", []) if g.get("grid_kind") == "square")
    out_dir = ROOT / "results/studies"

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
        side = float(inst.grid.cell_area_m2) ** 0.5
        dwell = float(mission.search_speed_mps) * float(inst.slice_s)
        pick, table = best_lane(side, dwell, inst.time_slice_count)
        paths = tuple(tuple(int(c) for c in p)
                      for p in json.load(open(out_dir / plan_file))["assignments"])
        print(f"\n[{case} {n}×{n}]  셀 {side:.0f} m,  단계 {inst.time_slice_count},  "
              f"입자 {PARTICLES:,}", flush=True)
        print("  대리문제 항로 탐색: " +
              ", ".join(f"{l:.0f}m={p:.3f}" for p, l in sorted(table, key=lambda x: -x[1])), flush=True)
        print(f"  -> 최선 간격 {pick:.0f} m,  규약 간격 {SUPPORT_HALF_M:.0f} m", flush=True)
        stored = json.load(open(out_dir / plan_file))
        if stored.get("fingerprint") not in (None, inst.fingerprint):
            raise SystemExit(f"{tag}: fingerprint 불일치 — 저장된 계획과 다른 인스턴스다")
        plans = [("중심 경유", paths, SUPPORT_HALF_M), ("최근접 진입", paths, SUPPORT_HALF_M),
                 ("최선", paths, pick)]
        res = evaluate(inst, targets, plans, mission=mission,
                       prior=cue_prior(cfg, mission.search_radius_m)[0],
                       terrain_coupling=BELIEF_TERRAIN_FULL,
                       particle_count=PARTICLES, dt_s=5.0, seed=SEED,
                       entry=(BASELINE, REFERENCE, BASELINE), start=("true",) * 3,
                       belief_delay_s=float(cfg.get("stone_spx.initial_belief_delay_s", 0.0)))
        res.update({"case": case, "n": n, "seed": SEED,
                    "lane_probe": [[l, p] for p, l in table]})
        dst = out_dir / f"{OUT_TAG}-{tag}.json"
        json.dump(res, open(dst, "w"), ensure_ascii=False, indent=2)
        print("  저장:", dst.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
