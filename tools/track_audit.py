"""셀 안 항적 감사 — 평가기가 날린 항적이 **계획이 말한 대로** 나는가.

모형 밖 MC 는 ``core/searcher_track.build_windows`` 가 편 항적 위에서 탐지를
계산한다. 그 항적이 틀리면 그 위의 수치가 전부 틀린다. 이 도구는 저장된 계획을
하나씩 실제로 깔아 다음을 **수치로** 확인한다.

    연속성      창과 창 사이가 끊기지 않는가, 창 하나의 길이가 v·dt 인가
    중심 복귀   같은 셀에 계속 머무는 슬라이스에서 이동 거리가 0 인가
    지각        셀을 옮길 때 한 슬라이스 안에 도착하는가 (못 하면 그만큼 밀린다)
    셀 준수     슬라이스 t 의 항적이 계획 셀(이동 중이면 직전 셀 포함) 안에 있는가
    소인 장부   항적이 셀마다 실제로 뿌린 hazard 가 셀 모형이 적은 값과 맞는가

항적을 만드는 코드는 건드리지 않는다. ``build_windows`` 에 장부(``ledger``)만
받아 적는다.

실행:  python3 tools/track_audit.py [Tank-8 ...]
       AUDIT_ENTRY   항적 규약: nearest(기준) | centre-sync(대조) | centre(이전 구현, 지각 결함)
감사하는 계획은 조건마다 열 개다 (``mappo2_rescore.combined_plans``).
"""
import json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.chapters.ch2_stone_spx import case_specs, case_mission
from cpp_search.chapters.ch2_studies import published_instance
from cpp_search.models.searcher_track import build_windows
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, _cell_mean_hazard
from cpp_search.theory.path_constrained import joint_survival

from mappo2_rescore import combined_plans                  # noqa: E402
from conditions import CONDITIONS                           # noqa: E402

SEED, DT_S = 20260201, 5.0
ENTRY = os.environ.get("AUDIT_ENTRY", "nearest")
SAMPLE_M = 10.0          # 셀 준수 검사에서 항적을 이 간격으로 찍는다
EDGE_M = 1.0             # 셀 경계 위를 나는 구간(차선 잇기)은 그 셀 안으로 센다


def _in_cell(grid, cell: int, px, py):
    """점들이 셀 ``cell`` 의 닫힌 상자(경계 포함) 안에 있는가."""
    cw, ch = float(grid.cell_width_m), float(grid.cell_height_m)
    x0 = float(grid.center.x) - float(grid.half_side_m) + (cell % grid.width) * cw
    y0 = float(grid.center.y) - float(grid.half_side_m) + (cell // grid.width) * ch
    return ((px >= x0 - EDGE_M) & (px <= x0 + cw + EDGE_M)
            & (py >= y0 - EDGE_M) & (py <= y0 + ch + EDGE_M))


def _sample(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """창 하나를 ``SAMPLE_M`` 간격의 점과 그 점이 대표하는 길이로 편다."""
    xs, ys, ws = [], [], []
    for a, b in zip(window[:-1], window[1:]):
        length = float(np.hypot(*(b - a)))
        if length < 1e-9:
            continue
        count = max(1, int(np.ceil(length / SAMPLE_M)))
        frac = (np.arange(count) + 0.5) / count
        xs.append(a[0] + frac * (b[0] - a[0]))
        ys.append(a[1] + frac * (b[1] - a[1]))
        ws.append(np.full(count, length / count))
    if not xs:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def audit_plan(instance, mission, paths, *, lane_m=SUPPORT_HALF_M,
               entry="nearest") -> dict:
    grid = instance.grid
    slice_s = float(instance.slice_s)
    sub = max(1, int(round(slice_s / DT_S)))
    side = float(grid.cell_area_m2) ** 0.5
    speed = float(mission.search_speed_mps)
    reach = float(mission.transit_speed_mps) * slice_s
    step_len = speed * slice_s / sub
    half = float(grid.half_side_m)
    cx, cy = float(grid.center.x), float(grid.center.y)

    max_jump = max_len_error = 0.0
    stay_slices = stay_with_transit = move_slices = late_moves = 0
    max_lag = final_lag = 0.0
    transit_actual = transit_model = 0.0
    length_total = length_in_plan = length_in_prev = length_outside_grid = 0.0
    stay_length = stay_in_cell = 0.0
    tracks = []
    for j, path in enumerate(paths):
        ledger: list[dict] = []
        start = instance.problem.searchers[j].start_state
        windows = build_windows(list(path), start, grid,
                                transit_speed=float(mission.transit_speed_mps),
                                search_speed=speed, slice_s=slice_s, lane_m=lane_m,
                                cell_side=side, sub=sub, entry=entry, ledger=ledger)
        tracks.append(windows)
        for k, window in enumerate(windows):
            length = float(np.hypot(np.diff(window[:, 0]), np.diff(window[:, 1])).sum())
            max_len_error = max(max_len_error, abs(length - step_len))
            if k:
                max_jump = max(max_jump, float(np.hypot(*(window[0] - windows[k - 1][-1]))))
        for row in ledger:
            t = row["slice"]
            previous = start if t == 0 else path[t - 1]
            if row["staying"]:
                stay_slices += 1
                stay_with_transit += row["transit_m"] > 1e-6
            elif int(previous) != int(path[t]):
                move_slices += 1
                late_moves += bool(row["late"])
                transit_actual += row["transit_m"]
                a, b = grid.center_of(int(previous)), grid.center_of(int(path[t]))
                transit_model += a.distance_to(b)
            max_lag = max(max_lag, row["lag_m"])
        final_lag += ledger[-1]["lag_m"]
        # 셀 준수: 슬라이스 t 의 창들이 어느 셀 위에 있는가
        for t in range(len(path)):
            previous = start if t == 0 else path[t - 1]
            for k in range(sub):
                px, py, w = _sample(windows[t * sub + k])
                if not w.size:
                    continue
                inside = (np.abs(px - cx) <= half + EDGE_M) & (np.abs(py - cy) <= half + EDGE_M)
                in_plan = _in_cell(grid, int(path[t]), px, py)
                in_prev = _in_cell(grid, int(previous), px, py) & ~in_plan
                length_total += w.sum()
                length_in_plan += w[in_plan].sum()
                length_in_prev += w[in_prev].sum()
                length_outside_grid += w[~inside].sum()
                if int(previous) == int(path[t]):
                    stay_length += w.sum()
                    stay_in_cell += w[in_plan].sum()

    # 소인 장부: 셀 모형이 (슬라이스, 셀) 에 적는 hazard 와 항적이 실제로 뿌린 면적평균
    model = np.array([-np.log(joint_survival(instance.problem, paths, t))
                      for t in range(instance.time_slice_count)])[:, :instance.cell_count]
    flown = _cell_mean_hazard(grid, tracks, instance.time_slice_count, sub)
    return {
        "max_jump_m": max_jump, "max_window_length_error_m": max_len_error,
        "stay_slices": stay_slices, "stay_slices_with_transit": int(stay_with_transit),
        "move_slices": move_slices, "late_moves": int(late_moves),
        "max_lag_slices": max_lag / (speed * slice_s),
        "mean_final_lag_slices": final_lag / len(paths) / (speed * slice_s),
        "transit_actual_over_model": (transit_actual / transit_model
                                      if transit_model else float("nan")),
        "share_in_planned_cell": length_in_plan / length_total,
        "share_in_previous_cell": length_in_prev / length_total,
        "share_elsewhere": 1.0 - (length_in_plan + length_in_prev) / length_total,
        "share_outside_grid": length_outside_grid / length_total,
        "stay_share_in_cell": stay_in_cell / stay_length if stay_length else float("nan"),
        "ledger_total_flown_over_model": float(flown.sum() / model.sum()),
        "ledger_mismatch": float(np.abs(flown - model).sum() / model.sum()),
    }


def main(argv: list[str]) -> None:
    wanted = set(argv) or None
    cfg = load_chapter_config("3")
    report = {}
    for case, n, _ in CONDITIONS:
        tag = f"{case}-{n}"
        if wanted and tag not in wanted:
            continue
        saved, combined = combined_plans(tag)
        block = next(c for c in case_specs(cfg) if c["name"] == case)
        mission = case_mission(cfg.mission(), block)
        instance = published_instance(cfg, case=block, width=n, seed=SEED)
        if instance.fingerprint != saved["fingerprint"]:
            raise SystemExit(f"{tag}: fingerprint 불일치 — 다른 인스턴스다")
        side = float(instance.grid.cell_area_m2) ** 0.5
        reach = float(mission.transit_speed_mps) * float(instance.slice_s)
        print(f"\n[{case} {n}×{n}]  셀 {side:.0f} m, 슬라이스 {instance.slice_s:.1f} s × "
              f"{instance.time_slice_count}, 도달거리/셀 {reach / side:.3f}"
              f"  [진입 규약: {ENTRY}]", flush=True)
        print(f"  {'계획':<14}{'끊김m':>7}{'체류':>5}{'복귀':>5}{'이동':>5}{'지각':>5}"
              f"{'최대밀림':>8}{'끝밀림':>7}{'이동거리비':>9}{'계획셀':>7}{'직전셀':>7}"
              f"{'그밖':>6}{'체류셀안':>8}{'소인합비':>8}{'어긋남':>7}")
        rows = {}
        for name, _, paths in combined:
            r = audit_plan(instance, mission, paths, entry=ENTRY)
            rows[name] = r
            print(f"  {name:<14}{r['max_jump_m']:>7.3f}{r['stay_slices']:>5}"
                  f"{r['stay_slices_with_transit']:>5}{r['move_slices']:>5}"
                  f"{r['late_moves']:>5}{r['max_lag_slices']:>8.2f}"
                  f"{r['mean_final_lag_slices']:>7.2f}{r['transit_actual_over_model']:>9.2f}"
                  f"{r['share_in_planned_cell']:>7.2f}{r['share_in_previous_cell']:>7.2f}"
                  f"{r['share_elsewhere']:>6.2f}{r['stay_share_in_cell']:>8.2f}"
                  f"{r['ledger_total_flown_over_model']:>8.2f}{r['ledger_mismatch']:>7.2f}",
                  flush=True)
        report[tag] = {"cell_side_m": side, "slice_s": float(instance.slice_s),
                       "reach_over_cell": reach / side, "plans": rows}
    suffix = "" if ENTRY == "nearest" else f"-{ENTRY}"
    for tag, block in report.items():      # 조건마다 파일 하나 — 조건별로 따로 돌릴 수 있다
        dst = ROOT / f"results/studies/track-audit{suffix}-{tag}.json"
        json.dump(block, open(dst, "w"), ensure_ascii=False, indent=2)
        print("저장:", dst.name)


if __name__ == "__main__":
    main(sys.argv[1:])
