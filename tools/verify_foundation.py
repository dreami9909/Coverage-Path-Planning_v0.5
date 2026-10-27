"""기초 검증 — 셀 모형, 모형 밖 평가기, 그리고 둘을 잇는 고리가 맞는가.

코드를 다시 읽는 것으로는 계산이 맞는지 알 수 없다. 여기서는 **답을 아는 문제**에
넣어 보거나 **다른 방법으로 다시 계산**해 맞춰 본다. 항목마다 기준을 미리 정해 두고
통과/실패를 적는다.

이론값이 있는 것 (인스턴스 없이, ``python3 tools/verify_foundation.py analytic``)
    A1  직선 한 번 통과, 균등 정지 표적      PD = L·∫(1 − e^{−m p(u)}) du / 면적
    A2  평행 차선으로 완전 소인              PD = (1/s)∫(1 − e^{−m(p(u)+p(s−u))}) du
    A3  통과당 확률 읽기, 직선 한 번 통과    PD = m·W·L / 면적
    A4  항적을 창으로 잘게 자르는 방식에 값이 의존하지 않는가

격자별 (``python3 tools/verify_foundation.py Tank-8 ...``)
    C1  저장된 계획을 다시 채점한 값 = 저장값, 인스턴스 fingerprint 일치
    C2  계획이 이동 가능·점유·분리 규칙 안에 있는가
    C3  어떤 이동이든 한 단계의 소인량 합이 같은가 (W·v·Δ/셀면적)
    C4  상한·하한·간격이 서로 맞는가
    C5  탐색 없이 표적만 움직였을 때 셀별 존재확률이 두 모형에서 같은가
    C6  항적: 끊김, 창 길이, 지각, 계획 셀 준수, 탐색자 사이 최소 거리
    C7  실제 항적이 지면에 뿌린 소인 총량 = W × 항적 길이
    C8  평가기에 셀 모형의 전제를 넣으면 셀 모형 값이 나오는가 (erase-*.json)
    C9  입자 수를 바꿔도 값이 유지되는가 (4만 대 12만)
    C10 반드시 성립해야 하는 관계 (hazard 읽기 ≤ 셀 모형, 확률 읽기 ≥ hazard 읽기)

저장:  results/studies/verify-analytic.json, verify-{케이스}-{N}.json
"""
import json, os, pathlib, sys
from math import pi

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from cpp_search.models.searcher_track import build_windows, cell_cover_polyline, _cum, _advance
from cpp_search.solvers.milp.continuous_mc import (
    SUPPORT_HALF_M, accumulate_window, lateral_detection, particle_trace, _cell_index)

ST = ROOT / "results/studies"
W = SUPPORT_HALF_M
SEED, DT_S, PARTICLES = 20260201, 5.0, 40_000


def check(name, value, limit, detail=""):
    """항목 하나: 값이 기준 이하면 통과."""
    ok = bool(value <= limit)
    print(f"  [{'통과' if ok else '실패'}] {name}: {value:.3g} (기준 {limit:.3g}) {detail}", flush=True)
    return {"name": name, "value": float(value), "limit": float(limit), "pass": ok,
            "detail": detail}


# ------------------------------------------------------------------ 이론값 검증
def analytic() -> list[dict]:
    rng = np.random.default_rng(7)
    out = []
    m = 0.874
    u = np.linspace(-W, W, 400_001)
    p = lateral_detection(u)
    du = u[1] - u[0]

    # A1/A3: 직선 한 번 통과. 표적은 넓은 상자 안에 균등, 항적은 그 가운데를 지난다.
    box, length, count = 9000.0, 6000.0, 2_000_000
    px = rng.uniform(-box / 2, box / 2, count); py = rng.uniform(-box / 2, box / 2, count)
    track = np.array([[-length / 2, 0.0], [length / 2, 0.0]])
    for law, width in (("exponential", float(np.sum(1 - np.exp(-m * p)) * du)),
                       ("probability", float(np.sum(m * p) * du))):
        h = accumulate_window(px, py, track, m, np.zeros(count), law=law)
        got = float(np.mean(1 - np.exp(-h)))
        want = width * length / box ** 2
        se = float(np.std(1 - np.exp(-h)) / np.sqrt(count))
        out.append(check(f"A{'1' if law == 'exponential' else '3'} 직선 한 번 통과 ({law})",
                         abs(got - want), 4 * se,
                         f"평가기 {got:.6f}, 이론 {want:.6f}, 유효 폭 {width:.1f} m"))

    # A2: 평행 차선 완전 소인. 가장자리 영향을 빼려고 안쪽 표적만 본다.
    for spacing in (400.0, 444.2, 500.0):
        lanes = 13
        ys = (np.arange(lanes) - lanes // 2) * spacing
        px = rng.uniform(-1500, 1500, 600_000)
        py = rng.uniform(-2 * spacing, 2 * spacing, 600_000)
        h = np.zeros(px.size)
        for y in ys:
            accumulate_window(px, py, np.array([[-4000.0, y], [4000.0, y]]), m, h)
        got = float(np.mean(1 - np.exp(-h)))
        grid_u = np.linspace(0, spacing, 200_001)
        field = sum(lateral_detection(grid_u - k * spacing) for k in range(-2, 4))
        want = float(np.mean(1 - np.exp(-m * field)))
        se = float(np.std(1 - np.exp(-h)) / np.sqrt(px.size))
        out.append(check(f"A2 평행 차선 완전 소인 (간격 {spacing:.0f} m)", abs(got - want),
                         max(4 * se, 1e-9),
                         f"평가기 {got:.6f}, 이론 {want:.6f}"
                         + (f", 셀 모형식 1−e^(−m) = {1 - np.exp(-m):.6f}" if spacing == 400 else "")))

    # A4: 같은 항적을 창 3개로 자르든 24개로 자르든, 정지 표적의 hazard 는 같아야 한다.
    side = 2221.0
    P, cum = _cum(cell_cover_polyline(0.0, 0.0, side, W, (-side / 2, -side / 2 + side / 10), "x", True))
    px = rng.uniform(-side, side, 300_000); py = rng.uniform(-side, side, 300_000)
    total = float(cum[-1])
    fields = []
    for parts in (3, 24):
        h = np.zeros(px.size)
        for k in range(parts):
            seg = np.asarray(_advance(P, cum, total * k / parts, total / parts), dtype=float)
            accumulate_window(px, py, seg, 1.0, h)
        fields.append(h)
    out.append(check("A4 창을 자르는 방식에 무관 (회로 한 바퀴, 3 조각 대 24 조각)",
                     float(np.abs(fields[0] - fields[1]).max()), 1e-9,
                     f"평균 hazard {fields[0].mean():.6f}"))
    out.append(check("A4 회로 한 바퀴의 소인 총량 = W × 길이",
                     abs(fields[1].mean() * (2 * side) ** 2 / (W * total) - 1.0), 0.01,
                     f"회로 길이 {total:.0f} m, 비 {fields[1].mean() * (2 * side) ** 2 / (W * total):.4f}"))
    return out


# ------------------------------------------------------------------ 격자별 검증
def per_condition(case: str, n: int, plan_file: str) -> list[dict]:
    from cpp_search.config import load_chapter_config
    from cpp_search.experiments.ch2_stone_spx import (
        case_specs, case_target_specs, case_mission, cue_prior)
    from cpp_search.experiments.ch2_studies import published_instance
    from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL
    from cpp_search.theory import transitions as transition_ops
    from run_mappo_stage1 import audit_paths

    tag = f"{case}-{n}"
    cfg = load_chapter_config("2")
    block = next(c for c in case_specs(cfg) if c["name"] == case)
    targets = case_target_specs(cfg, block)
    mission = case_mission(cfg.mission(), block)
    instance = published_instance(cfg, case=block, width=n, seed=SEED)
    stored = json.load(open(ST / plan_file))
    paths = tuple(tuple(int(c) for c in p) for p in stored["assignments"])
    grid = instance.grid
    side = float(grid.cell_area_m2) ** 0.5
    slice_s = float(instance.slice_s)
    speed = float(mission.search_speed_mps)
    cells = instance.cell_count
    out = []
    print(f"\n[{case} {n}×{n}]  셀 {side:.0f} m, 단계 {instance.time_slice_count}", flush=True)

    # C1 재채점
    scored = instance.score_paths(paths)
    out.append(check("C1 저장된 계획 재채점 = 저장값", abs(min(scored) - stored["worst_pd"]), 1e-12))
    out.append(check("C1 인스턴스 fingerprint 일치",
                     0.0 if stored.get("fingerprint") in (None, instance.fingerprint) else 1.0, 0.0))

    # C2 실행가능성
    audit = audit_paths(instance, paths)
    out.append(check("C2 계획의 규칙 위반 수 (이동·점유·분리)",
                     audit["adjacency"] + audit["occupancy"] + audit["separation"], 0.0))

    # C3 소인 장부: 어떤 arc 든 합이 같다
    searcher = instance.problem.searchers[0]
    base = float(np.asarray(searcher.detection_rate)[0, :cells].max())
    sums = [sum(v.values()) for v in (searcher.swept_hazard or {}).values()]
    want = W * speed * slice_s / side ** 2
    out.append(check("C3 한 단계 소인량: 모든 이동에서 같은가",
                     max(abs(s - want) for s in sums) / want if sums else abs(base - want) / want,
                     1e-9, f"이동 {len(sums)}종, 이론 W·v·Δ/셀면적 = {want:.5f}"))
    stay = [v.get(a, 0.0) for (a, b), v in (searcher.swept_hazard or {}).items() if a == b]
    out.append(check("C3 제자리 체류의 소인량 = 이론값",
                     max(abs(s - want) for s in stay) / want if stay else 0.0, 1e-9))

    # C4 경계
    if stored.get("lower_bound_nondetection") is not None:
        L, U = float(stored["lower_bound_nondetection"]), float(stored["upper_bound_nondetection"])
        out.append(check("C4 하한 ≤ 상계", max(0.0, L - U), 0.0, f"L {L:.6f}, U {U:.6f}"))
        out.append(check("C4 간격 = (U − L)/U", abs((U - L) / U - stored["gap"]), 1e-9))
        out.append(check("C4 계획의 PD = 1 − U", abs((1 - U) - stored["worst_pd"]), 1e-9))

    # C5 표적 과정: 셀 모형의 주변분포 대 연속 입자의 셀 점유
    cache = os.environ.get("CPP_SEARCH_TRACE_CACHE")
    sub = max(1, int(round(slice_s / DT_S)))
    steps = instance.time_slice_count * sub
    delay = float(cfg.get("stone_spx.initial_belief_delay_s", 0.0))
    prior = cue_prior(cfg, mission.search_radius_m)[0]
    worst_tv = worst_out = worst_excess = 0.0
    rng = np.random.default_rng(3)
    for k, target in enumerate(targets):
        model = instance.target_models[k]
        weights, positions, escaped = particle_trace(
            instance, target, k, mission=mission, prior=prior,
            terrain_coupling=BELIEF_TERRAIN_FULL, particle_count=PARTICLES, seed=SEED,
            steps=steps, dt=slice_s / sub, belief_delay_s=delay, cache_dir=cache,
            stepping="model", slice_s=slice_s)        # 평가기와 같은 전개 (기본값과 같다)
        mass = np.asarray(model.initial_mass, dtype=float).copy()
        steps_p = transition_ops.as_sequence(model.transitions)
        for t in range(instance.time_slice_count):
            px, py = positions[t * sub, :, 0], positions[t * sub, :, 1]
            esc = escaped[t * sub]
            index = np.where(esc, cells, _cell_index(grid, px, py))
            occupancy = np.bincount(index, weights=weights, minlength=cells + 1)
            tv = 0.5 * float(np.abs(occupancy - mass).sum())
            # 표본만으로 생기는 TV: 같은 분포에서 5천 개와 4만 개를 뽑았을 때
            a = rng.multinomial(5000, occupancy / occupancy.sum()) / 5000
            b = rng.multinomial(PARTICLES, occupancy / occupancy.sum()) / PARTICLES
            noise = 0.5 * float(np.abs(a - b).sum())
            worst_tv = max(worst_tv, tv); worst_excess = max(worst_excess, tv - noise)
            worst_out = max(worst_out, abs(float(occupancy[cells]) - float(mass[cells])))
            if t < instance.time_slice_count - 1:
                mass = mass @ steps_p[t]
    out.append(check("C5 표적 셀 점유: 두 모형의 총변동거리 − 표본만의 몫", worst_excess, 0.05,
                     f"총변동거리 최대 {worst_tv:.3f}"))
    out.append(check("C5 격자 밖으로 나간 질량의 차", worst_out, 0.02))

    # C6 항적
    separation = []
    for entry in ("nearest", "centre-sync"):
        tracks, late, jump, length_error, outside = [], 0, 0.0, 0.0, 0.0
        for j, path in enumerate(paths):
            ledger = []
            start = instance.problem.searchers[j].start_state
            windows = build_windows(list(path), start, grid, transit_speed=float(mission.transit_speed_mps),
                                    search_speed=speed, slice_s=slice_s, lane_m=W, cell_side=side,
                                    sub=sub, entry=entry, ledger=ledger)
            tracks.append(windows)
            late += sum(r["late"] for r in ledger)
            for i, w in enumerate(windows):
                length_error = max(length_error, abs(float(np.hypot(np.diff(w[:, 0]), np.diff(w[:, 1])).sum())
                                                     - speed * slice_s / sub))
                if i:
                    jump = max(jump, float(np.hypot(*(w[0] - windows[i - 1][-1]))))
            for t in range(len(path)):
                previous = start if t == 0 else path[t - 1]
                for i in range(sub):
                    mid = 0.5 * (tracks[-1][t * sub + i][:-1] + tracks[-1][t * sub + i][1:])
                    if entry == "nearest":
                        c = _cell_index(grid, mid[:, 0], mid[:, 1])
                        centre_a, centre_b = grid.center_of(int(path[t])), grid.center_of(int(previous))
                        inside = ((np.abs(mid[:, 0] - centre_a.x) <= side / 2 + 1) & (np.abs(mid[:, 1] - centre_a.y) <= side / 2 + 1)) | \
                                 ((np.abs(mid[:, 0] - centre_b.x) <= side / 2 + 1) & (np.abs(mid[:, 1] - centre_b.y) <= side / 2 + 1))
                        outside = max(outside, float((~inside).mean()))
        ends = np.array([[w[-1] for w in track] for track in tracks])       # [탐색자, 스텝, 2]
        gap = min(float(np.hypot(*(ends[a] - ends[b]).T).min())
                  for a in range(len(tracks)) for b in range(a + 1, len(tracks)))
        separation.append(gap)
        label = "최근접 진입" if entry == "nearest" else "중심 경유"
        out.append(check(f"C6 항적 끊김 [{label}]", jump, 1e-6))
        out.append(check(f"C6 창 길이 오차 [{label}]", length_error, 1e-6))
        if entry == "nearest":
            out.append(check(f"C6 지각 횟수 [{label}]", float(late), 0.0))
            out.append(check(f"C6 계획 셀·직전 셀 밖 비행 [{label}]", outside, 0.0))
        # C7 소인 총량 보존
        half = float(grid.half_side_m) + W
        pts = np.random.default_rng(11).uniform(-half, half, size=(250_000, 2))
        pts[:, 0] += float(grid.center.x); pts[:, 1] += float(grid.center.y)
        h = np.zeros(len(pts))
        for track in tracks:
            for w in track:
                accumulate_window(pts[:, 0], pts[:, 1], w, 1.0, h)
        total = len(tracks) * speed * slice_s * instance.time_slice_count
        ratio = float(h.mean() * (2 * half) ** 2 / (W * total))
        out.append(check(f"C7 지면에 뿌린 소인 총량 / (W × 항적 길이) [{label}]", abs(ratio - 1.0), 0.02,
                         f"비 {ratio:.4f}"))
    out.append({"name": "C6 탐색자 사이 최소 거리 [m] (최근접 진입, 중심 경유)", "value": separation,
                "limit": None, "pass": None, "detail": "기준 없음 — 기록용"})
    print(f"  [기록] 탐색자 사이 최소 거리: 최근접 진입 {separation[0]:.0f} m, 중심 경유 {separation[1]:.0f} m")

    # C8~C10 저장된 결과 사이의 관계
    erase, mc2, law = (json.load(open(ST / f"{name}-{tag}.json")) if (ST / f"{name}-{tag}.json").exists() else None
                       for name in ("erase", "mc2", "law"))
    if erase:
        idx = {e: i for i, e in enumerate(erase["erase"])}
        gaps = [abs(erase["cell_model_pd_per_target"][0][k] - t["pd"][idx["cell"]])
                for k, t in enumerate(erase["targets"])]
        out.append(check("C8 셀 모형의 전제를 넣은 평가기 = 셀 모형", max(gaps), 0.02))
        over = max(t["pd"][idx["none"]] / erase["cell_model_pd_per_target"][0][k]
                   for k, t in enumerate(erase["targets"]))
        out.append(check("C10 hazard 읽기 MC ≤ 셀 모형 (비의 최대)", over, 1.0 + 0.01))
    if erase and mc2:
        diff = max(abs(a["pd"][idx["none"]] - b["pd"][0]) for a, b in zip(erase["targets"], mc2["targets"]))
        out.append(check("C9 입자 4만 대 12만의 차", diff, 0.012))
    if law:
        li = {l: i for i, l in enumerate(law["law"])}
        short = min(t["pd"][li["probability"]] - t["pd"][li["exponential"]] for t in law["targets"])
        out.append(check("C10 확률 읽기 ≥ hazard 읽기 (차의 최소, 음수면 위반)", -short, 0.0))
    return out


def main(argv):
    from conditions import available_conditions
    wanted = set(argv)
    if not wanted or "analytic" in wanted:
        print("[이론값이 있는 문제]")
        rows = analytic()
        json.dump(rows, open(ST / "verify-analytic.json", "w"), ensure_ascii=False, indent=2)
    for case, n, plan_file in available_conditions():
        tag = f"{case}-{n}"
        if wanted and tag not in wanted:
            continue
        rows = per_condition(case, n, plan_file)
        json.dump(rows, open(ST / f"verify-{tag}.json", "w"), ensure_ascii=False, indent=2)
        failed = [r["name"] for r in rows if r["pass"] is False]
        print(f"  => {tag}: {'전부 통과' if not failed else '실패 ' + ', '.join(failed)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
