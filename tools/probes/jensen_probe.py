"""셀 모형이 왜 탐지확률을 높게 보는가 — 셀 하나로 좁혀 직접 잰다.

셀 모형은 한 셀의 hazard 를 ``a = 소인면적/셀면적`` 하나로 요약하고
탐지확률을 ``1 - exp(-a)`` 로 쓴다. 이 식은 **그 셀 안에서 hazard 가 고르게
깔렸다**고 전제한다. 실제 비행은 항적을 따라 띠 모양으로 쌓으므로 고르지 않다.

  1 - exp(-E[h])   셀 모형이 보는 값  (먼저 평균내고 지수)
  E[1 - exp(-h)]   실제 값            (먼저 지수내고 평균)

``1-exp(-·)`` 이 오목하므로 Jensen 부등식에 의해 항상 앞이 크거나 같다.
같아지는 것은 h 가 상수일 때뿐이다. 이 스크립트가 그 틈을 격자 해상도별로 잰다.
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from cpp_search.common.config import load_chapter_config
from cpp_search.experiments.ch2_stone_spx import auto_time_slice_count, case_specs, case_mission
from cpp_search.models.searcher_track import cell_cover_polyline, _cum, _advance
from cpp_search.solvers.milp.continuous_mc import accumulate_window, SUPPORT_HALF_M

GRID_POINTS = 700          # 셀 안 표본 격자 한 변


def probe(side_m: float, dwell_len_m: float, steps: int, *, lane_m=SUPPORT_HALF_M):
    """셀 하나에 ``steps`` 번 체류했을 때의 hazard 장을 만든다."""
    cover = _cum(cell_cover_polyline(0.0, 0.0, side_m, lane_m, (0.0, 0.0)))
    half = side_m / 2
    g = (np.arange(GRID_POINTS) + 0.5) / GRID_POINTS * side_m - half
    px, py = np.meshgrid(g, g)
    px, py = px.ravel(), py.ravel()
    hazard = np.zeros(px.size)
    consumed = 0.0
    P, cum = cover
    for _ in range(steps):
        pts = _advance(P, cum, consumed, dwell_len_m)
        accumulate_window(px, py, np.asarray(pts, dtype=float), 1.0, hazard)
        consumed += dwell_len_m
    return hazard


def main():
    cfg = load_chapter_config("2")
    H = float(cfg.resolve(None, "mission.mission_time_s", 300.0))
    m0, sensor = cfg.mission(), cfg.sensor()
    base = next(g for g in cfg.get("stone_spx.grid_conditions", [])
                if g.get("grid_kind") == "square")
    block = next(c for c in case_specs(cfg) if c["name"] == "Tank")
    mission = case_mission(m0, block)
    v = float(mission.search_speed_mps)

    print(f"{'N':>3} {'셀 c':>8} {'탐지폭/셀':>9} {'단계':>5} "
          f"{'E[h]':>8} {'이론 a':>8} {'셀모형':>8} {'실제':>8} {'비':>7} {'h 변동계수':>10}")
    rows = []
    for N in (4, 6, 8, 10, 12, 14):
        c = 2.0 * mission.search_radius_m / N
        T = auto_time_slice_count(mission, sensor, {**base, "grid_width": N}, H)
        dwell = v * (H / T)
        hazard = probe(c, dwell, T)
        mean_h = hazard.mean()
        theory_a = SUPPORT_HALF_M * v * H / (c * c)      # W·v·H/면적 (전 단계 체류)
        cell_model = 1.0 - np.exp(-mean_h)
        actual = float(np.mean(1.0 - np.exp(-hazard)))
        cv = hazard.std() / mean_h
        rows.append((SUPPORT_HALF_M / c, actual / cell_model, cv))
        print(f"{N:>3} {c:>7.0f}m {SUPPORT_HALF_M/c*100:>8.1f}% {T:>5} "
              f"{mean_h:>8.4f} {theory_a:>8.4f} {cell_model:>8.4f} {actual:>8.4f} "
              f"{actual/cell_model:>7.3f} {cv:>10.3f}")
    r = np.array([x[0] for x in rows]); ratio = np.array([x[1] for x in rows])
    cv = np.array([x[2] for x in rows])
    print(f"\n상관  탐지폭/셀 ↔ 비        : {np.corrcoef(r, ratio)[0,1]:+.3f}")
    print(f"상관  h 변동계수 ↔ 비        : {np.corrcoef(cv, ratio)[0,1]:+.3f}")
    print("\nE[h] 와 이론 a 가 일치하면 총 소인량은 맞는 것이다 — 틀린 것은 분포다.")


if __name__ == "__main__":
    main()
