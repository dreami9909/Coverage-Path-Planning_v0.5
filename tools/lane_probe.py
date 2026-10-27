"""MC 가 가정한 '셀 안 비행 패턴'에 결과가 얼마나 좌우되는가.

MILP 계획은 "단계 t 에 셀 j 에 있으라"까지만 말한다. **셀 안에서 어떻게 날지는
말하지 않는다.** 모형 밖 MC 는 그것을 스스로 정해야 하고, 지금은 항로 간격을
탐지폭과 같게 둔 보스트로페돈을 쓴다. 그 선택이 결과를 좌우한다면 "20 % 차이"의
일부는 이산화 오차가 아니라 **평가기의 가정**이다.
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from cpp_search.models.searcher_track import cell_cover_polyline, _cum, _advance
from cpp_search.solvers.milp.continuous_mc import accumulate_window, SUPPORT_HALF_M

SIDE, V, SLICE, STEPS = 2221.0, 44.4444, 54.5455, 11       # Tank 8x8 실측 조건
GRID = 600

def field(lane, side=SIDE, dwell=V*SLICE, steps=STEPS):
    P, cum = _cum(cell_cover_polyline(0.0, 0.0, side, lane, (0.0, 0.0)))
    g = (np.arange(GRID) + .5) / GRID * side - side / 2
    px, py = np.meshgrid(g, g); px, py = px.ravel(), py.ravel()
    h = np.zeros(px.size); used = 0.0
    for _ in range(steps):
        pts = _advance(P, cum, used, dwell)
        accumulate_window(px, py, np.asarray(pts, float), 1.0, h)
        used += dwell
    return h

theory = SUPPORT_HALF_M * V * SLICE * STEPS / (SIDE * SIDE)
print(f"Tank 8×8 조건: 셀 {SIDE:.0f} m, 탐지폭 {SUPPORT_HALF_M:.0f} m, {STEPS}단계 전부 체류")
print(f"셀 모형이 보는 값 = 1 - exp(-{theory:.4f}) = {1-np.exp(-theory):.4f}\n")
print(f"{'항로 간격':>10}{'항로 수':>8}{'E[h]':>9}{'셀밖 유출':>10}{'실제 PD':>10}{'셀모형 대비':>11}")
for lane in (800, 600, SUPPORT_HALF_M, 300, 200, 120):
    h = field(lane)
    n = max(1, int(SIDE // lane))
    actual = float(np.mean(1 - np.exp(-h)))
    print(f"{lane:>9.0f}m{n:>8}{h.mean():>9.4f}{1-h.mean()/theory:>9.1%}"
          f"{actual:>10.4f}{actual/(1-np.exp(-theory)):>11.3f}")
print("\n※ '셀밖 유출' 은 항적의 소인이 셀 경계를 넘어 이웃 셀에 떨어진 비율이다.")
print("   물리적으로는 사라진 게 아니라 이웃이 받는다 — 셀 모형만 전부 이 셀에 적는다.")
