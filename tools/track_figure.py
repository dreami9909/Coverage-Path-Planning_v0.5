"""저장된 계획 하나의 **실제 항적**을 규약별로 그린다 (눈으로 하는 감사).

``tools/track_audit.py`` 가 수치로 확인하는 것을 그림으로 본다. 탐색자마다 한 칸,
위 줄이 옛 규약(셀 중심으로 진입), 아래 줄이 새 규약(가장 가까운 차선 끝으로 진입).
색은 시간이다 — 슬라이스 경계마다 점을 찍고, 계획이 그 슬라이스에 지정한 셀을
옅게 칠한다. 점이 칠한 셀 밖에 있으면 그 슬라이스에 계획보다 뒤처진 것이다.

실행:  python3 tools/track_figure.py Tank 8
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from cpp_search.config import load_chapter_config
from cpp_search.chapters.ch2_stone_spx import case_specs, case_mission
from cpp_search.chapters.ch2_studies import published_instance
from cpp_search.models.searcher_track import build_windows
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M

from run_continuous_mc import CONDITIONS as BASE            # noqa: E402

SEED, DT_S = 20260201, 5.0
PLAN_FILES = {(c, n): f for c, n, f in BASE}
PLAN_FILES.update({("Tank", 12): "anchor-Tank-12-20260201.json",
                   ("Tank", 14): "anchor-Tank-14-20260201.json",
                   ("TEL", 12): "anchor-TEL-12-20260201.json"})


def main(argv):
    case, n = argv[0], int(argv[1])
    cfg = load_chapter_config("2")
    block = next(c for c in case_specs(cfg) if c["name"] == case)
    mission = case_mission(cfg.mission(), block)
    instance = published_instance(cfg, case=block, width=n, seed=SEED)
    grid = instance.grid
    stored = json.load(open(ROOT / "results/studies" / PLAN_FILES[(case, n)]))
    paths = [list(map(int, p)) for p in stored["assignments"]]
    side = float(grid.cell_area_m2) ** 0.5
    slice_s = float(instance.slice_s)
    sub = max(1, int(round(slice_s / DT_S)))
    half = float(grid.half_side_m)
    cx, cy = float(grid.center.x), float(grid.center.y)

    fig, axes = plt.subplots(2, len(paths), figsize=(3.4 * len(paths), 7.2),
                             squeeze=False)
    for row, (entry, label) in enumerate((("centre", "옛 규약: 셀 중심으로 진입"),
                                          ("nearest", "새 규약: 가장 가까운 차선 끝으로 진입"))):
        for j, path in enumerate(paths):
            ax = axes[row][j]
            ledger = []
            windows = build_windows(path, instance.problem.searchers[j].start_state, grid,
                                    transit_speed=float(mission.transit_speed_mps),
                                    search_speed=float(mission.search_speed_mps),
                                    slice_s=slice_s, lane_m=SUPPORT_HALF_M,
                                    cell_side=side, sub=sub, entry=entry, ledger=ledger)
            cells = sorted(set(path) | {instance.problem.searchers[j].start_state})
            xs = [grid.center_of(c).x for c in cells]; ys = [grid.center_of(c).y for c in cells]
            for c in cells:
                p = grid.center_of(c)
                ax.add_patch(Rectangle((p.x - side / 2, p.y - side / 2), side, side,
                                       fill=False, ec="0.75", lw=0.8))
            T = len(path)
            segs, colours = [], []
            for k, w in enumerate(windows):
                for a, b in zip(w[:-1], w[1:]):
                    segs.append([a, b]); colours.append(k / (T * sub))
            lc = LineCollection(segs, array=np.array(colours), cmap="viridis", lw=1.1)
            ax.add_collection(lc)
            late = 0
            for t in range(T):
                end = windows[(t + 1) * sub - 1][-1]
                p = grid.center_of(path[t])
                inside = (abs(end[0] - p.x) <= side / 2 + 1) and (abs(end[1] - p.y) <= side / 2 + 1)
                late += not inside
                ax.plot(end[0], end[1], "o", ms=3.2, mfc=("tab:red" if not inside else "white"),
                        mec=("tab:red" if not inside else "0.2"), mew=0.7, zorder=5)
            lag = max(r["lag_m"] for r in ledger) / (float(mission.search_speed_mps) * slice_s)
            n_late = sum(r["late"] for r in ledger)
            ax.set_title(f"UAV {j + 1} · 지각 {n_late}회 · 최대 밀림 {lag:.2f} 슬라이스",
                         fontsize=8.5)
            ax.set_xlim(min(xs) - side * 0.62, max(xs) + side * 0.62)
            ax.set_ylim(min(ys) - side * 0.62, max(ys) + side * 0.62)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=9.5)
    fig.suptitle(f"{case} {n}×{n} 절단평면 계획의 실제 항적 (셀 {side:.0f} m, 슬라이스 {slice_s:.1f} s, "
                 f"색 = 시간, 점 = 슬라이스 끝 위치, 빨간 점 = 그 슬라이스의 계획 셀 밖)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    dst = ROOT / f"results/studies/track-{case}-{n}.png"
    fig.savefig(dst, dpi=130)
    print("저장:", dst.name)


if __name__ == "__main__":
    main(sys.argv[1:])
