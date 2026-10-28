"""그림 7 — 셀 모형(MILP)과 모형 밖 MC 의 탐지확률 비가 무엇을 따라 움직이는가.

조건들에서는 격자 N, 탐지폭 비율, 시간 단계 수, 최적성 간격, 탐지확률 수준이
함께 커진다. 상관만으로는 원인을 가를 수 없으므로 한 변수만 다른 비교를 나란히 둔다.

    (가) N 에 따른 비 — 거동별로 나누어 본다. 참고 규약(최근접 진입)은 세 거동 평균만 겹친다
    (나) 최적성 간격에 따른 비 — N = 10 에서 간격만 다른 계획 다섯 개를 겹친다
    (다) 셀 모형 탐지확률 수준에 따른 절대 차 — 거동별

색 선과 표식은 본 표 규약(중심 경유)이다.
자료: results/studies/gap-decomposition.json (tools/audit/gap_decomposition.py, 입자 12만 실행)
      results/studies/mc2-rank-Tank-10.json   (같은 인스턴스의 계획 다섯 개, 공통 난수)
간격은 각 계획을 낸 인증 결과 파일에서 읽고, 셀 모형 값이 자릿수까지 같은지로 짝을 확인한다.
"""
import json, pathlib, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = pathlib.Path(__file__).resolve().parents[1]
ST = ROOT / "results/studies"
sys.path.insert(0, str(ROOT / "tools"))
from conditions import available_conditions                 # noqa: E402

for cand in ("Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Malgun Gothic"):
    if cand in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({
    "font.size": 7.2, "axes.unicode_minus": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6, "axes.labelsize": 7.2, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "mathtext.fontset": "dejavusans",
})
W = 3.23          # 단 폭 [inch] ~ 8.2 cm (make_figs.py 와 같다)
COL = {"STOP_HEAVY": "#a8542a", "CONTINUOUS_MOVE": "#176a73",
       "MANEUVER_HEAVY": "#6b7f2a", "RELOCATION_HEAVY": "#6a5a9c"}
INK, GREY = "#1a1a1a", "#5b6472"

# ---------------------------------------------------------------- 자료
rows = json.load(open(ST / "gap-decomposition.json"))
PLAN = {f"{case} {n}×{n}": f for case, n, f in available_conditions()}   # 도구들과 같은 계획
cond = {}
for c, f in PLAN.items():
    src = json.load(open(ST / f))
    rs = [r for r in rows if r["cond"] == c]
    if not rs:
        continue                               # 아직 채점하지 않은 조건
    lo_cell = min(r["cell"] for r in rs)
    assert abs(src["worst_pd"] - lo_cell) < 1e-12, f"{c}: 계획 불일치"
    cond[c] = {"n": int(c.split("×")[-1]), "gap": 100 * src["gap"],
               "ratio": min(r["conv"] for r in rs) / lo_cell, "tel": c.startswith("TEL"),
               "nearest_mean": float(np.mean([r["nearest"] / r["cell"] for r in rs]))}
rows = [r for r in rows if r["cond"] in cond]

rk = json.load(open(ST / "mc2-rank-Tank-10.json"))
def _gap_of(pd_cell):
    """셀 모형 값이 같은 인증 결과를 찾아 그 간격을 돌려준다."""
    cands = []
    for f in ("scale-Tank-10-300x60.json", "anchor-Tank-10-20260201.json", "budget-Tank-10-60x60.json"):
        d = json.load(open(ST / f)); cands.append((d["worst_pd"], d["gap"]))
    for f in ("budget-alloc-Tank-10-T3600-laneA.json", "budget-alloc-Tank-10-T3600-laneB.json"):
        r = json.load(open(ST / f))["rows"][0]
        cands.append((1 - r["upper_bound_nondetection"], r["relative_optimality_gap"]))
    hit = [g for p, g in cands if abs(p - pd_cell) < 1e-12]
    assert len(hit) == 1, pd_cell
    return 100 * hit[0]
rank_gap = [_gap_of(p) for p in rk["cell_model_pd"]]
rank_ratio = np.array([t["pd"] for t in rk["targets"]]).min(axis=0) / np.array(rk["cell_model_pd"])

# ---------------------------------------------------------------- 그림
fig, (a, b, c) = plt.subplots(3, 1, figsize=(W, W * 1.52),
                              gridspec_kw={"hspace": 0.62})
for ax in (a, b, c):
    ax.spines[["top", "right"]].set_visible(False)

# (가) N 에 따른 비 — 거동별
for beh, col in COL.items():
    tank = sorted((cond[r["cond"]]["n"], r["conv"] / r["cell"]) for r in rows
                  if r["target"] == beh and not cond[r["cond"]]["tel"])
    if tank:
        a.plot(*zip(*tank), "-o", color=col, ms=2.6, lw=1.3 if beh == "STOP_HEAVY" else 0.8,
               label=beh)
    tel = [(cond[r["cond"]]["n"] + 0.3, r["conv"] / r["cell"]) for r in rows
           if r["target"] == beh and cond[r["cond"]]["tel"]]
    if tel:
        a.plot(*zip(*tel), "D", mfc="white", mec=col, mew=0.9, ms=3.0,
               label=None if tank else beh)
# 참고 규약(최근접 진입): 세 거동 평균만, 회색 점선
centre = sorted((v["n"], v["nearest_mean"]) for v in cond.values() if not v["tel"])
a.plot(*zip(*centre), "--", color=GREY, lw=0.8, label="최근접 진입 (세 거동 평균)")
centre_tel = [(v["n"] + 0.3, v["nearest_mean"]) for v in cond.values() if v["tel"]]
if centre_tel:
    a.plot(*zip(*centre_tel), "D", mfc="white", mec=GREY, mew=0.7, ms=2.4)
grid_n = sorted({v["n"] for v in cond.values()})
a.set_xticks(grid_n); a.set_xlim(3.6, max(grid_n) + 0.9); a.set_ylim(0.4, 1.02)
a.set_xlabel("격자 해상도 $N$  ($N \\times N$)"); a.set_ylabel("비  (MC / 셀 모형)")
a.legend(frameon=False, fontsize=5.0, loc="lower right", handlelength=1.6, borderaxespad=0.2)
a.text(0.03, 0.88, "(가)", transform=a.transAxes, fontsize=6.8)
a.text(0.16, 0.885, "◇ TEL", transform=a.transAxes, fontsize=5.4, color=GREY)

# (나) 간격에 따른 비 — 조건 사이 대 N 고정
for c_, v in cond.items():
    b.plot(v["gap"], v["ratio"], "o", ms=3.4, color=INK, mfc="white" if v["tel"] else INK, mew=0.8)
    lab = f"{v['n']}" + ("T" if v["tel"] else "")
    off = (-3, 4) if v["tel"] else (3, -5.5)
    b.annotate(lab, (v["gap"], v["ratio"]), textcoords="offset points", xytext=off,
               fontsize=5.2, color=INK, ha="right" if v["tel"] else "left")
b.plot(rank_gap, rank_ratio, "s", ms=3.0, color="#c07a1a",
       label="$N$=10 고정, 계산 조건만 다른 계획 5개")
b.set_xscale("log"); b.set_xlim(0.006, 60); b.set_ylim(0.4, 1.02)
b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
b.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
b.set_xlabel("최적성 간격 [%]  (로그 눈금)"); b.set_ylabel("비  (MC / 셀 모형)")
b.legend(frameon=False, fontsize=5.4, loc="lower right", handlelength=1.2, borderaxespad=0.2)
b.text(0.03, 0.88, "(나)", transform=b.transAxes, fontsize=6.8)
b.text(0.16, 0.885, "숫자 = $N$,  T = TEL", transform=b.transAxes, fontsize=5.4, color=GREY)

# (다) 절대 차 대 셀 모형 탐지확률 — 거동별
moving = [r["conv"] / r["cell"] for r in rows if r["target"] != "STOP_HEAVY"]
lo, hi = min(moving), max(moving)
x = np.linspace(0.3, 0.9, 50)
c.fill_between(x, (1 - hi) * x, (1 - lo) * x, color=GREY, alpha=0.14, lw=0)
for r in rows:
    tel = cond[r["cond"]]["tel"]; col = COL[r["target"]]
    c.plot(r["cell"], r["cell"] - r["conv"], "D" if tel else "o", ms=3.0 if tel else 2.8,
           color=col, mfc="white" if tel else col, mew=0.9)
    if r["target"] == "STOP_HEAVY" and not tel:
        c.annotate(f"$N$={cond[r['cond']]['n']}", (r["cell"], r["cell"] - r["conv"]),
                   textcoords="offset points", xytext=(3.5, 1.5), fontsize=5.6, color=col)
c.set_xlim(0.3, 0.9); c.set_ylim(0, 0.15)
c.set_xlabel("셀 모형 탐지확률 (거동별)"); c.set_ylabel("차  (셀 모형 - MC)")
c.text(0.03, 0.88, "(다)", transform=c.transAxes, fontsize=6.8)
c.text(0.89, 0.006, f"회색 띠: 움직이는 거동의 비 {lo:.2f} ~ {hi:.2f}", fontsize=5.4,
       color=GREY, ha="right")

out = ROOT / "paper/fig/fig7_model_gap_trends.png"
fig.savefig(out); plt.close(fig)
print("저장:", out.relative_to(ROOT))
print("N=10 고정 계획:", [f"{g:.2f}% → {r:.3f}" for g, r in zip(rank_gap, rank_ratio)])
