"""셀 모형과 연속 공간의 차이를 거동별로 분해한다.

Tank 4x4 에서 드러난 것: 이동 가설은 차이가 작고(0.03~0.05) **정지 가설만 크게
벌어진다**(0.20). 이는 셀 모형의 Markov 축약이 셀 안 위치를 매 단계 지워
**모든 표적에게 섞임(mixing)을 준다**는 뜻이다. 실제로 그 성질을 가진 것은
움직이는 표적뿐이다. 이 스크립트가 그 상관을 잰다.

'머무름'은 선언된 모드 비중이 아니라 **추정된 전이행렬의 대각 질량**으로 잰다 —
측정값이므로 선언값과 달리 그 자체로 검증 대상이다.

두 실행을 쓴다.
    mc2-*.json  입자 12만, 규약 항로 — 본문 표 1·2 의 MC 값 (본 표 규약과 참고 규약)
    mc3-*.json  입자 4만, 규약·최선 항로를 같은 난수로 — 항로 자유도의 폭(부록 A.7.4)
항로 비교는 반드시 같은 실행 안에서 한다(mc3 규약 vs mc3 최선).
각 파일의 계획 순서는 [중심 경유(W, 본 표), 최근접 진입(W, 참고), 최선 간격(중심 경유)] 다
(``tools/runs/run_continuous_mc.py``, 2026-10-08).
"""
import glob, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from cpp_search.common.config import load_chapter_config
from cpp_search.common.options import RunOptions
from cpp_search.experiments.ch2_stone_spx import (
    build_instance, case_mission, case_specs, case_target_specs)

SEED, H = 20260201, 600.0
cfg = load_chapter_config("2")
m0, sensor = cfg.mission(), cfg.sensor()
base = next(g for g in cfg.get("stone_spx.grid_conditions", []) if g.get("grid_kind") == "square")

def stay_mass(inst, k):
    """전이행렬의 대각 질량 — 한 슬라이스에 같은 셀에 남을 확률(질량 가중)."""
    tm = inst.target_models[k]
    mass = np.asarray(tm.initial_mass, float).copy()
    total, weight = 0.0, 0.0
    for P in (tm.transitions if not hasattr(tm.transitions, "shape") else tm.transitions):
        A = np.asarray(P.todense() if hasattr(P, "todense") else P, float)
        total += float((mass * np.diag(A)).sum()); weight += float(mass.sum())
        mass = mass @ A
    return total / max(weight, 1e-12)

rows = []
sys.path.insert(0, str(ROOT / "tools"))
from conditions import CONDITIONS                           # noqa: E402
for case, n, _ in CONDITIONS:
    f = ROOT / f"results/studies/mc3-{case}-{n}.json"
    if not f.exists() or not (ROOT / f"results/studies/mc2-{case}-{n}.json").exists():
        continue
    d = json.load(open(f))
    block = next(c for c in case_specs(cfg) if c["name"] == case)
    targets = case_target_specs(cfg, block); mission = case_mission(m0, block)
    opts = RunOptions(sample_count=10, particle_count=5000, episode_count=1,
                      target_profile="both", seed=1, map_error=False,
                      communication_loss_probability=0.0, communication_latency_slices=0,
                      mission_time_s=H)
    inst = build_instance(cfg, opts, mission, sensor,
                          grid={**base, "grid_width": n, "grid_height": n},
                          seed=SEED, terrain_weighting=True, targets=targets)
    ratio_wc = d["sweep_width_m"] / d["cell_side_m"]
    d2 = json.load(open(ROOT / f"results/studies/mc2-{case}-{n}.json"))
    # 같은 계획인지: 셀 모형 값은 결정적이므로 자릿수까지 같아야 한다.
    assert d2["cell_model_pd"][0] == d["cell_model_pd"][0], f"{case} {n}: 계획 불일치"
    for k, t in enumerate(d["targets"]):
        t2 = d2["targets"][k]
        assert t2["target"] == t["target"]
        cell = d["cell_model_pd_per_target"][0][k]
        assert d2["entry"] == ["centre-sync", "nearest", "centre-sync"], "mc2 의 계획 순서가 다르다"
        assert d["entry"] == ["centre-sync", "nearest", "centre-sync"], "mc3 의 계획 순서가 다르다"
        rows.append({"cond": f"{case} {n}×{n}", "wc": ratio_wc, "target": t["target"],
                     "cell": cell, "conv": t2["pd"][0], "conv40": t["pd"][0],
                     "best": t["pd"][2], "nearest": t2["pd"][1],
                     "stay": stay_mass(inst, k), "lane": d["lane_m"][2]})

print(f"{'조건':<12}{'탐지폭/셀':>8} {'거동':<17}{'머무름':>7}{'셀모형':>8}"
      f"{'MC(12만)':>9}{'비':>7}   {'MC규약(4만)':>11}{'MC최선(4만)':>11}")
for r in rows:
    print(f"{r['cond']:<12}{r['wc']*100:>7.1f}% {r['target']:<17}{r['stay']:>7.3f}"
          f"{r['cell']:>8.4f}{r['conv']:>9.4f}{r['conv']/r['cell']:>7.3f}"
          f"   {r['conv40']:>11.4f}{r['best']:>11.4f}")

# 본문 표 1·2: 조건마다 각 모형에서 PD 가 가장 낮은 거동
print(f"\n{'조건':<12}{'MILP 최저 거동':<18}{'MILP':>8}{'MC 최저 거동':<18}{'MC':>8}"
      f"  {'같은 거동 MC':>10}{'비':>7}{'세 거동 평균 비':>10}")
conds, wcs, matched, mean3 = [], [], [], []
for c in dict.fromkeys(r["cond"] for r in rows):
    rs = [r for r in rows if r["cond"] == c]
    lo_cell = min(rs, key=lambda r: r["cell"]); lo_mc = min(rs, key=lambda r: r["conv"])
    conds.append(c); wcs.append(rs[0]["wc"])
    matched.append(lo_cell["conv"] / lo_cell["cell"])
    mean3.append(float(np.mean([r["conv"] / r["cell"] for r in rs])))
    print(f"{c:<12}{lo_cell['target']:<18}{lo_cell['cell']:>8.4f}{lo_mc['target']:<18}"
          f"{lo_mc['conv']:>8.4f}  {lo_cell['conv']:>10.4f}{matched[-1]:>7.3f}{mean3[-1]:>10.3f}")
print(f"상관  탐지폭/셀 ↔ 세 거동 평균 비      : {np.corrcoef(wcs, mean3)[0,1]:+.3f}")
print(f"상관  탐지폭/셀 ↔ MILP 최저 거동의 비  : {np.corrcoef(wcs, matched)[0,1]:+.3f}")

stay = np.array([r["stay"] for r in rows])
gap = np.array([r["cell"] - r["conv"] for r in rows])
gapb = np.array([r["cell"] - r["best"] for r in rows])
gapc = np.array([r["cell"] - r["nearest"] for r in rows])
print(f"\n상관  머무름 ↔ 차(중심 경유, 12만) : {np.corrcoef(stay, gap)[0,1]:+.3f}")
print(f"상관  머무름 ↔ 차(최선, 4만)  : {np.corrcoef(stay, gapb)[0,1]:+.3f}")
print(f"상관  머무름 ↔ 차(최근접 진입, 12만) : {np.corrcoef(stay, gapc)[0,1]:+.3f}")
gains = [(r["best"] - r["conv40"], r["cond"], r["target"]) for r in rows]
print(f"최선 간격의 이득: 최대 {max(gains)[0]:+.4f} ({max(gains)[1]} {max(gains)[2]}), "
      f"최소 {min(gains)[0]:+.4f} ({min(gains)[1]} {min(gains)[2]})")
print(f"\n항로 자유도의 폭 (같은 4만 실행, 최선 - 규약): "
      f"평균 {np.mean([r['best']-r['conv40'] for r in rows]):+.4f}, "
      f"최대 {max(r['best']-r['conv40'] for r in rows):+.4f}")
json.dump(rows, open(ROOT / "results/studies/gap-decomposition.json", "w"),
          ensure_ascii=False, indent=2)
print("저장: results/studies/gap-decomposition.json")
