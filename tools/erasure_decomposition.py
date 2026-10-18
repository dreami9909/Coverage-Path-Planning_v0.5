"""``tools/erasure_probe.py`` 의 결과를 사다리 하나로 분해한다.

    셀 모형  =  MC(none)  +  Δ위치  +  Δ띠  +  Δ유출  +  ΔMarkov

    Δ위치   = MC(step) − MC(none)   셀 안 실제 위치 대 매번 새로 뽑은 위치
    Δ띠     = MC(mean) − MC(step)   띠로 쌓인 소인 대 셀 안에 고르게 편 소인 (Jensen)
    Δ유출   = MC(cell) − MC(mean)   항적이 그 셀에 실제로 뿌린 양 대 셀 모형이 적은 양
    ΔMarkov = 셀 모형 − MC(cell)    실제 셀 열 대 Markov 사슬 (+ 입자 표본)

네 항 모두 같은 입자 집합·같은 항적에서 나오므로 짝지어져 있다.
저장: results/studies/erasure-decomposition.json
"""
import glob, json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from conditions import CONDITIONS                           # noqa: E402
ORDER = [f"{case}-{n}" for case, n, _ in CONDITIONS]

rows = []
for tag in ORDER:
    f = ROOT / f"results/studies/erase-{tag}.json"
    if not f.exists():
        continue
    d = json.load(open(f))
    idx = {e: i for i, e in enumerate(d["erase"])}
    for k, t in enumerate(d["targets"]):
        pd = t["pd"]; cell = d["cell_model_pd_per_target"][0][k]
        r = {"cond": f"{d['case']} {d['n']}×{d['n']}", "wc": d["sweep_width_m"] / d["cell_side_m"],
             "target": t["target"], "cell_model": cell, "particles": d["particles"]}
        for e in ("none", "slice", "step", "mean", "cell"):
            r[e] = pd[idx[e]]
        r["d_pos"] = r["step"] - r["none"]; r["d_strip"] = r["mean"] - r["step"]
        r["d_spill"] = r["cell"] - r["mean"]; r["d_markov"] = cell - r["cell"]
        r["total"] = cell - r["none"]
        rows.append(r)

print(f"{'조건':<11}{'거동':<17}{'셀모형':>7}{'MC':>7} | {'Δ위치':>7}{'Δ띠':>7}{'Δ유출':>7}{'ΔMarkov':>8} | {'합':>7}{'입자':>7}")
for r in rows:
    print(f"{r['cond']:<11}{r['target']:<17}{r['cell_model']:>7.4f}{r['none']:>7.4f} | "
          f"{r['d_pos']:>+7.4f}{r['d_strip']:>+7.4f}{r['d_spill']:>+7.4f}{r['d_markov']:>+8.4f} | "
          f"{r['total']:>+7.4f}{r['particles']:>7d}")
if rows:
    stop = [r for r in rows if r["target"] == "STOP_HEAVY"]
    move = [r for r in rows if r["target"] != "STOP_HEAVY"]
    print(f"\nΔ위치  STOP_HEAVY 평균 {np.mean([r['d_pos'] for r in stop]):+.4f}   움직이는 거동 평균 {np.mean([r['d_pos'] for r in move]):+.4f}")
    print(f"Δ띠    STOP_HEAVY 평균 {np.mean([r['d_strip'] for r in stop]):+.4f}   움직이는 거동 평균 {np.mean([r['d_strip'] for r in move]):+.4f}")
    print(f"Δ띠/셀모형 (비율)  전체 {np.mean([r['d_strip']/r['cell_model'] for r in rows]):.3f}  "
          f"범위 {min(r['d_strip']/r['cell_model'] for r in rows):.3f} ~ {max(r['d_strip']/r['cell_model'] for r in rows):.3f}")
    print(f"|Δ유출| 최대 {max(abs(r['d_spill']) for r in rows):.4f}   |ΔMarkov| 최대 {max(abs(r['d_markov']) for r in rows):.4f}")
    json.dump(rows, open(ROOT / "results/studies/erasure-decomposition.json", "w"), ensure_ascii=False, indent=2)
    print("저장: results/studies/erasure-decomposition.json")
