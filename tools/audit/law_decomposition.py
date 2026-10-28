"""``tools/probes/law_probe.py`` 의 결과를 한 표로 모은다 (부록 A.7.6).

측방 곡선을 hazard 로 읽을 때(exp)와 통과당 탐지확률로 읽을 때(prob)의 모형 밖 MC 를
셀 모형 값과 나란히 둔다. 같은 계획·같은 입자·같은 항적이다.
저장: results/studies/law-decomposition.json
"""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from conditions import CONDITIONS                           # noqa: E402

rows = []
for case, n, _ in CONDITIONS:
    f = ROOT / f"results/studies/law-{case}-{n}.json"
    if not f.exists():
        continue
    d = json.load(open(f))
    idx = {law: i for i, law in enumerate(d["law"])}
    for k, t in enumerate(d["targets"]):
        rows.append({"cond": f"{case} {n}×{n}", "wc": d["sweep_width_m"] / d["cell_side_m"],
                     "target": t["target"], "cell": d["cell_model_pd_per_target"][0][k],
                     "exp": t["pd"][idx["exponential"]], "prob": t["pd"][idx["probability"]]})

print(f"{'조건':<12}{'거동':<18}{'셀 모형':>8}{'hazard 읽기':>12}{'비':>7}{'확률 읽기':>11}{'비':>7}")
for r in rows:
    print(f"{r['cond']:<12}{r['target']:<18}{r['cell']:>8.4f}{r['exp']:>12.4f}"
          f"{r['exp'] / r['cell']:>7.3f}{r['prob']:>11.4f}{r['prob'] / r['cell']:>7.3f}")
print(f"\n{'조건':<12}{'셀 모형 최저':>10}{'hazard 최저':>12}{'비':>7}{'확률 최저':>10}{'비':>7}")
for cond in dict.fromkeys(r["cond"] for r in rows):
    rs = [r for r in rows if r["cond"] == cond]
    cell, expo, prob = (min(r[k] for r in rs) for k in ("cell", "exp", "prob"))
    print(f"{cond:<12}{cell:>10.4f}{expo:>12.4f}{expo / cell:>7.3f}{prob:>10.4f}{prob / cell:>7.3f}")
if rows:
    over = sum(r["prob"] > r["cell"] for r in rows)
    print(f"\n확률 읽기가 셀 모형을 넘는 (조건, 거동): {over} / {len(rows)}")
    json.dump(rows, open(ROOT / "results/studies/law-decomposition.json", "w"),
              ensure_ascii=False, indent=2)
    print("저장: results/studies/law-decomposition.json")
