"""1단계 결과를 표 하나로 모은다 (``results/studies/mappo1-*.json``).

열의 뜻
-------
    도달률      MAPPO / 절단평면 incumbent   — 인증이 닫힌 격자에서만 뜻이 곧다
    보장 하한   MAPPO / (1 - L)              — 열린 격자에서도 뜻이 흐려지지 않는다

같은 seed 가 아니라 **같은 인스턴스** 위의 학습 seed 여러 개다. 정책이 seed 에
얼마나 흔들리는지가 이 측정의 절반이므로 평균과 최소·최대를 함께 싣는다.
"""
import glob, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from conditions import CONDITIONS                           # noqa: E402
ORDER = [f"{case}-{n}" for case, n, _ in CONDITIONS]

rows = []
for path in glob.glob(str(ROOT / "results/studies/mappo1-*.json")):
    if path.endswith("summary.json"):
        continue
    d = json.load(open(path))
    tag = f"{d['case']}-{d['n']}"
    best = [r["worst_target_detection_probability"] for r in d["rows"]]
    ratio = [r["achievement_ratio"] for r in d["rows"]]
    bound = [r["certified_ratio_lower_bound"] for r in d["rows"]]
    infeasible = [r for r in d["rows"] if not r["reported_plan_audit"]["feasible"]]
    rows.append({
        "tag": tag, "case": d["case"], "n": d["n"], "cells": d["cells"],
        "time_slices": d["time_slices"], "spx": d["spx_worst_pd"],
        "certified": d["spx_certified"], "spx_gap": d["spx_relative_gap"],
        "bound": d["achievable_pd_upper_bound"],
        "mappo_mean": sum(best) / len(best), "mappo_max": max(best), "mappo_min": min(best),
        "ratio_mean": sum(ratio) / len(ratio), "ratio_max": max(ratio),
        "bound_mean": sum(bound) / len(bound), "bound_max": max(bound),
        "seed_count": len(d["rows"]),
        "infeasible_plans": len(infeasible),
        "mappo_runtime_mean_s": sum(r["runtime_s"] for r in d["rows"]) / len(d["rows"]),
        "iterations": d["mappo_settings"]["iterations"],
    })
rows.sort(key=lambda r: ORDER.index(r["tag"]) if r["tag"] in ORDER else 99)

print(f"{'조건':<12}{'셀':>5}{'단계':>5}{'인증':>6}{'절단평면':>9}{'상한':>8}"
      f"{'MAPPO 평균':>11}{'최고':>8}{'도달률':>8}{'보장 하한':>10}{'seed':>6}{'위반계획':>9}")
for r in rows:
    print(f"{r['tag']:<12}{r['cells']:>5}{r['time_slices']:>5}"
          f"{'닫힘' if r['certified'] else '열림':>6}{r['spx']:>9.4f}{r['bound']:>8.4f}"
          f"{r['mappo_mean']:>11.4f}{r['mappo_max']:>8.4f}{r['ratio_mean']:>8.3f}"
          f"{r['bound_mean']:>10.3f}{r['seed_count']:>6}{r['infeasible_plans']:>9}")

dst = ROOT / "results/studies/mappo1-summary.json"
json.dump(rows, open(dst, "w"), ensure_ascii=False, indent=2)
print("\n저장:", dst.relative_to(ROOT))
