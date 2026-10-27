"""탐지함수를 **포화형**으로 바꾸면 셀 모형이 계획을 구별하는가 (4.8절).

셀 모형은 지수 탐지함수 ``1 − exp(−a)`` 를 쓴다. 무작위탐색 전제다. 그렇다면
"셀을 완전히 덮으면 반드시 찾는다"는 전제, 즉 포화형 ``min(1, a)`` 로 바꾸면
계획 사이의 차이가 드러날까? 같은 다섯 계획(4.8절, 그림 7 나)을 두 법칙으로 다시
채점해 순위를 견준다.

    지수형   생존 = exp(−m·Y)        셀 모형이 쓰는 법칙
    포화형   생존 = max(0, 1 − m·Y)  셀을 덮은 몫만큼 반드시 찾는다

``Y`` 는 슬라이스마다 셀에 쌓인 소인량이고 ``m`` 은 표적 배율이다. 둘 다 같은
재귀식(사전분포 → 깎기 → 전이)을 지나며 법칙만 바뀐다. 지수형으로 돌린 값이
``instance.score_paths`` 와 자릿수까지 같은지 매 실행 확인한다 — 다르면 재귀식을
잘못 옮긴 것이므로 포화형 값도 믿을 수 없다.

실행:  python3 tools/law_rank_probe.py
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.experiments.ch2_stone_spx import case_specs
from cpp_search.experiments.ch2_studies import published_instance
from cpp_search.theory import transitions as transition_ops
from cpp_search.theory.path_constrained import joint_survival

from rank_probe import collect_plans                      # noqa: E402

SEED, CASE, WIDTH = 20260201, "Tank", 10
TIE = 1e-6          # 이 안이면 같은 값으로 본다


def detection_probabilities(instance, paths, law: str) -> list[float]:
    """표적별 탐지확률. ``law`` 만 다르고 재귀식은 같다."""

    problem = instance.problem
    # 슬라이스별 셀 hazard (표적 배율 전). 셀 모형이 그 셀에 적는 양이다.
    hazard = np.array([-np.log(joint_survival(problem, paths, t))
                       for t in range(problem.time_count)])
    out = []
    for target in instance.target_models:
        multiplier = float(np.asarray(target.hazard_multiplier, dtype=float))
        transitions = transition_ops.as_sequence(target.transitions)
        mass = np.asarray(target.initial_mass, dtype=float).copy()
        for t in range(problem.time_count):
            applied = multiplier * hazard[t]
            if law == "exponential":
                survival = np.exp(-applied)
            elif law == "saturating":
                survival = np.clip(1.0 - applied, 0.0, 1.0)
            elif law == "cover":
                # 극단: 소인이 조금이라도 닿은 셀은 반드시 찾는다
                survival = np.where(applied > 1e-12, 0.0, 1.0)
            else:
                raise ValueError(f"unknown law {law!r}")
            mass = mass * survival
            if t < problem.time_count - 1:
                mass = mass @ transitions[t]
        out.append(float(1.0 - mass.sum()))
    return out


def main() -> None:
    cfg = load_chapter_config("2")
    block = next(c for c in case_specs(cfg) if c["name"] == CASE)
    instance = published_instance(cfg, case=block, width=WIDTH, seed=SEED)
    plans = collect_plans()

    rows = []
    for name, paths in plans:
        expo = detection_probabilities(instance, paths, "exponential")
        official = list(instance.score_paths(paths))
        drift = max(abs(a - b) for a, b in zip(expo, official))
        if drift > 1e-9:
            raise SystemExit(f"지수형 재귀식이 score_paths 와 다르다: {drift:.2e}")
        sat = detection_probabilities(instance, paths, "saturating")
        cover = detection_probabilities(instance, paths, "cover")
        rows.append({"plan": name, "exponential": min(expo), "saturating": min(sat),
                     "cover": min(cover), "exponential_per_target": expo,
                     "saturating_per_target": sat, "cover_per_target": cover,
                     "recursion_drift": drift})

    # 모형 밖 MC 순위 (같은 다섯 계획, tools/rank_probe.py 가 저장한 것)
    mc_path = ROOT / "results/studies/mc2-rank-Tank-10.json"
    mc = json.load(open(mc_path)) if mc_path.exists() else None
    mc_worst = (np.array([t["pd"] for t in mc["targets"]]).min(axis=0)
                if mc else np.full(len(rows), np.nan))

    expo = np.array([r["exponential"] for r in rows])
    sat = np.array([r["saturating"] for r in rows])
    cover = np.array([r["cover"] for r in rows])
    def rank(values):
        return np.argsort(np.argsort(-values)) + 1
    print(f"[{CASE} {WIDTH}×{WIDTH}]  계획 {len(rows)}개 (재귀식 일치 확인 완료)")
    print(f"  {'계획':<14}{'지수형':>9}{'순위':>5}{'포화형':>9}{'순위':>5}"
          f"{'덮개형':>9}{'순위':>5}{'연속 MC':>9}{'순위':>5}")
    for i, r in enumerate(rows):
        print(f"  {r['plan']:<14}{expo[i]:>9.4f}{rank(expo)[i]:>5}{sat[i]:>9.4f}"
              f"{rank(sat)[i]:>5}{cover[i]:>9.4f}{rank(cover)[i]:>5}"
              f"{mc_worst[i]:>9.4f}{rank(mc_worst)[i]:>5}")

    def tie_report(values, label):
        ties = sum(1 for i in range(len(values)) for j in range(i + 1, len(values))
                   if abs(values[i] - values[j]) < TIE)
        distinct = len({round(float(v) / TIE) for v in values})
        print(f"  {label}: 서로 다른 값 {distinct} 개 (동점 쌍 {ties}), "
              f"폭 {values.max()-values.min():.4f}")
    tie_report(sat, "포화형"); tie_report(cover, "덮개형")
    best_sat = int(np.argmax(sat))
    print(f"  지수형 폭 {expo.max()-expo.min():.4f} | 연속 MC 폭 "
          f"{np.nanmax(mc_worst)-np.nanmin(mc_worst):.4f}")
    if mc:
        best_cover = int(np.argmax(cover))
        print(f"  포화형 최선({rows[best_sat]['plan']})의 연속 순위 {rank(mc_worst)[best_sat]}"
              f" | 덮개형 최선({rows[best_cover]['plan']})의 연속 순위 {rank(mc_worst)[best_cover]}")

    dst = ROOT / "results/studies/law-rank-Tank-10.json"
    json.dump({"case": CASE, "n": WIDTH, "seed": SEED, "tie_tolerance": TIE,
               "rows": rows,
               "mc_worst": [float(v) for v in mc_worst] if mc else None},
              open(dst, "w"), ensure_ascii=False, indent=2)
    print("  저장:", dst.name)


if __name__ == "__main__":
    main()
