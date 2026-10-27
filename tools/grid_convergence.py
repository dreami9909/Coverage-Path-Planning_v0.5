"""격자를 키우면 셀 모형(MILP) 값과 연속 공간(MC) 값이 **가까워지는가**.

같은 인증 계획을 두 모형으로 채점한 값을 격자 크기 순으로 늘어놓는다. 새로
계산하지 않고 저장된 결과만 모은다 (2026-10-08, 본 표 규약 = 중심 경유).

    mc2-{케이스}-{N}.json          입자 12만, 두 규약 (tools/run_continuous_mc.py)
    start-{케이스}-{N}.json        출발점을 흔들었을 때의 폭 (tools/start_probe.py)
    track-audit*-{케이스}-{N}.json 그 항적이 계획대로 났는지 (tools/track_audit.py)

비교 기준선으로 **통과 한 번의 이론 비**를 같이 적는다. 균등하게 퍼진 정지 표적 위를
직선으로 한 번 지나갈 때

    연속 모형   ∫ (1 − exp(−m·p(u))) du          측방 곡선 p 를 hazard 로 읽은 유효 폭
    셀 모형     c · (1 − exp(−m·W / c))          같은 소인을 셀 폭 c 에 고르게 편 것

이고 둘의 비는 셀 한 변 c 만의 함수다 (m 은 표적 배율). 소인을 고르게 펴는 것이
지수 법칙에서 가장 유리하므로(Jensen) 이 비는 c 가 탐지폭보다 충분히 클 때 1 보다
작다. 계획이 차선을 **붙여** 깔면 띠가 서로 메워져 비가 1 로 올라가고, 같은 띠를
**겹쳐** 지나가거나 정지 표적이 쓸지 않은 자리에 남으면 이 값 아래로도 내려간다
(4x4 처럼 셀이 큰 격자). 그래서 이 값은 한계가 아니라 **눈금**이다 — 통과들이 서로
돕지도 방해하지도 않을 때의 비.

실행:  python3 tools/grid_convergence.py
저장:  results/studies/grid-convergence.json
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.experiments.ch2_stone_spx import case_specs, signature_hazard_multipliers
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, lateral_detection

from conditions import CONDITIONS, BASELINE, REFERENCE      # noqa: E402

PLAN = "절단평면"
AUDIT_SUFFIX = {"centre-sync": "-centre-sync", "nearest": ""}   # tools/track_audit.py 의 이름


def single_pass_ratio(cell_m: float, multiplier: float) -> float:
    """통과 한 번의 (연속 / 셀 모형) 탐지 비."""
    u = np.linspace(-SUPPORT_HALF_M, SUPPORT_HALF_M, 20001)
    width = float(np.sum(1.0 - np.exp(-multiplier * lateral_detection(u))) * (u[1] - u[0]))
    return width / (cell_m * (1.0 - np.exp(-multiplier * SUPPORT_HALF_M / cell_m)))


def _load(path):
    return json.load(open(path)) if path.exists() else None


def main() -> None:
    studies = ROOT / "results/studies"
    cfg = load_chapter_config("2")
    by_profile = signature_hazard_multipliers(cfg)       # 시그니처에서 유도된 표적 배율
    multipliers = {str(c["name"]): float(by_profile[str(c["signature_profile"])])
                   for c in case_specs(cfg)}
    rows = []
    print(f"{'조건':<9}{'셀[m]':>6}{'W/셀':>6}{'간격':>7} | {'셀 모형':>7}{'중심 경유':>9}{'비':>6}"
          f"{'±흔들림':>8} | {'최근접':>7}{'비':>6}{'±흔들림':>8} | {'같은거동 비':>10}{'3거동 평균':>9}"
          f"{'통과1회':>7} | 최저 거동 (셀 / 중심 / 최근접)")
    for case, n, plan_file in CONDITIONS:
        tag = f"{case}-{n}"
        mc2 = _load(studies / f"mc2-{tag}.json")
        if mc2 is None:
            print(f"{tag:<9} (아직 없음)")
            continue
        stored = json.load(open(studies / plan_file))
        plans = {name: i for i, name in enumerate(mc2["plans"])}
        b, r = plans["중심 경유"], plans["최근접 진입"]
        assert mc2["entry"][b] == BASELINE and mc2["entry"][r] == REFERENCE, f"{tag}: 계획 순서가 다르다"
        assert mc2.get("target_stepping") == "model", f"{tag}: 표적 전개가 셀 모형과 다르다"
        names = [t["target"] for t in mc2["targets"]]
        cell = np.array(mc2["cell_model_pd_per_target"][b])
        base = np.array([t["pd"][b] for t in mc2["targets"]])
        ref = np.array([t["pd"][r] for t in mc2["targets"]])
        assert abs(cell.min() - stored["worst_pd"]) < 1e-9, f"{tag}: 계획 불일치"
        start = _load(studies / f"start-{tag}.json")
        spread = {entry: (float(next(x for x in start["rows"] if x["entry"] == entry)["jitter_worst_std"])
                          if start else float("nan")) for entry in (BASELINE, REFERENCE)}
        audit = {entry: _load(studies / f"track-audit{AUDIT_SUFFIX[entry]}-{tag}.json")
                 for entry in (BASELINE, REFERENCE)}
        late = {entry: ([a["plans"][PLAN]["late_moves"], a["plans"][PLAN]["move_slices"]]
                        if a and PLAN in a["plans"] else None) for entry, a in audit.items()}
        side = float(mc2["cell_side_m"])
        floor = single_pass_ratio(side, multipliers[case])
        k = int(cell.argmin())
        rows.append({
            "case": case, "n": n, "cell_side_m": side,
            "sweep_width_over_cell": SUPPORT_HALF_M / side, "gap": float(stored["gap"]),
            "targets": names, "cell_model_pd": cell.tolist(),
            "mc_centre_sync_pd": base.tolist(), "mc_nearest_pd": ref.tolist(),
            "ratio_centre_sync": float(base.min() / cell.min()),
            "ratio_nearest": float(ref.min() / cell.min()),
            "matched_ratio_centre_sync": float(base[k] / cell[k]),
            "matched_ratio_nearest": float(ref[k] / cell[k]),
            "mean_ratio_centre_sync": float(np.mean(base / cell)),
            "mean_ratio_nearest": float(np.mean(ref / cell)),
            "start_spread_centre_sync": spread[BASELINE], "start_spread_nearest": spread[REFERENCE],
            "single_pass_ratio": floor,
            "late_moves_centre_sync": late[BASELINE], "late_moves_nearest": late[REFERENCE],
            "particles": mc2["particles"],
        })
        print(f"{tag:<9}{side:>6.0f}{SUPPORT_HALF_M / side:>6.3f}{100 * stored['gap']:>6.2f}% | "
              f"{cell.min():>7.4f}{base.min():>9.4f}{base.min() / cell.min():>6.3f}{spread[BASELINE]:>8.4f} | "
              f"{ref.min():>7.4f}{ref.min() / cell.min():>6.3f}{spread[REFERENCE]:>8.4f} | "
              f"{base[k] / cell[k]:>10.3f}{np.mean(base / cell):>9.3f}{floor:>7.3f} | "
              f"{names[k]} / {names[int(base.argmin())]} / {names[int(ref.argmin())]}")
    if rows:
        wc = np.array([r["sweep_width_over_cell"] for r in rows])
        for key, name in (("centre_sync", "중심 경유"), ("nearest", "최근접 진입")):
            mean3 = np.array([r[f"mean_ratio_{key}"] for r in rows])
            matched = np.array([r[f"matched_ratio_{key}"] for r in rows])
            print(f"[{name}] 상관 W/셀 ↔ 세 거동 평균 비 {np.corrcoef(wc, mean3)[0, 1]:+.3f}, "
                  f"↔ 같은 거동 비 {np.corrcoef(wc, matched)[0, 1]:+.3f}; 비 범위 "
                  f"{min(r[f'ratio_{key}'] for r in rows):.3f}~{max(r[f'ratio_{key}'] for r in rows):.3f}")
    dst = studies / "grid-convergence.json"
    json.dump({"plan": PLAN, "baseline": BASELINE, "reference": REFERENCE, "rows": rows},
              open(dst, "w"), ensure_ascii=False, indent=2)
    print("\n저장:", dst.name)


if __name__ == "__main__":
    main()
