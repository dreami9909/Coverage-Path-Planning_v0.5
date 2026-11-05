"""센서 가정 민감도 스윕 — 실비행 MC 탐지모델 계수를 문헌 범위 양끝으로.

    PYTHONPATH=src python3 tools/probes/sensor_sensitivity_sweep.py

demo_custom([truth] 가설 2개)을 바탕으로, 계획은 base 변형에서 한 번 풀고
나머지 변형은 plan.json 재사용 — 변형 간 차이는 순수하게 탐지모델
파라미터(algorithm_efficiency·위장 등급)의 몫이다. OAT(한 번에 한 축),
축 양끝은 문헌 보고 범위(노션 '센서 파라미터 — 근거와 민감도' 절).

산출: results/studies/sensor-sensitivity.json (+ 변형별 run 디렉터리)
실측 (2026-11-05): 최악 가설(은닉 정지형) PD 0.020 이 7변형 전수 동일
— 센서 가정 무감, 커버리지 기하 지배. 회랑 가설만 0.110~0.160 이동,
지배 축은 위장 등급(전폭 0.050), 효율 축은 ±0.010.
"""
import json
import shutil
from dataclasses import replace
from pathlib import Path

import cpp_search.pipeline.run as runm
from cpp_search.pipeline.scenario import load_scenario
from cpp_search.models.sensor_observation import (
    CAMOUFLAGE_HEAVY,
    CAMOUFLAGE_NOMINAL,
    CAMOUFLAGE_NONE,
    SENSOR_PERFORMANCE_SR_Z50,
    SpatialDetectionModel,
)

ROOT = Path("results/studies/sensor-sensitivity-runs")
SCEN = "config/scenarios/demo_custom.toml"
OUT = Path("results/studies/sensor-sensitivity.json")
BASE_SP = SENSOR_PERFORMANCE_SR_Z50


def sp_with(eo_eff=None, ir_eff=None):
    sp = BASE_SP
    if eo_eff is not None:
        sp = replace(sp, eo=replace(sp.eo, algorithm_efficiency=eo_eff))
    if ir_eff is not None:
        sp = replace(sp, ir=replace(sp.ir, algorithm_efficiency=ir_eff))
    return sp


# OAT — 기준(채택값) + 축별 문헌 범위 양끝
VARIANTS = {
    "base":        (BASE_SP, CAMOUFLAGE_NOMINAL),
    "camo_none":   (BASE_SP, CAMOUFLAGE_NONE),
    "camo_heavy":  (BASE_SP, CAMOUFLAGE_HEAVY),
    "eo_eff_0.80": (sp_with(eo_eff=0.80), CAMOUFLAGE_NOMINAL),
    "eo_eff_0.99": (sp_with(eo_eff=0.99), CAMOUFLAGE_NOMINAL),
    "ir_eff_0.70": (sp_with(ir_eff=0.70), CAMOUFLAGE_NOMINAL),
    "ir_eff_0.95": (sp_with(ir_eff=0.95), CAMOUFLAGE_NOMINAL),
}


def main() -> None:
    scenario = load_scenario(SCEN)
    results = {}
    base_plan = ROOT / "base" / scenario.name / "plan.json"

    for name, (sp, camo) in VARIANTS.items():
        out_root = ROOT / name
        run_dir = out_root / scenario.name
        run_dir.mkdir(parents=True, exist_ok=True)
        if name != "base" and base_plan.exists():
            shutil.copy(base_plan, run_dir / "plan.json")

        def patched(**kwargs):
            kwargs["sensor_performance"] = sp
            kwargs["camouflage"] = camo
            return SpatialDetectionModel(**kwargs)

        runm.SpatialDetectionModel = patched
        print(f"\n===== {name} =====", flush=True)
        metrics = runm.run_scenario(scenario, out_root=out_root, reuse_plan=True)
        mc = metrics["mc"]
        results[name] = {
            "kind": mc.get("kind"),
            "by_target": {
                k: v["detection_probability"] for k, v in mc["by_target"].items()
            },
            "mean": mc.get("detection_probability_mean"),
            "worst": mc.get("detection_probability_worst"),
            "worst_target": mc.get("detection_probability_worst_target"),
            "model_by_target": metrics["solver"].get(
                "model_detection_probability_by_target"
            ),
        }

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n==== 요약 ====")
    for name, r in results.items():
        bt = " ".join(f"{k}:{v:.4f}" for k, v in r["by_target"].items())
        print(f"{name:14s} mean={r['mean']:.4f} "
              f"worst={r['worst']:.4f}({r['worst_target']}) | {bt}")
    print(f"\n산출: {OUT}")


if __name__ == "__main__":
    main()
