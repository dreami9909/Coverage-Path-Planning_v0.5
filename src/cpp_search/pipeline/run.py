"""시나리오 하나를 Input -> Model -> Solver -> Output 으로 돌린다.

실행 흐름은 Chapter 4 의 평가 루프와 같다 — 계획(모형 PD·인증)과
실비행(Monte Carlo PD)을 **같은 인스턴스**에서 재서, 계획모형과 실비행의
차이(``model_minus_flown``)까지 결과에 남긴다. 함수는 전부 실험 챕터의
것을 재사용한다. 여기서 새 모형을 만들지 않는다.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from cpp_search.experiments.ch2_stone_spx import (
    IMM5_BEHAVIORS,
    case_mission,
    case_specs,
    case_target_specs,
    launch_positions,
    resolved_grid,
    route_config,
    searcher_classes,
    signature_hazard_multipliers,
)
from cpp_search.models.evaluation import evaluate_routes
from cpp_search.models.models import Point2D
from cpp_search.models.sensor_observation import (
    SENSOR_PERFORMANCE_SR_Z50,
    SpatialDetectionModel,
)
from cpp_search.models.simulation import EvaluationConfig, evaluate_monte_carlo
from cpp_search.models.terrain import build_declared_terrain, build_synthetic_terrain
from cpp_search.reporting.kpi import mission_kpi
from cpp_search.pipeline.scenario import CUSTOM_BEHAVIOUR_NAME, Scenario, ScenarioError
from cpp_search.solvers.milp.stone_spx import (
    StoneSPXTargetSpec,
    build_stone_grid_instance,
    route_plan_from_solution,
    solve_stone_grid_paths,
)
from cpp_search.models.truth import behaviour_profile, cue_prior, generate_truth, target_profile


def run_scenario(
    scenario: Scenario,
    *,
    out_root: str | Path = "results/runs",
    reuse_plan: bool = True,
) -> dict:
    """시나리오를 실행하고 ``<out_root>/<name>/`` 에 산출물을 남긴다.

    ``reuse_plan`` 이면 같은 입력(fingerprint 일치)의 저장된 계획
    (``plan.json``)을 다시 풀지 않고 쓴다 — solve 가 실행마다 다른 계획을
    낼 수 있어(비결정성 관측 2026-11-03: 같은 입력이 gap 4.07%→0.91%→0.99%)
    공표 실행의 재현에는 계획 자체를 보관해야 한다.
    """

    started = time.time()
    config = scenario.chapter_config()
    options = scenario.run_options()

    # ---------------------------------------------------------------- Input
    case = _select_case(config, scenario.target.case)
    mission = case_mission(config.mission(), case)
    if scenario.target.aoi_radius_m is not None:
        mission = replace(mission, search_radius_m=scenario.target.aoi_radius_m)
    # AOI 가 해소됐으니 명시 입력의 좌표를 지금 검증한다 — solver 전 거부.
    scenario.validate_geometry(mission.search_radius_m)
    sensor = config.sensor()
    targets = case_target_specs(config, case)
    if scenario.behaviour is not None:
        # 케이스 최악 거동 집합에 custom 가설을 **추가**한다 (설계 확정
        # 2026-10-31) — 보수성은 유지하고 "내 거동이 실제였다면"도 채점된다.
        targets = (*targets, _custom_target_spec(config, case))

    base_grid = _base_grid_declaration(config)
    grid = resolved_grid(
        mission, sensor, scenario.grid_declaration(base_grid), options.mission_time_s
    )
    plan_config = route_config(
        grid, options=options, terrain_weighting=scenario.terrain.weighting
    )

    prior = cue_prior(config, mission.search_radius_m)[0]
    if scenario.layout is not None:
        terrain = build_declared_terrain(
            mission.search_radius_m,
            corridors=tuple(
                (c.start, c.end, c.width_m) for c in scenario.layout.corridors
            ),
            barriers=tuple(
                (c.center, c.radius_m) for c in scenario.layout.obstacles
            ),
            concealments=tuple(
                (c.center, c.radius_m, c.strength)
                for c in scenario.layout.concealments
            ),
            background=scenario.terrain.background,
        )
    else:
        terrain = build_synthetic_terrain(
            mission.search_radius_m,
            seed=scenario.seed,
            corridor_count=scenario.terrain.corridor_count,
            barrier_count=scenario.terrain.barrier_count,
            concealment_count=scenario.terrain.concealment_count,
            background=scenario.terrain.background,
        )
    positions = launch_positions(
        mission,
        ring_ratio=float(config.get("stone_spx.initial_position_ring_ratio", 0.25)),
    )

    # ---------------------------------------------------------------- Model
    instance = build_stone_grid_instance(
        mission,
        sensor,
        prior=prior,
        terrain=terrain,
        targets=targets,
        searcher_classes=searcher_classes(config, mission),
        initial_positions=positions,
        planning_time_s=options.mission_time_s,
        initial_belief_delay_s=scenario.mission.ingress_s,
        seed=scenario.seed,
        config=plan_config,
    )

    # --------------------------------------------------------------- Solver
    out_dir = Path(out_root) / scenario.name
    plan_path = out_dir / "plan.json"
    solution = _cached_solution(plan_path, instance.fingerprint) if reuse_plan else None
    plan_reused = solution is not None
    if solution is None:
        solution = solve_stone_grid_paths(instance, method="stone-spx")
        _store_solution(plan_path, instance.fingerprint, solution)
    plan = route_plan_from_solution(instance, solution, method="stone-spx")
    diagnostics = plan.diagnostics

    # --------------------------------------------------------------- Output
    geometry = evaluate_routes(
        plan.routes, mission, sensor, time_limit_s=options.mission_time_s
    )
    # MC 열은 규약 맞춤 연속 MC 다 (B안, 2026-11-03 확정) — 셀 모형과 항적
    # 규약·소인 법칙·이송 처리를 맞춰서 model−MC 가 곧 셀 모형 가정(주로
    # Markov 섞임)의 몫이 되게 한다. [truth] 고정 출발점 가설만 예외로
    # 실비행 MC 로 채점한다 (고정 initial_points 를 지원하는 유일한 장치).
    if scenario.hypotheses:
        mc = _flight_hypotheses_block(
            scenario, config, case, targets, plan, mission, sensor,
            terrain, prior, options, geometry,
        )
    else:
        mc = _continuous_mc_block(
            scenario, instance, plan, targets, mission, prior, options
        )

    model_by_target = {
        name: float(value)
        for name, value in zip(
            instance.target_names, solution.target_detection_probabilities
        )
    }
    # 최악값에는 **누가** 최악인지를 항상 같이 남긴다 — 모형과 실비행이 서로
    # 다른 거동을 최악으로 지목하는 일이 흔하다 (profile-rank 연구, 2026-10-31).
    model_worst_target = min(model_by_target, key=model_by_target.get)
    model_worst = model_by_target[model_worst_target]
    # model−MC: 규약 맞춤 MC 는 minimax 끼리(최악 vs 최악) 비교가 성립하고,
    # 실비행 가설 채점은 v0.4 규약대로 등가중 평균과 비교한다.
    if mc["kind"] == "continuous-centre-sync":
        mc["model_minus_mc"] = model_worst - mc["detection_probability_worst"]
    else:
        mc["model_minus_mc"] = model_worst - mc["detection_probability_mean"]
    certified = bool(
        np.isfinite(diagnostics.relative_optimality_gap)
        and diagnostics.relative_optimality_gap <= scenario.solver.certificate_gap
    )

    metrics: dict[str, Any] = {
        "scenario": scenario.snapshot(),
        "provenance": {
            # 신설 입력 블록 사용 여부 — 하나라도 쓰면 20260201~03 anchor 와
            # 다른 세계이므로 v0.4 인증 실적 표에 나란히 싣지 않는다.
            "custom_inputs": list(scenario.custom_inputs),
            "comparable_to_v04_anchors": not scenario.custom_inputs,
        },
        "resolved": {
            "aoi_radius_m": mission.search_radius_m,
            "grid_kind": diagnostics.grid_kind,
            "grid_shape": list(diagnostics.grid_shape),
            "cell_area_m2": diagnostics.cell_area_m2,
            "cell_m": float(np.sqrt(diagnostics.cell_area_m2)),
            "time_slice_count": instance.time_slice_count,
            "slice_s": instance.slice_s,
            "effective_sweep_width_m": diagnostics.effective_sweep_width_m,
            "launch_positions": [[p.x, p.y] for p in positions],
            "targets": list(instance.target_names),
            "common_input_fingerprint": diagnostics.common_input_fingerprint,
        },
        "solver": {
            "kind": scenario.solver.kind,
            "plan_reused": plan_reused,
            "model_detection_probability_worst": model_worst,
            "model_detection_probability_worst_target": model_worst_target,
            "model_detection_probability_by_target": model_by_target,
            "relative_optimality_gap": float(diagnostics.relative_optimality_gap),
            "certified": certified,
            "required_relative_gap": scenario.solver.certificate_gap,
            "lower_bound_nondetection": diagnostics.lower_bound_nondetection,
            "upper_bound_nondetection": diagnostics.upper_bound_nondetection,
            "iterations": diagnostics.iterations,
            "converged": diagnostics.converged,
            "runtime_s": diagnostics.runtime_s,
        },
        "geometry": {
            # 커버리지는 경로 기하라 가설과 무관 — MC 장치가 바뀌어도 그대로다.
            "unique_area_coverage_ratio": _maybe_float(
                geometry.unique_area_coverage_ratio
            ),
            "probability_mass_coverage": _maybe_float(
                geometry.probability_mass_coverage
            ),
            "total_distance_m": float(geometry.total_distance_m),
        },
        "mc": mc,
        "wall_time_s": None,  # 아래에서 채움
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    from cpp_search.pipeline.figures import weight_figure, track_figure

    figure_paths = [
        weight_figure(instance, terrain, out_dir / "figure1_weights.png"),
        track_figure(instance, plan, out_dir / "figure2_tracks.png"),
    ]
    metrics["figures"] = [str(path) for path in figure_paths]
    metrics["wall_time_s"] = time.time() - started

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1, default=_json_default),
        encoding="utf-8",
    )
    if scenario.source_path is not None:
        (out_dir / "scenario.toml").write_text(
            Path(scenario.source_path).read_text(encoding="utf-8"), encoding="utf-8"
        )
    return metrics


_SOLUTION_FIELDS = (
    "paths", "target_detection_probabilities", "lower_bound_nondetection",
    "upper_bound_nondetection", "relative_optimality_gap", "iterations",
    "converged", "fallback_used", "fallback_reason", "runtime_s",
    "incumbent_source",
)


def _store_solution(path: Path, fingerprint: str, solution) -> None:
    """solve 결과 중 계획 재구성에 필요한 필드만 ``plan.json`` 에 남긴다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "solution": {
            name: getattr(solution, name) for name in _SOLUTION_FIELDS
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=_json_default),
        encoding="utf-8",
    )


def _cached_solution(path: Path, fingerprint: str):
    """저장된 계획이 **같은 인스턴스**의 것일 때만 되살린다.

    fingerprint 가 다르면 입력이 바뀐 것 — 조용히 다시 푼다. 반복별
    bound_history 류는 저장하지 않으므로 재사용 실행에는 비어 있다
    (파이프라인 지표·그림은 그것을 쓰지 않는다).
    """

    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if stored.get("fingerprint") != fingerprint:
        return None
    from cpp_search.solvers.milp.stone_spx import StoneGridPathSolution

    raw = stored["solution"]
    return StoneGridPathSolution(
        paths=tuple(tuple(int(cell) for cell in p) for p in raw["paths"]),
        target_detection_probabilities=tuple(
            float(v) for v in raw["target_detection_probabilities"]
        ),
        lower_bound_nondetection=float(raw["lower_bound_nondetection"]),
        upper_bound_nondetection=float(raw["upper_bound_nondetection"]),
        relative_optimality_gap=float(raw["relative_optimality_gap"]),
        iterations=int(raw["iterations"]),
        converged=bool(raw["converged"]),
        fallback_used=bool(raw["fallback_used"]),
        fallback_reason=raw["fallback_reason"],
        runtime_s=float(raw["runtime_s"]),
        incumbent_source=str(raw.get("incumbent_source", "unknown")),
    )


def _continuous_mc_block(
    scenario: Scenario, instance, plan, targets, mission, prior, options
) -> dict:
    """규약 맞춤 연속 MC — 파이프라인의 MC 열 (B안, 2026-11-03 확정).

    셀 모형과 항적 규약(centre-sync)·지수 소인 법칙·이송 처리(belief 480 s
    predict 만)를 전부 맞추고, 표적이 셀 안 실제 위치를 기억한다는 것만
    현실로 되돌린다. 그래서 model−MC 가 곧 셀 모형 가정(주로 Markov 섞임)의
    몫이다. 끝단 실비행 평가는 [truth] 고정 출발점 가설 경로에만 남는다.
    """

    from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M
    from cpp_search.solvers.milp.continuous_mc import evaluate as continuous_evaluate
    from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL

    paths = tuple(tuple(int(cell) for cell in path) for path in plan.assignments)
    result = continuous_evaluate(
        instance,
        targets,
        [("centre-sync", paths, SUPPORT_HALF_M)],
        mission=mission,
        prior=prior,
        terrain_coupling=BELIEF_TERRAIN_FULL,
        particle_count=options.particle_count,
        dt_s=5.0,
        seed=scenario.seed,
        erase=("none",),
        entry=("centre-sync",),
        belief_delay_s=scenario.mission.ingress_s,
    )
    by_target = {
        spec.profile.name: {
            "detection_probability": float(row["pd"][0]),
            "failure_rate": 1.0 - float(row["pd"][0]),
        }
        for spec, row in zip(targets, result["targets"])
    }
    values = {
        name: block["detection_probability"] for name, block in by_target.items()
    }
    worst_target = min(values, key=values.get)
    return {
        "kind": "continuous-centre-sync",
        "particles": options.particle_count,
        "by_target": by_target,
        "detection_probability_mean": float(np.mean(list(values.values()))),
        "detection_probability_worst": values[worst_target],
        "detection_probability_worst_target": worst_target,
    }


def _flight_hypotheses_block(
    scenario: Scenario, config, case, targets, plan, mission, sensor,
    terrain, prior, options, geometry,
) -> dict:
    """[truth] 고정 출발점 가설의 채점 — 실비행 MC.

    고정 initial_points 를 지원하는 유일한 장치라 이 경로만 실비행으로
    남는다. 공표 MC(규약 맞춤 연속)와 장치가 다르므로 kind 로 구분한다.
    """

    evaluation = EvaluationConfig(
        sample_count=options.sample_count,
        seed=scenario.seed ^ 0x4556_414C,
        detection_time_limit_s=options.mission_time_s,
    )
    strata = _evaluation_strata(scenario, config, case, targets)
    flight_blocks: list[dict] = []
    for index, (label, profile, start) in enumerate(strata):
        truths = generate_truth(
            config,
            mission,
            replace(evaluation, seed=evaluation.seed ^ (index * 0x9E37)),
            profile,
            horizon_s=options.mission_time_s,
            terrain=terrain,
            prior=prior,
            initial_points=(
                [Point2D(*start)] * options.sample_count if start is not None else None
            ),
        )
        performance = evaluate_monte_carlo(
            plan.routes,
            mission,
            sensor,
            truths,
            evaluation,
            detection_model=SpatialDetectionModel(
                condition="sensor",
                sensor_performance=SENSOR_PERFORMANCE_SR_Z50,
                target_signature=profile.signature,
            ),
            sensor_detection_seed=scenario.seed ^ 0x5345_4E53 ^ (index * 0x9E37),
        )
        block = mission_kpi(
            performance, uav_count=mission.uav_count, geometry=geometry
        )
        block["target_profile"] = label
        flight_blocks.append(block)

    by_target = {
        block["target_profile"]: {
            "detection_probability": float(
                block["detection_probability_within_limit"]
            ),
            "failure_rate": 1.0
            - float(block["detection_probability_within_limit"]),
            "conditional_mean_detection_time_s": _maybe_float(
                block.get("conditional_mean_detection_time_s")
            ),
            "restricted_mean_detection_time_s": _maybe_float(
                block.get("restricted_mean_detection_time_s")
            ),
        }
        for block in flight_blocks
    }
    values = {
        name: block["detection_probability"] for name, block in by_target.items()
    }
    worst_target = min(values, key=values.get)
    return {
        "kind": "flight-hypotheses",
        "monte_carlo_samples": options.sample_count,
        "by_target": by_target,
        "detection_probability_mean": float(np.mean(list(values.values()))),
        "detection_probability_worst": values[worst_target],
        "detection_probability_worst_target": worst_target,
    }


def _custom_target_spec(config, case: dict) -> StoneSPXTargetSpec:
    """[target.behaviour] 의 custom 프로파일을 평가 가설 하나로 포장한다.

    base 프로파일은 config 오버레이(assignment=CUSTOM)를 거쳐 이미 custom
    거동을 달고 있다. 배율 규칙은 ``case_target_specs`` 와 동일하게 따른다.
    """

    base = target_profile(config, str(case["signature_profile"]))
    from_signature = bool(config.get("stone_spx.target_hazard_from_signature", False))
    multiplier = (
        signature_hazard_multipliers(config)[base.name] if from_signature else 1.0
    )
    base_name = str(
        config.common_get("target_motion.custom_behaviour_base", "?")
    )
    profile = replace(base, name=f"{base.name}/CUSTOM(base={base_name})")
    return StoneSPXTargetSpec(profile, multiplier)


def _evaluation_strata(
    scenario: Scenario, config, case: dict, targets
) -> list[tuple[str, Any, tuple[float, float] | None]]:
    """실비행 평가가 돌 가설 목록: (표기 이름, 프로파일, 시작좌표 | None).

    [truth] hypotheses 가 없으면 현행 그대로 — 계획 인스턴스의 표적 집합
    (custom 포함)을 사전분포 추첨 출발로 평가한다. 있으면 **평가 쪽만**
    그 가설들로 바뀐다 (계획·인증은 그대로 — 설계 의미 규칙).
    """

    if not scenario.hypotheses:
        return [(spec.profile.name, spec.profile, None) for spec in targets]

    base = target_profile(config, str(case["signature_profile"]))
    strata: list[tuple[str, Any, tuple[float, float] | None]] = []
    for hypothesis in scenario.hypotheses:
        name = hypothesis.behaviour
        if name == "custom":
            base_name = str(
                config.common_get("target_motion.custom_behaviour_base", "?")
            )
            profile = replace(base, name=f"{base.name}/CUSTOM(base={base_name})")
        elif name in IMM5_BEHAVIORS:
            profile = replace(
                base, imm5_behavior=IMM5_BEHAVIORS[name], name=f"{base.name}/{name}"
            )
        else:
            try:
                declared = behaviour_profile(config, name)
            except KeyError:
                raise ScenarioError(
                    f"truth.hypotheses 의 거동 {name!r} 은 코드 레지스트리에도 "
                    f"config behaviour_profiles 에도 없다 — "
                    f"코드: {sorted(IMM5_BEHAVIORS)}"
                ) from None
            profile = replace(base, imm5_behavior=declared, name=f"{base.name}/{name}")
        label = f"{name}@({hypothesis.x:.0f},{hypothesis.y:.0f})"
        strata.append((label, profile, (hypothesis.x, hypothesis.y)))
    return strata


def _select_case(config, case_name: str) -> dict:
    for case in case_specs(config):
        if str(case.get("name", "")).lower() == case_name.lower():
            return dict(case)
    raise KeyError(
        f"config 의 stone_spx.cases 에 {case_name!r} 케이스가 없다 — "
        f"선언된 케이스: {[c.get('name') for c in case_specs(config)]}"
    )


def _base_grid_declaration(config) -> dict:
    """config 가 선언한 첫 square 격자를 기본 한도 창고로 쓴다."""

    for grid in config.get("stone_spx.grid_conditions", []) or []:
        if str(grid.get("grid_kind", "square")) == "square":
            return dict(grid)
    return {}


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)
