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
    case_mission,
    case_specs,
    case_target_specs,
    launch_positions,
    resolved_grid,
    route_config,
    searcher_classes,
)
from cpp_search.models.evaluation import evaluate_routes
from cpp_search.models.sensor_observation import (
    SENSOR_PERFORMANCE_SR_Z50,
    SpatialDetectionModel,
)
from cpp_search.models.simulation import EvaluationConfig, evaluate_monte_carlo
from cpp_search.models.terrain import build_synthetic_terrain
from cpp_search.reporting.kpi import mission_kpi
from cpp_search.pipeline.scenario import Scenario
from cpp_search.solvers.milp.stone_spx import (
    build_stone_grid_instance,
    route_plan_from_solution,
    solve_stone_grid_paths,
)
from cpp_search.models.truth import cue_prior, generate_truth


def run_scenario(scenario: Scenario, *, out_root: str | Path = "results/runs") -> dict:
    """시나리오를 실행하고 ``<out_root>/<name>/`` 에 산출물을 남긴다."""

    started = time.time()
    config = scenario.chapter_config()
    options = scenario.run_options()

    # ---------------------------------------------------------------- Input
    case = _select_case(config, scenario.target.case)
    mission = case_mission(config.mission(), case)
    if scenario.target.aoi_radius_m is not None:
        mission = replace(mission, search_radius_m=scenario.target.aoi_radius_m)
    sensor = config.sensor()
    targets = case_target_specs(config, case)

    base_grid = _base_grid_declaration(config)
    grid = resolved_grid(
        mission, sensor, scenario.grid_declaration(base_grid), options.mission_time_s
    )
    plan_config = route_config(
        grid, options=options, terrain_weighting=scenario.terrain.weighting
    )

    prior = cue_prior(config, mission.search_radius_m)[0]
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
    solution = solve_stone_grid_paths(instance, method="stone-spx")
    plan = route_plan_from_solution(instance, solution, method="stone-spx")
    diagnostics = plan.diagnostics

    # --------------------------------------------------------------- Output
    geometry = evaluate_routes(
        plan.routes, mission, sensor, time_limit_s=options.mission_time_s
    )
    evaluation = EvaluationConfig(
        sample_count=options.sample_count,
        seed=scenario.seed ^ 0x4556_414C,
        detection_time_limit_s=options.mission_time_s,
    )
    strata = tuple(spec.profile for spec in targets)
    flight_blocks: list[dict] = []
    for index, profile in enumerate(strata):
        truths = generate_truth(
            config,
            mission,
            replace(evaluation, seed=evaluation.seed ^ (index * 0x9E37)),
            profile,
            horizon_s=options.mission_time_s,
            terrain=terrain,
            prior=prior,
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
        block["target_profile"] = profile.name
        flight_blocks.append(block)

    model_worst = float(min(solution.target_detection_probabilities))
    flown_values = [
        float(block["detection_probability_within_limit"]) for block in flight_blocks
    ]
    flown_worst = min(flown_values)
    # v0.4 규약(ch4): 거동 등가중 평균이 "실비행 PD"다 — 모형의 minimax 와
    # 비교하는 공표값. 최악 거동값은 참고로 같이 남긴다.
    flown_mean = float(np.mean(flown_values))
    certified = bool(
        np.isfinite(diagnostics.relative_optimality_gap)
        and diagnostics.relative_optimality_gap <= scenario.solver.certificate_gap
    )

    metrics: dict[str, Any] = {
        "scenario": scenario.snapshot(),
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
            "model_detection_probability_worst": model_worst,
            "model_detection_probability_by_target": {
                name: float(value)
                for name, value in zip(
                    instance.target_names, solution.target_detection_probabilities
                )
            },
            "relative_optimality_gap": float(diagnostics.relative_optimality_gap),
            "certified": certified,
            "required_relative_gap": scenario.solver.certificate_gap,
            "lower_bound_nondetection": diagnostics.lower_bound_nondetection,
            "upper_bound_nondetection": diagnostics.upper_bound_nondetection,
            "iterations": diagnostics.iterations,
            "converged": diagnostics.converged,
            "runtime_s": diagnostics.runtime_s,
        },
        "flight": {
            "monte_carlo_samples": options.sample_count,
            "by_target": {
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
                    "unique_area_coverage_ratio": _maybe_float(
                        block.get("unique_area_coverage_ratio")
                    ),
                    "probability_mass_coverage": _maybe_float(
                        block.get("probability_mass_coverage")
                    ),
                }
                for block in flight_blocks
            },
            "detection_probability_mean": flown_mean,
            "detection_probability_worst": flown_worst,
            "model_minus_flown": model_worst - flown_mean,
        },
        "wall_time_s": None,  # 아래에서 채움
    }

    out_dir = Path(out_root) / scenario.name
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
