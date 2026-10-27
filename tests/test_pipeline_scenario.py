"""pipeline 입력 검증 — 실행 전에 거부되는지, 번역이 정확한지.

솔버를 돌리지 않는 빠른 테스트만 둔다. 전체 실행은
``config/scenarios/smoke.toml`` 로 수동 확인한다 (수십 초).
"""

from __future__ import annotations

import unittest

from cpp_search.pipeline.scenario import (
    ScenarioError,
    SEEKER_TEMPLATES,
    build_scenario,
    load_scenario,
)


def minimal(**overrides) -> dict:
    raw: dict = {"name": "unit", "target": {"case": "TEL"}}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key].update(value)
        else:
            raw[key] = value
    return raw


class ScenarioValidationTests(unittest.TestCase):
    def test_minimal_scenario_builds_with_template_defaults(self) -> None:
        scenario = build_scenario(minimal())
        self.assertEqual(scenario.seeker.template, "SR-Z50")
        self.assertEqual(
            scenario.seeker.gimbal.max_angle_deg,
            SEEKER_TEMPLATES["SR-Z50"]["gimbal"]["max_angle_deg"],
        )
        self.assertEqual(scenario.grid.n, 10)
        self.assertEqual(scenario.target.tp_error_sigma_m, 500.0)

    def test_scenario_keys_override_template(self) -> None:
        scenario = build_scenario(
            minimal(seeker={"altitude_m": 900.0, "eo": {"hfov_deg": 24.0}})
        )
        self.assertEqual(scenario.seeker.altitude_m, 900.0)
        self.assertEqual(scenario.seeker.eo.hfov_deg, 24.0)
        # 건드리지 않은 채널은 템플릿 값 유지
        self.assertEqual(scenario.seeker.ir.hfov_deg, 18.0)

    def test_unknown_template_rejected(self) -> None:
        with self.assertRaisesRegex(ScenarioError, "미등록"):
            build_scenario(minimal(seeker={"template": "EO-9000"}))

    def test_unknown_case_rejected(self) -> None:
        with self.assertRaisesRegex(ScenarioError, "target.case"):
            build_scenario(minimal(target={"case": "Submarine"}))

    def test_mappo_solver_rejected_until_stage_two(self) -> None:
        with self.assertRaisesRegex(ScenarioError, "2단계"):
            build_scenario(minimal(solver={"kind": "mappo"}))

    def test_particle_cache_limit_enforced(self) -> None:
        with self.assertRaisesRegex(ScenarioError, "4만"):
            build_scenario(minimal(evaluation={"particles": 50_000}))

    def test_vfov_is_recorded_as_warning_not_used(self) -> None:
        scenario = build_scenario(minimal(seeker={"eo": {"vfov_deg": 10.1}}))
        self.assertEqual(scenario.seeker.eo.vfov_deg, 10.1)
        self.assertTrue(any("vfov" in w.lower() for w in scenario.warnings))

    def test_run_name_must_be_pathsafe(self) -> None:
        with self.assertRaisesRegex(ScenarioError, "name"):
            build_scenario({"name": "bad name/with space"})


class ScenarioTranslationTests(unittest.TestCase):
    def test_overrides_land_in_chapter_config(self) -> None:
        scenario = build_scenario(
            minimal(
                seeker={"altitude_m": 800.0},
                target={"tp_error_sigma_m": 250.0},
                mission={"ingress_s": 300.0},
                vehicle={"count": 4},
            )
        )
        config = scenario.chapter_config()
        self.assertEqual(config.common_get("sensor.altitude_m"), 800.0)
        self.assertEqual(config.common_get("aoi.cue_ring.sigma_m"), 250.0)
        self.assertEqual(config.common_get("mission.uav_count"), 4)
        self.assertEqual(config.get("stone_spx.initial_belief_delay_s"), 300.0)
        self.assertEqual(config.mission().uav_count, 4)
        self.assertEqual(config.sensor().altitude_m, 800.0)

    def test_per_channel_hfov_reaches_sensor_spec(self) -> None:
        scenario = build_scenario(
            minimal(seeker={"eo": {"hfov_deg": 24.0}, "ir": {"hfov_deg": 12.0}})
        )
        sensor = scenario.chapter_config().sensor()
        self.assertEqual(sensor.eo.horizontal_fov_deg, 24.0)
        self.assertEqual(sensor.ir.horizontal_fov_deg, 12.0)

    def test_grid_declaration_overrides_base(self) -> None:
        scenario = build_scenario(
            minimal(grid={"n": 8}, solver={"time_limit_s": 120.0})
        )
        declared = scenario.grid_declaration({"master_backend": "highs"})
        self.assertEqual(declared["grid_width"], 8)
        self.assertEqual(declared["square_fit"], "circumscribed")
        self.assertEqual(declared["time_slice_count"], "auto")
        self.assertEqual(declared["master_time_limit_s"], 120.0)
        self.assertEqual(declared["master_backend"], "highs")

    def test_run_options_mapping(self) -> None:
        scenario = build_scenario(
            minimal(evaluation={"mc_samples": 123, "particles": 456})
        )
        options = scenario.run_options()
        self.assertEqual(options.sample_count, 123)
        self.assertEqual(options.particle_count, 456)
        self.assertEqual(options.target_profile, "tel")
        self.assertEqual(options.mission_time_s, 600.0)


class ScenarioFileTests(unittest.TestCase):
    def test_shipped_scenarios_parse(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "config" / "scenarios"
        for name in ("baseline_tel", "smoke"):
            scenario = load_scenario(root / f"{name}.toml")
            self.assertEqual(scenario.name, name)


if __name__ == "__main__":
    unittest.main()
