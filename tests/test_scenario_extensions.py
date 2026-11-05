"""시나리오 입력 확장([target.behaviour]·[belief]·[terrain.layout]·[truth])의 계약.

설계(docs/SCENARIO.md 2026-10-31 확정안)가 약속한 것을 코드로 지킨다:
① 잘못된 값은 로딩/준비 시점에 거부된다, ② custom 프로파일은 config 오버레이
하나를 통해 모든 소비처에 들어간다(단일 출처), ③ 신설 블록 사용은 provenance 에
기록되고 v0.4 anchor 비교 불가로 표시된다, ④ 명시 가설은 평가만 바꾼다.
"""

from __future__ import annotations

import unittest

from cpp_search.pipeline.scenario import (
    ScenarioError,
    build_scenario,
)


def _full_raw(**overrides):
    """신설 블록을 전부 쓰는 최소 시나리오 (연기 시험 크기)."""

    raw = {
        "name": "ext-smoke",
        "seed": 7,
        "target": {
            "case": "Tank",
            "max_speed_kph": 40.0,
            "behaviour": {
                "persistence_alpha": 0.5,
                "mode_probabilities": [0.4, 0.2, 0.2, 0.1, 0.1],
            },
        },
        "belief": {"kind": "tp-centered", "sigma_m": 500.0, "center_m": [1500.0, -800.0]},
        "terrain": {
            "layout": {
                "corridors": [
                    {"from": [-3000.0, 1000.0], "to": [3500.0, -500.0], "width_m": 800.0}
                ],
                "obstacles": [{"center": [1500.0, 2000.0], "radius_m": 900.0}],
                "concealments": [{"center": [-1000.0, -1500.0], "radius_m": 700.0}],
            }
        },
        "truth": {
            "hypotheses": [
                {"x": 1200.0, "y": -900.0, "behaviour": "STOP_HEAVY"},
                {"x": -800.0, "y": 1500.0, "behaviour": "custom"},
            ]
        },
        "grid": {"n": 4},
        "solver": {"time_limit_s": 60},
        "evaluation": {"mc_samples": 10, "particles": 200},
    }
    for key, value in overrides.items():
        raw[key] = value
    return raw


class ParseAndValidateTests(unittest.TestCase):
    def test_full_template_parses_and_records_custom_inputs(self) -> None:
        scenario = build_scenario(_full_raw())
        custom = set(scenario.custom_inputs)
        # 신설 4블록은 블록 단위 라벨로 올라간다.
        self.assertLessEqual(
            {"target.behaviour", "belief", "terrain.layout", "truth.hypotheses"},
            custom,
        )
        # 전수 감시: 연기 시험 크기(격자·시간한도·입자)도 비교 불가 사유다.
        self.assertLessEqual(
            {"grid.n", "solver.time_limit_s", "evaluation.particles"}, custom
        )
        # center_m 은 선언만 — 경고가 남는다.
        self.assertTrue(any("belief.center_m" in w for w in scenario.warnings))

    def test_any_departure_from_v04_defaults_is_flagged(self) -> None:
        """전수 감시 (2026-11-03): 목록 밖 키도 기본형과의 diff 로 잡힌다."""
        base = {"name": "p", "target": {"case": "TEL"}}
        self.assertEqual(build_scenario(base).custom_inputs, ())
        self.assertEqual(
            build_scenario({**base, "vehicle": {"count": 4}}).custom_inputs,
            ("vehicle.count",),
        )
        self.assertEqual(
            build_scenario({**base, "mission": {"search_s": 900.0}}).custom_inputs,
            ("mission.search_s",),
        )
        # 기본값을 명시로 적은 것은 같은 세계 — 올라가지 않는다.
        explicit = {**base, "mission": {"ingress_s": 480.0, "search_s": 600.0}}
        self.assertEqual(build_scenario(explicit).custom_inputs, ())
        # 평가 표본 수는 정밀도만 — 예외.
        samples = {**base, "evaluation": {"mc_samples": 50}}
        self.assertEqual(build_scenario(samples).custom_inputs, ())

    def test_unknown_keys_are_rejected(self) -> None:
        """오타가 기본값으로 조용히 떨어지지 않는다 (2026-11-04)."""
        for raw in (
            _full_raw(vehical={"count": 4}),                       # 최상위 오타
            _full_raw(evaluation={"mc_sample": 10}),               # 블록 안 오타
            _full_raw(target={"case": "Tank", "behavior": {}}),    # 영/미 철자
        ):
            with self.assertRaisesRegex(ScenarioError, "알 수 없는 키"):
                build_scenario(raw)
        # 테이블 목록 항목 안의 오타도 잡는다.
        raw = _full_raw()
        raw["truth"]["hypotheses"][0]["behavior"] = "STOP_HEAVY"
        with self.assertRaisesRegex(ScenarioError, "hypotheses\\[\\].behavior"):
            build_scenario(raw)

    def test_hypothesis_requires_coordinates(self) -> None:
        raw = _full_raw()
        del raw["truth"]["hypotheses"][0]["x"]
        with self.assertRaisesRegex(ScenarioError, "x·y"):
            build_scenario(raw)

    def test_validate_inputs_rejects_unknown_hypothesis_behaviour(self) -> None:
        """가설 거동 이름 검사가 solve 전(--validate-only 경로)에 돈다."""
        from cpp_search.pipeline.run import validate_inputs

        raw = _full_raw()
        raw["truth"]["hypotheses"][0]["behaviour"] = "NO_SUCH_BEHAVIOUR"
        scenario = build_scenario(raw)
        with self.assertRaisesRegex(ScenarioError, "NO_SUCH_BEHAVIOUR"):
            validate_inputs(scenario)
        # 정상 시나리오는 해소값을 돌려준다.
        resolved = validate_inputs(build_scenario(_full_raw()))
        self.assertGreater(resolved["aoi_radius_m"], 0.0)

    def test_vehicle_count_override_resizes_searcher_classes(self) -> None:
        """vehicle.count ≠ 6 이 챕터 config 의 동질 탐색자 등급 선언과
        충돌해 solve 직전에 죽던 결함 (2026-11-05) — count 가 따라가야 한다."""
        for count in (3, 12):
            scenario = build_scenario({"name": "s", "vehicle": {"count": count}})
            classes = scenario.chapter_config().get("stone_spx.searcher_classes")
            self.assertEqual(len(classes), 1)
            self.assertEqual(classes[0]["count"], count)

    def test_mode_probabilities_must_sum_to_one(self) -> None:
        raw = _full_raw()
        raw["target"]["behaviour"]["mode_probabilities"] = [0.4, 0.4, 0.4, 0.1, 0.1]
        with self.assertRaisesRegex(ScenarioError, "합은 1"):
            build_scenario(raw)

    def test_halt_speed_bound_must_be_zero(self) -> None:
        raw = _full_raw()
        raw["target"]["behaviour"]["mode_speed_bounds_kph"] = [
            [0, 5], [0, 20], [20, 40], [2, 32], [0, 40]
        ]
        with self.assertRaisesRegex(ScenarioError, "HALT"):
            build_scenario(raw)

    def test_speed_bound_cannot_exceed_target_max(self) -> None:
        raw = _full_raw()
        raw["target"]["behaviour"]["mode_speed_bounds_kph"] = [
            [0, 0], [0, 20], [20, 55], [2, 32], [0, 40]
        ]
        with self.assertRaisesRegex(ScenarioError, "max_speed_kph"):
            build_scenario(raw)

    def test_custom_hypothesis_requires_behaviour_block(self) -> None:
        raw = _full_raw()
        del raw["target"]["behaviour"]
        with self.assertRaisesRegex(ScenarioError, "custom"):
            build_scenario(raw)

    def test_empty_layout_is_rejected(self) -> None:
        raw = _full_raw()
        raw["terrain"]["layout"] = {}
        with self.assertRaisesRegex(ScenarioError, "layout"):
            build_scenario(raw)

    def test_geometry_outside_aoi_is_rejected(self) -> None:
        scenario = build_scenario(_full_raw())
        with self.assertRaisesRegex(ScenarioError, "AOI"):
            scenario.validate_geometry(1000.0)   # 모든 좌표가 1 km 원 밖
        scenario.validate_geometry(8884.0)       # Tank AOI 안 — 통과


class ChapterConfigOverlayTests(unittest.TestCase):
    def test_custom_profile_is_injected_as_single_source(self) -> None:
        scenario = build_scenario(_full_raw())
        config = scenario.chapter_config()
        self.assertEqual(
            config.common_get("target_motion.assignment.Tank-nominal"), "CUSTOM"
        )
        self.assertEqual(
            float(
                config.common_get(
                    "target_motion.behaviour_profiles.CUSTOM.persistence_alpha"
                )
            ),
            0.5,
        )
        # base 생략 -> 케이스 기본 배정으로 해소된다.
        self.assertEqual(
            config.common_get("target_motion.custom_behaviour_base"),
            "MANEUVER_HEAVY",
        )
        # 오버라이드하지 않은 키는 base 프로파일 값이 유지된다.
        self.assertIsNotNone(
            config.common_get("target_motion.behaviour_profiles.CUSTOM.mode_speed_bounds_kph")
        )

    def test_unknown_base_is_rejected(self) -> None:
        raw = _full_raw()
        raw["target"]["behaviour"]["base"] = "NO_SUCH_PROFILE"
        scenario = build_scenario(raw)
        with self.assertRaisesRegex(ScenarioError, "behaviour_profiles"):
            scenario.chapter_config()

    def test_belief_kind_and_sigma_flow_into_cue_ring(self) -> None:
        scenario = build_scenario(_full_raw())
        config = scenario.chapter_config()
        self.assertEqual(config.common_get("aoi.cue_ring.kind"), "tp-centered")
        self.assertEqual(float(config.common_get("aoi.cue_ring.sigma_m")), 500.0)


class EndToEndTests(unittest.TestCase):
    """연기 시험 크기의 전체 실행 — 수치가 아니라 배선을 검사한다."""

    def test_custom_run_produces_labelled_outputs(self) -> None:
        import tempfile
        from pathlib import Path

        from cpp_search.pipeline.run import run_scenario

        scenario = build_scenario(_full_raw())
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_scenario(scenario, out_root=Path(tmp))

        provenance = metrics["provenance"]
        self.assertFalse(provenance["comparable_to_v04_anchors"])
        self.assertIn("terrain.layout", provenance["custom_inputs"])

        # 계획 인스턴스: 케이스 최악 거동 집합 + custom 이 추가된다.
        model_targets = list(metrics["solver"]["model_detection_probability_by_target"])
        self.assertTrue(any("CUSTOM(base=" in name for name in model_targets))
        self.assertGreater(len(model_targets), 1)

        # 최악값에는 누가 최악인지 이름이 같이 남는다.
        self.assertIn(
            metrics["solver"]["model_detection_probability_worst_target"],
            model_targets,
        )
        self.assertIn(
            metrics["mc"]["detection_probability_worst_target"],
            metrics["mc"]["by_target"],
        )

        # 평가: [truth] 가설이 있으면 실비행 MC 로, 두 판이 라벨 그대로 채점된다.
        self.assertEqual(metrics["mc"]["kind"], "flight-hypotheses")
        mc_keys = sorted(metrics["mc"]["by_target"])
        self.assertEqual(len(mc_keys), 2)
        self.assertTrue(any(key.startswith("STOP_HEAVY@") for key in mc_keys))
        self.assertTrue(any(key.startswith("custom@") for key in mc_keys))
        for block in metrics["mc"]["by_target"].values():
            self.assertGreaterEqual(block["detection_probability"], 0.0)
            self.assertLessEqual(block["detection_probability"], 1.0)

    def test_standard_run_scores_continuous_convention_mc(self) -> None:
        """[truth] 없음 → MC 열은 규약 맞춤 연속 MC, 표적 집합은 모형과 같다."""
        import tempfile
        from pathlib import Path

        from cpp_search.pipeline.run import run_scenario

        raw = _full_raw()
        raw.pop("truth")
        scenario = build_scenario(raw)
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_scenario(scenario, out_root=Path(tmp))

        mc = metrics["mc"]
        self.assertEqual(mc["kind"], "continuous-centre-sync")
        self.assertEqual(
            sorted(mc["by_target"]),
            sorted(metrics["solver"]["model_detection_probability_by_target"]),
        )
        self.assertIn(mc["detection_probability_worst_target"], mc["by_target"])
        for block in mc["by_target"].values():
            self.assertGreaterEqual(block["detection_probability"], 0.0)
            self.assertLessEqual(block["detection_probability"], 1.0)
        self.assertIsNotNone(
            metrics["geometry"]["unique_area_coverage_ratio"]
        )

    def test_second_run_reuses_stored_plan(self) -> None:
        """같은 입력의 재실행은 plan.json 을 재사용한다 — solve 비결정성 차단."""
        import tempfile
        from pathlib import Path

        from cpp_search.pipeline.run import run_scenario

        raw = _full_raw()
        raw.pop("truth")
        scenario = build_scenario(raw)
        with tempfile.TemporaryDirectory() as tmp:
            first = run_scenario(scenario, out_root=Path(tmp))
            second = run_scenario(scenario, out_root=Path(tmp))

        self.assertFalse(first["solver"]["plan_reused"])
        self.assertTrue(second["solver"]["plan_reused"])
        self.assertEqual(
            first["solver"]["model_detection_probability_by_target"],
            second["solver"]["model_detection_probability_by_target"],
        )
        self.assertEqual(
            first["solver"]["relative_optimality_gap"],
            second["solver"]["relative_optimality_gap"],
        )

    def test_changed_solver_settings_invalidate_stored_plan(self) -> None:
        """solver 설정은 인스턴스 fingerprint 에 없다 — 따로 대조해서,
        time_limit·인증 기준을 바꾼 재실행이 옛 계획을 재사용하지 않는다."""
        import tempfile
        from pathlib import Path

        from cpp_search.pipeline.run import run_scenario

        raw = _full_raw()
        raw.pop("truth")
        with tempfile.TemporaryDirectory() as tmp:
            first = run_scenario(build_scenario(raw), out_root=Path(tmp))
            raw["solver"]["time_limit_s"] = 90
            changed = run_scenario(build_scenario(raw), out_root=Path(tmp))

        self.assertFalse(first["solver"]["plan_reused"])
        self.assertFalse(changed["solver"]["plan_reused"])


if __name__ == "__main__":
    unittest.main()
