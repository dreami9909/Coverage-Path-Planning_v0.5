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
        self.assertEqual(
            scenario.custom_inputs,
            ("target.behaviour", "belief", "terrain.layout", "truth.hypotheses"),
        )
        # center_m 은 선언만 — 경고가 남는다.
        self.assertTrue(any("belief.center_m" in w for w in scenario.warnings))

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
            metrics["flight"]["detection_probability_worst_target"],
            metrics["flight"]["by_target"],
        )

        # 평가: 명시 가설 두 판이 라벨 그대로 채점된다.
        flight_keys = sorted(metrics["flight"]["by_target"])
        self.assertEqual(len(flight_keys), 2)
        self.assertTrue(any(key.startswith("STOP_HEAVY@") for key in flight_keys))
        self.assertTrue(any(key.startswith("custom@") for key in flight_keys))
        for block in metrics["flight"]["by_target"].values():
            self.assertGreaterEqual(block["detection_probability"], 0.0)
            self.assertLessEqual(block["detection_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
