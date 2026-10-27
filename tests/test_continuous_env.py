"""연속 보상 환경이 **평가기와 같은 것을 재는지** 검사한다.

두 가지가 지켜져야 2단계 비교가 성립한다.

1. 묶음 래스터가 평가기의 ``accumulate_window`` 와 같은 값을 낸다. 다르면 학습이
   평가기와 다른 탐지 모형을 최적화하게 된다.
2. 에피소드가 온라인으로 쌓은 탐지확률이, 같은 경로를 따로 채점한 값과 같다.
   다르면 보상과 보고가 어긋난다.
"""

from __future__ import annotations

import unittest

import numpy as np

from cpp_search.models.probability import TargetPrior
from cpp_search.learning.continuous_env import (
    ContinuousRewardConfig,
    ContinuousSPXEnvironment,
    LocalRaster,
    _windows_to_segments,
    rasterize,
)
from cpp_search.planning.continuous_mc import accumulate_window
from cpp_search.planning.terrain_belief import BELIEF_TERRAIN_FULL
from tests.fixtures import small_instance, small_mission, small_targets


def _environment(**reward_overrides) -> ContinuousSPXEnvironment:
    instance = small_instance(time_slice_count=3)
    reward = ContinuousRewardConfig(
        particle_count=400, pixel_m=100.0, sub_slices=2, seed=7, **reward_overrides
    )
    return ContinuousSPXEnvironment(
        instance,
        targets=small_targets(),
        mission=small_mission(),
        prior=TargetPrior(kind="moving-ring"),
        terrain_coupling=BELIEF_TERRAIN_FULL,
        reward=reward,
    )


class RasterTests(unittest.TestCase):
    def test_batched_raster_matches_the_evaluator_kernel(self) -> None:
        rng = np.random.default_rng(11)
        gx = rng.uniform(-3_000.0, 3_000.0, 400)
        gy = rng.uniform(-3_000.0, 3_000.0, 400)
        windows = [rng.uniform(-2_000.0, 2_000.0, size=(count, 2))
                   for count in (2, 3, 4, 2)]
        expected = np.zeros(gx.size)
        for window in windows:
            accumulate_window(gx, gy, window, 1.0, expected)
        got = rasterize(_windows_to_segments(windows), gx, gy)
        self.assertLess(float(np.abs(got - expected).max()), 1e-12)

    def test_raster_is_zero_outside_the_sweep_support(self) -> None:
        segments = np.array([[0.0, 0.0, 1_000.0, 0.0]])
        raster = LocalRaster(segments, 100.0)
        far = np.array([[500.0, 5_000.0], [-9_000.0, 0.0]])
        self.assertEqual(float(np.abs(raster.at(far)).max()), 0.0)


class OnlineScoringTests(unittest.TestCase):
    def test_online_accumulation_matches_scoring_the_same_paths(self) -> None:
        environment = _environment()
        rng = np.random.default_rng(3)
        environment.reset(0)
        done = False
        metrics = environment.metrics()
        while not done:
            actions = rng.integers(
                0, environment.action_count, size=environment.searcher_count
            )
            *_, done, metrics = environment.step(actions)
        standalone = environment.detection_probabilities(metrics.paths)
        for left, right in zip(metrics.target_detection_probabilities, standalone):
            self.assertAlmostEqual(left, right, places=12)

    def test_reward_is_measured_outside_the_cell_model(self) -> None:
        """연속 값과 셀 모형 값이 **같지 않아야** 한다 — 같으면 자를 안 바꾼 것이다."""

        environment = _environment()
        rng = np.random.default_rng(5)
        environment.reset(0)
        done = False
        metrics = environment.metrics()
        while not done:
            actions = rng.integers(
                0, environment.action_count, size=environment.searcher_count
            )
            *_, done, metrics = environment.step(actions)
        cell = environment.instance.score_paths(metrics.paths)
        continuous = metrics.target_detection_probabilities
        self.assertEqual(len(cell), len(continuous))
        self.assertGreater(
            max(abs(a - b) for a, b in zip(cell, continuous)), 1e-6
        )


if __name__ == "__main__":
    unittest.main()
