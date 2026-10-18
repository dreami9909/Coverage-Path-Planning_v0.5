"""모형 밖 평가기의 **재현성**을 검사한다.

1. 입자 궤적 캐시를 쓰든 안 쓰든 결과가 같다. 다르면 캐시를 켠 실행과 끈 실행의
   수치를 한 표에 놓을 수 없다.
2. 항적 규약 세 가지가 평가기까지 실제로 전달된다. 전달이 끊기면 "두 규약을
   나란히" 잰 표가 같은 값을 두 번 적은 것이 된다.
"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from cpp_search.core.probability import TargetPrior
from cpp_search.planning.continuous_mc import SUPPORT_HALF_M, evaluate
from cpp_search.planning.terrain_belief import BELIEF_TERRAIN_FULL
from tests.fixtures import small_instance, small_mission, small_targets


def _paths(instance):
    """머무름과 이동이 섞인 실행가능 경로 — 탐색자마다 갈 수 있는 셀을 번갈아 고른다."""
    problem = instance.problem
    paths = []
    for index, searcher in enumerate(problem.searchers):
        cell, path = searcher.start_state, []
        for t in range(problem.time_count):
            options = [k for k in searcher.successors(cell) if k < instance.cell_count]
            cell = options[(index + 2 * t) % len(options)]
            path.append(int(cell))
        paths.append(tuple(path))
    return tuple(paths)


def _score(instance, entries, **options):
    result = evaluate(
        instance, small_targets(), entries, mission=small_mission(),
        prior=TargetPrior(kind="moving-ring"), terrain_coupling=BELIEF_TERRAIN_FULL,
        particle_count=600, dt_s=5.0, seed=7, progress=lambda *_: None, **options)
    return np.array([target["pd"] for target in result["targets"]])


class TraceCacheTests(unittest.TestCase):
    def test_cached_trajectories_reproduce_the_uncached_result_exactly(self) -> None:
        instance = small_instance(time_slice_count=3)
        entries = [("plan", _paths(instance), SUPPORT_HALF_M)]
        plain = _score(instance, entries)
        with tempfile.TemporaryDirectory() as folder:
            written = _score(instance, entries, trace_cache=folder)
            reread = _score(instance, entries, trace_cache=folder)
        np.testing.assert_array_equal(plain, written)
        np.testing.assert_array_equal(plain, reread)


class EntryConventionTests(unittest.TestCase):
    def test_the_entry_convention_reaches_the_tracks(self) -> None:
        instance = small_instance(time_slice_count=3)
        paths = _paths(instance)
        entries = [(name, paths, SUPPORT_HALF_M) for name in ("a", "b", "c")]
        pd = _score(instance, entries, entry=("nearest", "centre-sync", "nearest"))
        np.testing.assert_array_equal(pd[:, 0], pd[:, 2])       # 같은 규약 = 같은 값
        self.assertGreater(float(np.abs(pd[:, 0] - pd[:, 1]).max()), 1e-9)

    def test_an_unknown_convention_is_rejected(self) -> None:
        instance = small_instance(time_slice_count=3)
        with self.assertRaises(ValueError):
            _score(instance, [("plan", _paths(instance), SUPPORT_HALF_M)], entry=("edge",))


if __name__ == "__main__":
    unittest.main()
