"""평가기가 날리는 항적이 **계획이 말한 대로** 나는지 검사한다.

모형 밖 MC 의 모든 수치가 이 항적 위에서 나온다. 과거에 여기서 결함이 다섯 번
나왔고(``core/searcher_track`` 머리말), 그중 둘은 눈으로는 안 보이는 종류였다 —
체류마다 셀 중심으로 돌아가기, 셀을 옮길 때 계획보다 뒤처지기. 그래서 항적의
성질을 식으로 적어 매번 확인한다.

    1. 끊기지 않는다, 창 하나가 정확히 v·dt 다
    2. 같은 셀에 머무는 슬라이스는 셀 중심으로 돌아가지 않고 회로를 이어 간다
    3. 인접 셀로 옮기면 항상 그 슬라이스 안에 도착한다 (뒤처지지 않는다)
    4. 슬라이스 t 의 항적은 계획 셀 아니면 직전 셀 안에 있다
    5. 가장자리로 들어오면 첫 차선을 온전히 쓴다
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from cpp_search.core.searcher_track import (
    build_windows,
    cell_cover_polyline,
    nearest_entry,
)

SIDE, WIDTH, SPEED = 2000.0, 5, 40.0
SLICE_S = 55.0            # 도달거리/셀 = 1.1 (본 실험의 1.09~1.20 과 같은 구간)
LANE, SUB = 400.0, 11


class _Grid:
    """한 변 ``WIDTH`` 칸짜리 정사각 격자. 원점이 격자 중심."""

    def center_of(self, cell: int):
        ix, iy = cell % WIDTH, cell // WIDTH
        half = 0.5 * SIDE * WIDTH
        return SimpleNamespace(x=-half + (ix + 0.5) * SIDE, y=-half + (iy + 0.5) * SIDE)


def _box(cell: int):
    c = _Grid().center_of(cell)
    return c.x - SIDE / 2, c.x + SIDE / 2, c.y - SIDE / 2, c.y + SIDE / 2


def _fly(path, start, **overrides):
    ledger: list[dict] = []
    options = dict(transit_speed=SPEED, search_speed=SPEED, slice_s=SLICE_S,
                   lane_m=LANE, cell_side=SIDE, sub=SUB, ledger=ledger)
    options.update(overrides)
    return build_windows(list(path), start, _Grid(), **options), ledger


def _length(window) -> float:
    return float(np.hypot(np.diff(window[:, 0]), np.diff(window[:, 1])).sum())


def _inside(window, cells, tolerance=1e-6) -> bool:
    """창의 모든 점이 ``cells`` 중 하나의 상자 안에 있는가 (구간별 중점 포함)."""
    points = np.vstack([window, 0.5 * (window[:-1] + window[1:])])
    ok = np.zeros(len(points), dtype=bool)
    for cell in cells:
        x0, x1, y0, y1 = _box(cell)
        ok |= ((points[:, 0] >= x0 - tolerance) & (points[:, 0] <= x1 + tolerance)
               & (points[:, 1] >= y0 - tolerance) & (points[:, 1] <= y1 + tolerance))
    return bool(ok.all())


# 머무름·직교 이동·되돌아오기·연속 이동이 섞인 경로 (셀 12 가 격자 한가운데)
PATHS = (
    (12, [12, 12, 12, 13, 13, 13, 13, 8, 8, 7, 12, 12, 12, 12, 12, 12, 12, 12]),
    (12, [13, 14, 9, 4, 3, 2, 7, 12, 17, 22, 21, 20, 15, 10, 5, 0, 1, 2]),
    (6, [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
    (0, [5, 5, 10, 10, 10, 11, 11, 6, 6, 6, 1, 1, 0, 0]),
)


class NearestEntryTrackTests(unittest.TestCase):
    def test_track_is_continuous_and_every_window_is_one_step_long(self) -> None:
        for start, path in PATHS:
            windows, _ = _fly(path, start)
            self.assertEqual(len(windows), len(path) * SUB)
            for k, window in enumerate(windows):
                self.assertAlmostEqual(_length(window), SPEED * SLICE_S / SUB, places=6)
                if k:
                    self.assertAlmostEqual(
                        float(np.hypot(*(window[0] - windows[k - 1][-1]))), 0.0, places=9)

    def test_dwelling_never_returns_to_the_cell_centre(self) -> None:
        for start, path in PATHS:
            _, ledger = _fly(path, start)
            for row in ledger:
                if row["staying"]:
                    self.assertEqual(row["transit_m"], 0.0)
                    self.assertAlmostEqual(row["dwell_m"], SPEED * SLICE_S, places=6)

    def test_adjacent_moves_always_arrive_within_the_slice(self) -> None:
        for start, path in PATHS:
            _, ledger = _fly(path, start)
            for row in ledger:
                self.assertFalse(row["late"])
                self.assertEqual(row["lag_m"], 0.0)
                self.assertAlmostEqual(row["transit_m"] + row["dwell_m"],
                                       SPEED * SLICE_S, places=6)
                self.assertLessEqual(row["transit_m"], SIDE + 0.5 * LANE)

    def test_each_slice_stays_inside_the_planned_or_previous_cell(self) -> None:
        for start, path in PATHS:
            windows, _ = _fly(path, start)
            for t, cell in enumerate(path):
                previous = start if t == 0 else path[t - 1]
                for k in range(SUB):
                    self.assertTrue(_inside(windows[t * SUB + k], {cell, previous}),
                                    msg=f"slice {t} left cells {cell}/{previous}")
            # 머무는 슬라이스는 계획 셀을 벗어나지 않는다
            for t, cell in enumerate(path):
                if t and path[t - 1] == cell:
                    for k in range(SUB):
                        self.assertTrue(_inside(windows[t * SUB + k], {cell}))

    def test_edge_entry_sweeps_a_whole_first_lane(self) -> None:
        for position, axis in (((-1500.0, 130.0), "x"), ((300.0, 1900.0), "y"),
                               ((1700.0, -640.0), "x"), ((-220.0, -1300.0), "y")):
            entry, got_axis = nearest_entry(0.0, 0.0, SIDE, LANE, position)
            self.assertEqual(got_axis, axis)
            circuit = np.asarray(cell_cover_polyline(
                0.0, 0.0, SIDE, LANE, entry, got_axis, wide_side_first=True))
            first = circuit[1] - circuit[0]
            if np.hypot(*first) < 1e-9:
                first = circuit[2] - circuit[1]
            self.assertAlmostEqual(float(np.hypot(*first)), SIDE, places=6)
            # 진입점은 셀 경계 위에 있고, 지금 위치에서 한 셀보다 멀지 않다
            self.assertAlmostEqual(max(abs(entry[0]), abs(entry[1])), SIDE / 2, places=6)

    def test_a_move_out_of_reach_is_carried_over_not_accumulated(self) -> None:
        # 세 칸 떨어진 셀 (6,000 m) 은 한 슬라이스(2,200 m)에 못 닿는다. 닿는 데까지만
        # 가고 다음 슬라이스가 이어받아야 한다 — 총 항적 길이가 늘면 안 된다.
        path = [10, 13, 13, 13, 13, 13]
        windows, ledger = _fly(path, 10)
        self.assertEqual([row["late"] for row in ledger], [False, True, True, False, False, False])
        for window in windows:
            self.assertAlmostEqual(_length(window), SPEED * SLICE_S / SUB, places=6)
        self.assertTrue(_inside(windows[-1], {13}))

    def test_eastward_and_northward_runs_are_mirror_images(self) -> None:
        # x 와 y 를 바꾼 경로는 x 와 y 를 바꾼 항적을 내야 한다. 격자를 대각선으로
        # 뒤집으면 셀 (ix, iy) 가 (iy, ix) 가 된다.
        def mirror(cell: int) -> int:
            return (cell % WIDTH) * WIDTH + cell // WIDTH

        # 둘째·셋째 경로는 차선을 끝까지 쓴 뒤 **셀의 변 위에서** 다음 셀로 넘어간다.
        # 변 위의 위치를 셀 안쪽으로 취급하던 때는 이 경로들에서 거울상이 깨졌다.
        for start, path in ((10, [11, 12, 13, 14, 14, 9, 4]),
                            (4, [3, 3, 3, 3, 2, 3, 3, 8]),
                            (17, [12, 7, 12, 12, 12, 12, 17, 22])):
            east, _ = _fly(path, start)
            north, _ = _fly([mirror(c) for c in path], mirror(start))
            for a, b in zip(east, north):
                np.testing.assert_allclose(a, b[:, ::-1], atol=1e-6)

    def test_a_true_start_position_only_changes_where_the_track_begins(self) -> None:
        # 출발 셀 중심에서 600 m 떨어진 실제 위치에서 시작한다. 항적은 그 점에서
        # 시작하고, 길이·연속성·셀 준수는 그대로다.
        start, path = 12, [12, 12, 13, 13, 8]
        origin = (350.0, -480.0)
        windows, ledger = _fly(path, start, start_xy=origin)
        np.testing.assert_allclose(windows[0][0], origin)
        for k, window in enumerate(windows):
            self.assertAlmostEqual(_length(window), SPEED * SLICE_S / SUB, places=6)
            if k:
                self.assertAlmostEqual(
                    float(np.hypot(*(window[0] - windows[k - 1][-1]))), 0.0, places=9)
        for t, cell in enumerate(path):
            previous = start if t == 0 else path[t - 1]
            for k in range(SUB):
                self.assertTrue(_inside(windows[t * SUB + k], {cell, previous}))
        self.assertFalse(any(row["late"] for row in ledger))

    def test_lane_width_is_fixed_when_the_cell_is_entered(self) -> None:
        path = [12, 12, 13, 13]
        lanes = [400.0, 200.0, 800.0, 200.0]      # 머무는 중에 바꾼 값은 무시된다
        fixed, _ = _fly(path, 12, lane_m=[400.0, 400.0, 800.0, 800.0])
        varied, _ = _fly(path, 12, lane_m=lanes)
        for a, b in zip(fixed, varied):
            np.testing.assert_allclose(a, b)


class CentreEntryRegressionTests(unittest.TestCase):
    """2026-10-05 이전 규약. 결함이 든 채로 **그대로** 재현되어야 대조가 성립한다."""

    def test_centre_entry_falls_behind_the_plan(self) -> None:
        # 세 슬라이스 머문 뒤 왼쪽 셀로 간다. 회로 위 위치가 왼쪽 셀 중심에서
        # 한 슬라이스 거리(2,200 m)보다 800 m 멀어, 그만큼이 끝까지 밀린다.
        path = [12, 12, 12, 11, 11, 11]
        _, ledger = _fly(path, 12, entry="centre")
        self.assertEqual([row["late"] for row in ledger],
                         [False, False, False, True, False, False])
        self.assertAlmostEqual(ledger[-1]["lag_m"], 800.0, places=6)
        # 같은 경로를 새 규약으로 날리면 밀리지 않는다
        _, ledger = _fly(path, 12)
        self.assertFalse(any(row["late"] for row in ledger))

    def test_centre_entry_with_exact_timing_never_accumulates_lag(self) -> None:
        # 같은 경로, 같은 중심 경유. 못 닿은 800 m 를 다음 슬라이스가 먼저 날고
        # 그만큼 덜 쓴다 — 총 항적 길이가 늘지 않는다.
        path = [12, 12, 12, 11, 11, 11]
        windows, ledger = _fly(path, 12, entry="centre-sync")
        self.assertEqual([row["late"] for row in ledger],
                         [False, False, False, True, False, False])
        self.assertTrue(all(row["lag_m"] == 0.0 for row in ledger))
        self.assertAlmostEqual(ledger[4]["transit_m"], 800.0, places=6)
        self.assertAlmostEqual(ledger[4]["dwell_m"], SPEED * SLICE_S - 800.0, places=6)
        for window in windows:
            self.assertAlmostEqual(_length(window), SPEED * SLICE_S / SUB, places=6)
        # 지각이 없는 구간에서는 이전 규약과 **같은 항적**이다 (회로 모양이 같다)
        old, _ = _fly(path, 12, entry="centre")
        for a, b in zip(windows[: 3 * SUB], old[: 3 * SUB]):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_old_options_are_rejected_under_the_new_convention(self) -> None:
        with self.assertRaises(ValueError):
            _fly([12, 12], 12, dwell_from_centre=True)
        with self.assertRaises(ValueError):
            _fly([12, 12], 12, entry="centre-sync", dwell_from_centre=True)


if __name__ == "__main__":
    unittest.main()
