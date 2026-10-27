"""연속 공간에서 보상을 재는 학습 환경 — **행동은 그대로, 자만 바꾼다.**

왜 이 파일이 있나
-----------------
셀 모형은 슬라이스마다 셀 안의 위치를 지운다. 그래서 멈춰 있는 표적도 매 방문
새 자리에 있는 것처럼 다뤄지고, 거친 격자에서 정지 성향 표적의 탐지확률을 두 배
가까이 높게 본다 (본문 표 2: Tank 4x4 의 STOP_HEAVY 는 0.3822 대 0.1868).
셀 모형 위에서 학습한 정책은 그 전제까지 같이 물려받는다.

이 환경은 ``StoneSPXEnvironment`` 에서 **보상을 재는 자리 하나만** 갈아 끼운다.

    같은 것   행동공간(인접 셀), 실행가능집합(점유·분리 복구), 관측, 학습규칙
    다른 것   슬라이스 보상을 셀 질량이 아니라 **연속 공간의 입자**에서 잰다

그래서 두 정책의 차이를 "무엇을 최적화했는가" 하나로 돌릴 수 있다.

어떻게 빠르게 재나
------------------
모형 밖 평가기(``planning/continuous_mc.py``)는 입자마다 항적까지의 측방거리를
계산한다. 정확하지만 계획 하나에 수십 초라 학습에는 쓸 수 없다. 여기서는 순서를
뒤집는다.

    1. 조각마다 항적이 만든 hazard 를 **그 항적 둘레에만** 래스터로 그린다
       (픽셀 수는 입자 수와 무관하고, 항적은 한 슬라이스에 두어 셀만 지난다)
    2. 입자는 자기 자리의 픽셀 값을 읽기만 한다

표적 궤적은 계획에 의존하지 않으므로 학습 내내 **한 번만** 전개해 캐시한다
(공통 난수). 래스터 해상도와 슬라이스 분할이 만드는 오차는 승인된 평가기와
직접 대조해 보고한다 (``tools/continuous_env_check.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import os

import numpy as np

from cpp_search.models.searcher_track import build_windows
from cpp_search.solvers.mappo.spx_env import SPXEnvironmentConfig, StoneSPXEnvironment
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, lateral_detection


@dataclass(frozen=True, slots=True)
class ContinuousRewardConfig:
    """보상을 어떤 해상도로 잴지. 학습 하이퍼파라미터가 아니라 **측정 조건**이다."""

    #: 표적 입자 수. 평가기(4만~12만)보다 적게 쓰되, 최종 채점은 평가기가 한다.
    particle_count: int = 4000
    #: 래스터 한 픽셀의 크기 [m]. 탐지폭(400 m)보다 충분히 작아야 띠가 보인다.
    pixel_m: float = 100.0
    #: 슬라이스를 몇 조각으로 나눠 hazard 를 그릴지. 평가기는 dt = 5 s 로 나눈다.
    sub_slices: int = 3
    #: 셀 안 항로 간격. 규약값은 탐지폭과 같게 둔다 (본문 4.8절).
    lane_m: float = SUPPORT_HALF_M
    #: 입자 표본 seed. 학습 내내 고정한다 (공통 난수).
    seed: int = 20260201
    #: 이송 480 s 동안의 확산. 계획이 시작될 때 표적은 이미 퍼져 있다.
    belief_delay_s: float = 0.0
    #: 새 셀에 들어가는 규약 — "centre-sync"(본 표) | "nearest"(참고). 최종 채점과
    #: **같은 값**이어야 한다 (``core/searcher_track``, ``tools/conditions.py``).
    entry: str = "centre-sync"
    #: 정책이 고를 수 있는 셀 안 항로 간격 [m]. 비우면 ``lane_m`` 하나로 고정한다.
    #: 값을 주면 행동이 (셀, 간격) 쌍이 되어 행동 수가 후보 셀 수 x 간격 수가 된다.
    lane_options: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.particle_count < 100:
            raise ValueError("입자가 너무 적으면 보상이 표본잡음에 묻힌다")
        if self.pixel_m <= 0.0 or self.sub_slices < 1:
            raise ValueError("해상도는 양수여야 한다")
        if self.entry not in ("nearest", "centre-sync"):
            raise ValueError("학습 보상은 보고하는 두 규약에서만 잰다")


def rasterize(segments: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """구간 묶음이 각 점에 남기는 hazard. ``continuous_mc.accumulate_window`` 의 묶음판.

    같은 식을 쓰되 구간을 하나씩 돌지 않고 한 번에 계산한다. 두 함수가 같은 값을
    내는지는 ``tests/test_continuous_env.py`` 가 확인한다 — 다르면 학습이 평가기와
    다른 것을 최적화하게 된다.
    """

    if segments.size == 0:
        return np.zeros(gx.size, dtype=float)
    x0, y0, x1, y1 = (segments[:, i] for i in range(4))
    sx, sy = x1 - x0, y1 - y0
    length2 = sx * sx + sy * sy
    live = length2 > 1e-18
    if not np.any(live):
        return np.zeros(gx.size, dtype=float)
    x0, y0, sx, sy, length2 = x0[live], y0[live], sx[live], sy[live], length2[live]
    wx = gx[:, None] - x0[None, :]
    wy = gy[:, None] - y0[None, :]
    s = (wx * sx[None, :] + wy * sy[None, :]) / length2[None, :]
    inside = (s >= 0.0) & (s <= 1.0)
    distance = np.hypot(wx - s * sx[None, :], wy - s * sy[None, :])
    return np.where(inside, lateral_detection(distance), 0.0).sum(axis=1)


def _windows_to_segments(windows) -> np.ndarray:
    """창(꺾은선) 목록을 (x0, y0, x1, y1) 구간 배열로 편다."""

    rows = []
    for window in windows:
        points = np.asarray(window, dtype=float)
        if len(points) < 2:
            continue
        rows.append(np.column_stack([points[:-1, 0], points[:-1, 1],
                                     points[1:, 0], points[1:, 1]]))
    return np.concatenate(rows) if rows else np.zeros((0, 4))


class LocalRaster:
    """항적 둘레에만 깔린 hazard 격자. 밖은 0 이다 (측방거리가 지지반폭을 넘는다)."""

    __slots__ = ("x0", "y0", "pixel_m", "nx", "ny", "values")

    def __init__(self, segments: np.ndarray, pixel_m: float) -> None:
        self.pixel_m = float(pixel_m)
        if segments.size == 0:
            self.x0 = self.y0 = 0.0
            self.nx = self.ny = 0
            self.values = np.zeros((0, 0))
            return
        xs = np.concatenate([segments[:, 0], segments[:, 2]])
        ys = np.concatenate([segments[:, 1], segments[:, 3]])
        self.x0 = float(xs.min()) - SUPPORT_HALF_M
        self.y0 = float(ys.min()) - SUPPORT_HALF_M
        self.nx = int(ceil((float(xs.max()) + SUPPORT_HALF_M - self.x0) / self.pixel_m)) + 1
        self.ny = int(ceil((float(ys.max()) + SUPPORT_HALF_M - self.y0) / self.pixel_m)) + 1
        gx = self.x0 + (np.arange(self.nx) + 0.5) * self.pixel_m
        gy = self.y0 + (np.arange(self.ny) + 0.5) * self.pixel_m
        mesh_x, mesh_y = np.meshgrid(gx, gy)
        self.values = rasterize(segments, mesh_x.ravel(), mesh_y.ravel()).reshape(
            self.ny, self.nx)

    def at(self, positions: np.ndarray) -> np.ndarray:
        """각 점이 받는 hazard. 네 이웃 픽셀을 겹선형 보간한다. 래스터 밖은 0.

        가장 가까운 픽셀 값을 그대로 쓰면 띠 가장자리에서 한쪽으로 치우친다
        (Tank 8x8 에서 평균 -0.010). 보간하면 같은 픽셀 크기로 그 편의가 줄어든다.
        """

        if self.nx == 0:
            return np.zeros(positions.shape[0], dtype=float)
        fx = (positions[:, 0] - self.x0) / self.pixel_m - 0.5
        fy = (positions[:, 1] - self.y0) / self.pixel_m - 0.5
        inside = ((positions[:, 0] >= self.x0)
                  & (positions[:, 1] >= self.y0)
                  & (positions[:, 0] < self.x0 + self.nx * self.pixel_m)
                  & (positions[:, 1] < self.y0 + self.ny * self.pixel_m))
        ix = np.clip(np.floor(fx).astype(int), 0, self.nx - 2 if self.nx > 1 else 0)
        iy = np.clip(np.floor(fy).astype(int), 0, self.ny - 2 if self.ny > 1 else 0)
        tx = np.clip(fx - ix, 0.0, 1.0)
        ty = np.clip(fy - iy, 0.0, 1.0)
        ix1 = np.minimum(ix + 1, self.nx - 1)
        iy1 = np.minimum(iy + 1, self.ny - 1)
        value = (
            self.values[iy, ix] * (1 - tx) * (1 - ty)
            + self.values[iy, ix1] * tx * (1 - ty)
            + self.values[iy1, ix] * (1 - tx) * ty
            + self.values[iy1, ix1] * tx * ty
        )
        return np.where(inside, value, 0.0)


class ParticleTrace:
    """표적 하나의 입자 궤적. **계획과 무관**하므로 학습 내내 재사용한다."""

    __slots__ = ("positions", "escaped", "weights", "multiplier", "name")

    def __init__(self, positions: np.ndarray, escaped: np.ndarray,
                 weights: np.ndarray, multiplier: float, name: str) -> None:
        self.positions = positions          # [시각, 입자, 2]
        self.escaped = escaped              # [시각, 입자]
        self.weights = weights              # [입자]
        self.multiplier = float(multiplier)
        self.name = name


def build_particle_traces(instance, targets, *, mission, prior, terrain_coupling,
                          reward: ContinuousRewardConfig) -> list[ParticleTrace]:
    """표적별 입자를 임무 전체에 걸쳐 전개하고 조각 시작 시각의 위치를 남긴다.

    평가기의 ``particle_trace`` 를 **그대로** 부른다 — 같은 seed 유도, 같은 전개 간격
    (셀 모형과 같은 ``stepping="model"``), 같은 순서(hazard 를 먼저 재고 그다음에 민다).
    조각 수를 평가기와 같게 두면 (``sub_slices`` = ``round(slice_s / dt_s)``) 입자 표본까지
    똑같아진다. 전에는 여기서 따로 5 초씩 전개했고, 그 표적은 셀 모형의 표적과 달랐다
    (2026-10-08 수정, ``tools/target_step_probe.py``).
    """

    from cpp_search.solvers.milp.continuous_mc import particle_trace

    slice_s = float(instance.slice_s)
    dt = slice_s / reward.sub_slices
    steps = instance.time_slice_count * reward.sub_slices
    traces: list[ParticleTrace] = []
    for k, target in enumerate(targets):
        weights, positions, escaped = particle_trace(
            instance, target, k, mission=mission, prior=prior,
            terrain_coupling=terrain_coupling, particle_count=reward.particle_count,
            seed=reward.seed, steps=steps, dt=dt, belief_delay_s=reward.belief_delay_s,
            cache_dir=os.environ.get("CPP_SEARCH_TRACE_CACHE") or None,
            stepping="model", slice_s=slice_s)
        multiplier = float(
            np.asarray(instance.target_models[k].hazard_multiplier, dtype=float)
        )
        traces.append(ParticleTrace(
            positions, escaped, weights, multiplier,
            str(instance.target_names[k]).split("/")[-1],
        ))
    return traces


class ContinuousSPXEnvironment(StoneSPXEnvironment):
    """셀 모형과 같은 행동공간, 연속 공간에서 재는 보상."""

    def __init__(self, instance, config: SPXEnvironmentConfig | None = None, *,
                 targets=None, mission, prior=None, terrain_coupling=None,
                 reward: ContinuousRewardConfig | None = None,
                 traces: list[ParticleTrace] | None = None) -> None:
        super().__init__(instance, config)
        self.reward_config = reward or ContinuousRewardConfig()
        self.mission = mission
        if traces is None:
            if targets is None or prior is None or terrain_coupling is None:
                raise ValueError("입자 궤적을 만들려면 표적·사전분포·지형결합이 필요하다")
            traces = build_particle_traces(
                instance, targets, mission=mission, prior=prior,
                terrain_coupling=terrain_coupling, reward=self.reward_config)
        self._traces = traces
        if len(self._traces) != self.target_count:
            raise ValueError("입자 궤적은 표적마다 하나씩이어야 한다")
        if not hasattr(instance.grid, "half_side_m"):
            raise ValueError("연속 보상은 정사각 격자에서만 지원한다")
        self._side = float(instance.grid.cell_area_m2) ** 0.5
        self._lane_choices = tuple(float(v) for v in self.reward_config.lane_options) \
            or (float(self.reward_config.lane_m),)
        self._lanes: list[list[float]] = []

    # ------------------------------------------------------------ 행동 공간

    @property
    def action_count(self) -> int:
        """(셀, 항로 간격) 쌍의 수. 간격이 하나면 부모와 같다."""

        return self.config.action_count * len(self._lane_choices)

    @property
    def lane_choices(self) -> tuple[float, ...]:
        return self._lane_choices

    def _resolve_destinations(self, selected: np.ndarray):
        """행동을 (셀 후보, 항로 간격)으로 풀고 간격을 기록한다.

        복구 규칙은 부모가 그대로 맡는다 — 실행가능집합은 셀만이 정하므로
        간격 선택은 거기에 영향을 주지 않는다.
        """

        lanes = len(self._lane_choices)
        selected = np.asarray(selected, dtype=int)
        cells, picks = np.divmod(selected, lanes)
        for searcher, pick in enumerate(picks):
            self._lanes[searcher].append(self._lane_choices[int(pick)])
        return super()._resolve_destinations(cells)

    def plan_metadata(self) -> dict:
        """보고하는 계획에 딸린 것 — 탐색자마다 슬라이스별 항로 간격."""

        return {"lane_m": [list(row) for row in self._lanes]}

    # ------------------------------------------------------------ 내부 도구

    def _piece_segments(self, paths, time_index: int, lanes=None) -> list[np.ndarray]:
        """슬라이스 하나를 조각별 구간 묶음으로. 항적은 경로 **전체**에서 편다.

        셀 안 회로는 앞 슬라이스에서 어디까지 깔았는지에 이어지므로 슬라이스
        하나만 따로 그릴 수 없다. 매번 앞부분부터 다시 편다 (기하 계산이라 싸다).
        """

        sub = self.reward_config.sub_slices
        pieces: list[list] = [[] for _ in range(sub)]
        for searcher, path in enumerate(paths):
            windows = build_windows(
                list(path), self.instance.problem.searchers[searcher].start_state,
                self.instance.grid,
                transit_speed=float(self.mission.transit_speed_mps),
                search_speed=float(self.mission.search_speed_mps),
                slice_s=float(self.instance.slice_s),
                lane_m=(float(self.reward_config.lane_m) if lanes is None
                        else lanes[searcher]),
                cell_side=self._side, sub=sub, entry=self.reward_config.entry)
            for piece in range(sub):
                pieces[piece].append(windows[time_index * sub + piece])
        return [_windows_to_segments(group) for group in pieces]

    def _apply_slice(self, paths, time_index: int, survival: list[np.ndarray],
                     lanes=None) -> None:
        """이 슬라이스의 hazard 를 표적별 생존확률에 곱해 넣는다."""

        sub = self.reward_config.sub_slices
        for piece, segments in enumerate(
                self._piece_segments(paths, time_index, lanes)):
            raster = LocalRaster(segments, self.reward_config.pixel_m)
            step = time_index * sub + piece
            for index, trace in enumerate(self._traces):
                hazard = raster.at(trace.positions[step]) * trace.multiplier
                hazard = np.where(trace.escaped[step], 0.0, hazard)
                survival[index] *= np.exp(-hazard)

    # --------------------------------------------------------------- rollout

    def reset(self, seed: int = 0):
        observations = super().reset(seed)
        self._survival = [np.ones(trace.weights.size, dtype=float)
                          for trace in self._traces]
        self._lanes = [[] for _ in range(self.searcher_count)]
        return observations

    def _advance_targets(self, time_index: int, destinations: np.ndarray) -> np.ndarray:
        # 셀 모형 질량도 같이 굴린다 — 관측(local_observations)이 그걸 본다.
        super()._advance_targets(time_index, destinations)

        paths = [list(path) + [int(destinations[searcher])]
                 for searcher, path in enumerate(self._paths)]
        before = np.array([float(trace.weights @ self._survival[index])
                           for index, trace in enumerate(self._traces)])
        self._apply_slice(paths, time_index, self._survival,
                          lanes=[list(row) for row in self._lanes])
        after = np.array([float(trace.weights @ self._survival[index])
                          for index, trace in enumerate(self._traces)])
        return before - after

    def _score_paths(self, paths) -> tuple[float, ...]:
        """완주한 경로의 표적별 탐지확률 — **연속 공간에서** 잰 값."""

        return tuple(
            float(1.0 - trace.weights @ self._survival[index])
            for index, trace in enumerate(self._traces)
        )

    # ------------------------------------------------------------ 계획 채점

    def detection_probabilities(self, paths, lanes=None) -> tuple[float, ...]:
        """임의의 계획을 이 환경의 자로 채점한다 (학습과 무관, 대조용).

        절단평면 계획과 학습 계획을 같은 자로 나란히 놓을 때 쓴다. 최종 수치는
        승인된 평가기가 내지만, 그 전에 여기서 순위를 볼 수 있다.
        """

        paths = [list(path) for path in paths]
        survival = [np.ones(trace.weights.size, dtype=float) for trace in self._traces]
        for time_index in range(self.time_count):
            self._apply_slice(
                [path[: time_index + 1] for path in paths], time_index, survival,
                lanes=(None if lanes is None
                       else [list(row)[: time_index + 1] for row in lanes]))
        return tuple(float(1.0 - trace.weights @ survival[index])
                     for index, trace in enumerate(self._traces))
