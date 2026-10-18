"""모형 밖 몬테카를로 — 셀 이산화를 쓰지 않는 평가기.

셀 모형(``theory.path_constrained.nondetection_trace``)은 자기 자신을 검증할 수
없다. 전이행렬과 사전분포를 입력으로 받아 쓰므로 같은 이산화가 양쪽에 들어가
이산화 오차가 원리적으로 드러나지 않는다. 이 파일은 그 바깥에서 잰다.

    표적    연속 좌표 위를 IMM5 로 직접 전개 (입자 하나가 진리 궤적 하나)
    탐색자  ``core.searcher_track`` 이 편 항적을 실제로 비행
    탐지    항적에 대한 **측방거리**로 계산. 셀도 격자도 쓰지 않는다

두 모형이 같은 양을 비교하도록 단위시간당 소인량을 일치시켰다
(``selftest`` 가 매 실행마다 확인한다).

주의 — 여기서 얻는 표적별 탐지확률은 셀 모형의 ``score_paths`` 와 **표적 첨자가
같은 순서**다. 최악 표적만 저장하면 나중에 엉뚱한 기준끼리 대조하게 되므로
표적별 배열을 모두 남긴다.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from math import pi

import numpy as np

from cpp_search.core.searcher_track import build_windows
from cpp_search.theory.path_constrained import joint_survival

#: 횡방향 지지반폭 [m]. 탐지폭 W 는 이 값과 같다 (raised cosine 의 평균이 1/2).
SUPPORT_HALF_M = 400.0


def lateral_detection(offset_m: np.ndarray, half: float = SUPPORT_HALF_M) -> np.ndarray:
    """항적으로부터 측방거리 offset 에서의 탐지율 (raised cosine)."""
    u = np.abs(offset_m) / half
    return np.where(u >= 1.0, 0.0, 0.5 * (1.0 + np.cos(pi * np.clip(u, 0.0, 1.0))))


def accumulate_window(px, py, window, multiplier, out, law="exponential"):
    """창 하나(꺾은선 여러 구간)가 각 점에 남기는 hazard 를 ``out`` 에 더한다.

    ``law`` 는 측방 곡선 ``p(u)`` 를 무엇으로 읽을지다.
        "exponential"  hazard 로 읽는다: 생존 ×= exp(−p). 셀 모형(Stone 의 지수
                       탐지함수)과 같은 법칙이며, 소인량 ∫p du = W 가 그대로 맞는다.
        "probability"  한 번 지나갈 때의 탐지확률로 읽는다: 생존 ×= (1 − p).
                       1장의 소인폭 정의(W = ∫P_D(u) du, 중심선 PD0)를 글자 그대로
                       따른 것이다. 창 하나가 곧 한 번의 통과다.
    """
    take_log = law == "probability"
    for i in range(len(window) - 1):
        u0, u1 = window[i], window[i + 1]
        sx, sy = u1[0] - u0[0], u1[1] - u0[1]
        length2 = sx * sx + sy * sy
        if length2 < 1e-18:
            continue
        wx, wy = px - u0[0], py - u0[1]
        s = (wx * sx + wy * sy) / length2
        inside = (s >= 0.0) & (s <= 1.0)
        distance = np.hypot(wx - s * sx, wy - s * sy)
        value = np.where(inside, lateral_detection(distance), 0.0) * multiplier
        if take_log:
            value = -np.log1p(-np.clip(value, 0.0, 1.0 - 1e-12))
        out += value
    return out


def selftest(seed: int = 11, tolerance: float = 0.02) -> bool:
    """직선 비행의 평균 hazard 가 W·v·t/면적 과 맞는지 확인한다.

    이 검사가 통과해야 셀 모형과 같은 양을 비교하는 것이다. 실패하면 탐지폭
    규약이나 항적 길이가 어긋난 것이므로 그 위의 모든 수치가 무의미해진다.
    """
    rng = np.random.default_rng(seed)
    speed, duration, box, count = 60.0, 120.0, 12000.0, 500_000
    px = rng.uniform(-box / 2, box / 2, count)
    py = rng.uniform(-box / 2, box / 2, count)
    steps = 120
    dt = duration / steps
    acc = np.zeros(count)
    x = -speed * duration / 2
    for _ in range(steps):
        accumulate_window(px, py, np.array([[x, 0.0], [x + speed * dt, 0.0]]), 1.0, acc)
        x += speed * dt
    got = acc.mean()
    want = SUPPORT_HALF_M * speed * duration / (box * box)
    print(f"selftest  평균 hazard {got:.6e} / 이론 {want:.6e} = {got / want:.4f}", flush=True)
    return abs(got / want - 1.0) < tolerance


def _cell_index(grid, px, py):
    """정사각 격자에서 각 점의 셀 번호 (``_SquareGrid.cell_of`` 의 벡터판).

    격자 밖 점은 가장자리 셀로 잘라 넣는다 — 호출자가 ``escaped`` 로 걸러야 한다.
    """
    cx, cy, half = float(grid.center.x), float(grid.center.y), float(grid.half_side_m)
    ix = np.clip(np.floor((px - cx + half) / float(grid.cell_width_m)).astype(int), 0, grid.width - 1)
    iy = np.clip(np.floor((py - cy + half) / float(grid.cell_height_m)).astype(int), 0, grid.height - 1)
    return iy * grid.width + ix


def _cell_mean_hazard(grid, plan_tracks, slice_count, steps_per_slice, per_side=24):
    """슬라이스별·셀별로, 항적이 그 셀 위에 실제로 뿌린 hazard 의 면적 평균.

    셀마다 ``per_side²`` 개의 격자점을 깔고 그 슬라이스의 창을 모두 더해 평균한다.
    셀 모형의 ``Y_t[i]`` 와 견주면 **셀 경계 밖으로 샌 소인**이 드러난다.
    """
    cx, cy, half = float(grid.center.x), float(grid.center.y), float(grid.half_side_m)
    cw, ch = float(grid.cell_width_m), float(grid.cell_height_m)
    frac = (np.arange(per_side) + 0.5) / per_side
    fx, fy = np.meshgrid(frac, frac); fx, fy = fx.ravel(), fy.ravel()
    cells = np.arange(grid.width * grid.height)
    ix, iy = cells % grid.width, cells // grid.width
    gx = (cx - half + (ix[:, None] + fx[None, :]) * cw).ravel()
    gy = (cy - half + (iy[:, None] + fy[None, :]) * ch).ravel()
    out = np.zeros((slice_count, cells.size))
    for t in range(slice_count):
        h = np.zeros(gx.size)
        for track in plan_tracks:
            for sub in range(steps_per_slice):
                accumulate_window(gx, gy, track[t * steps_per_slice + sub], 1.0, h)
        out[t] = h.reshape(cells.size, per_side * per_side).mean(axis=1)
    return out


def _virtual_positions(grid, px, py, rng):
    """각 입자가 **지금 있는 셀** 안에서 균등하게 다시 뽑은 가상 위치.

    셀 모형이 표적에 대해 전제하는 것이 정확히 이것이다 — 어느 셀에 있는지만
    기억하고 셀 안 위치는 매 단계 새로 뽑힌다. 정사각 격자만 지원한다.
    """
    cx, cy, half = float(grid.center.x), float(grid.center.y), float(grid.half_side_m)
    cw, ch = float(grid.cell_width_m), float(grid.cell_height_m)
    cells = _cell_index(grid, px, py)
    ix, iy = cells % grid.width, cells // grid.width
    vx = cx - half + (ix + rng.uniform(0.0, 1.0, px.size)) * cw
    vy = cy - half + (iy + rng.uniform(0.0, 1.0, py.size)) * ch
    return vx, vy


def particle_trace(instance, target, index, *, mission, prior, terrain_coupling,
                   particle_count: int, seed: int, steps: int, dt: float,
                   belief_delay_s: float = 0.0, cache_dir=None,
                   stepping: str = "model", slice_s: float | None = None):
    """표적 하나의 입자 궤적 — (가중치, 위치[스텝, 입자, 2], 이탈[스텝, 입자]).

    표적 궤적은 계획에도 항적 규약에도 의존하지 않는다. 그래서 같은 조건을 여러
    도구가 채점할 때(분해·법칙·규약 대조·학습 계획 채점) 궤적은 **한 번만** 전개하면
    된다. ``cache_dir`` 을 주면 거기에 저장해 두고 다시 읽는다. 읽은 값은 전개한 값과
    비트 단위로 같다 (float64 그대로 저장). 열쇠는 인스턴스 fingerprint 와 표적·입자
    수·seed·스텝·시간 간격·지연·사전분포·지형 결합 전부다 — 하나라도 다르면 새로 전개한다.

    ``stepping`` 은 표적을 **어떤 시간 간격으로 전개하는가**다.
        "model"  셀 모형이 전이행렬을 만들 때와 **같은 간격**으로 전개한다 — 슬라이스마다
                 ``predict(slice_s)`` 가 하는 대로 운동모형의 적분 간격(30 초)씩, 나머지를
                 한 번. 그 사이의 위치는 직선으로 잇는다 (기본, 2026-10-08).
                 ``slice_s`` 를 주지 않으면 ``instance.slice_s`` 를 쓴다.
        "fine"   평가기의 스텝 ``dt``(약 5 초)마다 ``predict(dt)`` 를 부른다. 2026-10-08
                 이전의 전개이며, 그 영향을 재는 비교용으로만 남긴다.
    둘은 같은 과정이 아니다. 운동모형에는 **한 번 부를 때마다** 적용되는 항(지형이
    정지 모드로 끌어당기는 확률, 선회율의 감쇠)이 있어서, 잘게 부를수록 정지가 늘고
    순변위가 줄어든다 (Tank 8x8 에서 정지 모드 비율 0.11 대 0.27, 순변위 −16 %).
    선언된 모형은 적분 간격으로 정의되어 있고 셀 모형이 그것을 쓴다.
    ``tools/target_step_probe.py`` 가 두 전개의 차이를 잰다.
    """
    if stepping not in ("fine", "model"):
        raise ValueError("stepping 은 'fine' | 'model' 이다")
    if slice_s is None:
        slice_s = float(instance.slice_s)
    from cpp_search.planning.terrain_belief import TerrainParticleBelief

    path = None
    if cache_dir:
        key = json.dumps([
            "trace-v1", instance.fingerprint, int(index), str(target.profile.name),
            repr(target.profile), int(particle_count), int(seed), int(steps), repr(float(dt)),
            repr(float(belief_delay_s)), repr(prior), repr(terrain_coupling),
            repr(float(mission.search_radius_m)),
        ] + ([] if stepping == "fine" else [stepping, repr(float(slice_s))]),
            ensure_ascii=False)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        path = pathlib.Path(cache_dir) / f"trace-{digest}.npz"
        if path.exists():
            with np.load(path) as saved:
                return saved["weights"], saved["positions"], saved["escaped"]

    belief = TerrainParticleBelief(
        mission, target.profile, prior=prior, particle_count=particle_count,
        seed=(seed ^ (0x5350_5800 + index)) ^ 0x00C3_0000,
        radial_step_m=max(mission.search_radius_m, 1.0), angular_bin_count=8,
        terrain=instance.terrain, terrain_coupling=terrain_coupling, roughening=False)
    if belief_delay_s > 0:
        belief.filter.predict(belief_delay_s)
    filt = belief.filter
    weights = np.asarray(filt.weights, dtype=float)
    weights = weights / weights.sum()
    positions = np.empty((steps, weights.size, 2), dtype=float)
    escaped = np.empty((steps, weights.size), dtype=bool)

    def snapshot():
        return (np.array([[p.x, p.y] for p in filt.points], dtype=float),
                np.asarray(filt.escaped, dtype=bool).copy())

    if stepping == "fine":
        for step in range(steps):
            positions[step], escaped[step] = snapshot()
            filt.predict(dt)
    else:
        # 셀 모형과 같은 전개: 슬라이스 하나를 적분 간격씩 끊어 전진하고 그 사이를 직선으로 잇는다.
        per_slice = int(round(slice_s / dt))
        chunk_s = float(filt.motion.step_s) if filt.motion is not None else float(slice_s)
        for first in range(0, steps, per_slice):
            knots_t, knots = [0.0], [snapshot()]
            remaining = float(slice_s)
            while remaining > 1e-9:
                piece = min(chunk_s, remaining)
                filt.predict(piece)
                remaining -= piece
                knots_t.append(float(slice_s) - max(remaining, 0.0))
                knots.append(snapshot())
            for j in range(min(per_slice, steps - first)):
                t = j * dt
                k = max(i for i, start in enumerate(knots_t[:-1]) if start <= t + 1e-9)
                share = (t - knots_t[k]) / (knots_t[k + 1] - knots_t[k])
                positions[first + j] = knots[k][0] + share * (knots[k + 1][0] - knots[k][0])
                escaped[first + j] = knots[k][1]        # 그 구간이 시작할 때의 상태
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(scratch, weights=weights, positions=positions, escaped=escaped)
        os.replace(scratch, path)               # 다 쓴 뒤에 이름을 바꾼다 (반쯤 쓴 파일 방지)
    return weights, positions, escaped


def evaluate(instance, targets, plans, *, mission, prior, terrain_coupling,
             particle_count: int, dt_s: float, seed: int,
             belief_delay_s: float = 0.0, erase=None, law=None,
             dwell=None, entry=None, start=None, trace_cache=None,
             target_stepping: str = "model", progress=print) -> dict:
    """계획들을 연속 공간에서 평가한다. 표적별·계획별 탐지확률을 돌려준다.

    계획 여러 개를 주면 **같은 입자 집합**으로 동시에 평가한다(공통 난수).
    표적 궤적은 계획에 의존하지 않으므로 유효하며, 짝지어져 분산이 준다.

    ``erase`` 는 계획마다 하나씩, 셀 모형의 표적 전제를 얼마나 흉내 낼지다.
        "none"   입자의 실제 위치에서 hazard 를 잰다 (기본)
        "slice"  슬라이스 시작마다 그 셀 안에서 위치를 새로 뽑고 슬라이스 동안 고정
        "step"   입자 스텝마다 (슬라이스 시작 셀 안에서) 위치를 새로 뽑는다
        "mean"   위치 대신, 항적이 그 슬라이스에 슬라이스 시작 셀 전체에 뿌린
                 hazard 의 **셀 평균**을 준다 — 띠 모양을 지우고 고르게 편 것
        "cell"   위치를 아예 쓰지 않고, 슬라이스 시작 셀에 셀 모형이 적은 hazard
                 ``Y_t[cell]`` 을 그대로 준다 — 표적 셀 열만 실제 궤적에서 온다
    표적 궤적 자체는 건드리지 않는다 — hazard 를 재는 자리만 바꾼다. 그래서
    "none" 과의 차이가 곧 "셀 안 위치를 지우면 얼마나 달라지는가"이고,
    "cell" 과 셀 모형 값의 차이가 곧 "셀 열을 Markov 사슬로 줄이면 얼마나 달라지는가"다.

    ``law`` 는 계획마다 하나씩, 측방 곡선을 읽는 법칙("exponential" | "probability",
    ``accumulate_window`` 참조). 기본은 전부 "exponential". "cell"/"mean" 소거는 셀
    모형의 지수 법칙을 전제하므로 "probability" 와 함께 쓸 수 없다.

    ``entry`` 는 계획마다 하나씩, 새 셀에 어떻게 들어가는가다 (``build_windows``).
        "centre-sync"  셀 중심까지 간 뒤 중심에서 회로를 시작하되, 슬라이스마다 정확히
                   한 슬라이스 분량만 난다 (기본, 2026-10-08). 셀 모형이 가정한 경로
                   (중심에서 중심으로)를 그대로 그린 것이라 본 표와 인증 판정에 쓴다.
                   출발점을 흔들어도 거의 흔들리지 않는다 (``tools/start_probe.py``).
        "nearest"  가장 가까운 차선 끝으로 들어가 거기서 회로를 시작한다. 실제 비행에
                   가까워 PD 가 더 나오지만 출발점에 민감하다 — **참고 열**로 나란히 싣는다.
        "centre"   셀 중심까지 간 뒤 회로를 시작한다. 이전 규약이며 지각 결함이
                   들어 있다 — 대조용.

    ``dwell`` 은 ``entry="centre"`` 에서만 뜻이 있다. 계획마다 하나씩, 같은 셀에 계속
    머물 때 슬라이스마다 셀 중심으로
    돌아갈지(True) 있던 자리에서 이어 갈지(False, 기본)다. 돌아가는 쪽이
    2026-09-27 이전의 규약이며 체류 슬라이스의 18~30 % 를 복귀 비행에 썼다.

    ``start`` 는 계획마다 하나씩, 탐색자가 어디서 임무를 시작하는가다.
        "true"  실제 출발 위치 ``instance.initial_positions`` (기본, 2026-10-08) — 셀 중심으로
                반올림하기 전의 점. 격자와 무관한 유일한 선택이다.
        "cell"  출발 셀의 중심. 셀 모형이 계획을 세울 때의 전제이며, 격자가 거칠수록
                실제 위치에서 멀다 (4x4 에서 최대 2.8 km). 그 영향을 재는 비교용.
        좌표 열  탐색자마다 (x, y) 하나씩. 출발점을 임의로 흔들어 볼 때 쓴다.
    둘의 차이와 흔들림의 크기는 ``tools/start_probe.py`` 가 잰다.

    ``target_stepping`` 은 표적 입자를 전개하는 간격이다 (``particle_trace``). 기본
    "model" 은 셀 모형과 같은 간격이고, "fine" 은 2026-10-08 이전의 5 초 전개다.

    ``trace_cache`` 는 입자 궤적을 저장해 둘 폴더다 (``particle_trace``). 주지 않으면
    환경변수 ``CPP_SEARCH_TRACE_CACHE`` 를 보고, 그것도 없으면 매번 새로 전개한다.
    결과는 어느 쪽이든 같다.
    """
    if trace_cache is None:
        trace_cache = os.environ.get("CPP_SEARCH_TRACE_CACHE") or None
    trace_cache = trace_cache or None           # 빈 문자열은 "쓰지 않는다"

    grid = instance.grid
    slice_s = float(instance.slice_s)
    steps_per_slice = max(1, int(round(slice_s / dt_s)))
    dt = slice_s / steps_per_slice
    side = float(grid.cell_area_m2) ** 0.5
    transit = float(mission.transit_speed_mps)
    search = float(mission.search_speed_mps)

    # 셀 모형 값은 표적별로 **전부** 남긴다. 최악만 남기면 표적별 대조가 깨진다.
    cell_per_target = [[float(v) for v in instance.score_paths(e[1])] for e in plans]
    cell_worst = [float(min(v)) for v in cell_per_target]

    # 계획마다 **셀 안 항로 간격**을 따로 줄 수 있다. 계획은 "어느 셀에 있으라"
    # 까지만 말하고 셀 안을 어떻게 날지는 말하지 않으므로, 그 자유도를 드러내려면
    # 같은 계획을 여러 간격으로 날려 봐야 한다. 같은 입자 집합을 쓰므로 짝지어진다.
    lanes = [(plan[2] if len(plan) > 2 else SUPPORT_HALF_M) for plan in plans]
    dwell = tuple(bool(d) for d in dwell) if dwell is not None else (False,) * len(plans)
    if len(dwell) != len(plans):
        raise ValueError("dwell 은 계획마다 하나씩이어야 한다")
    entry_modes = (tuple(str(e) for e in entry) if entry is not None
                   else ("centre-sync",) * len(plans))
    if len(entry_modes) != len(plans):
        raise ValueError("entry 는 계획마다 하나씩이어야 한다")
    start_modes = tuple(start) if start is not None else ("true",) * len(plans)
    if len(start_modes) != len(plans) or any(
            isinstance(v, str) and v not in ("cell", "true") for v in start_modes):
        raise ValueError("start 는 계획마다 'cell' | 'true' | 좌표 열 하나씩이어야 한다")

    def origin_of(mode, searcher):
        if isinstance(mode, str):
            if mode == "cell":
                return None
            point = instance.initial_positions[searcher]
            return (float(point.x), float(point.y))
        return (float(mode[searcher][0]), float(mode[searcher][1]))
    tracks = []
    for (plan, lane, dw, how, origin) in zip(plans, lanes, dwell, entry_modes, start_modes):
        paths = plan[1]
        # 간격은 스칼라이거나 **탐색자마다 하나**(그 안은 슬라이스별 열일 수 있다).
        # 정책이 셀 안 항로까지 고르는 계획이 뒤엣것을 쓴다.
        per_searcher = not np.isscalar(lane)
        tracks.append([
            build_windows(list(path), instance.problem.searchers[j].start_state, grid,
                          transit_speed=transit, search_speed=search, slice_s=slice_s,
                          lane_m=(lane[j] if per_searcher else lane),
                          cell_side=side, sub=steps_per_slice, entry=how,
                          dwell_from_centre=dw,
                          start_xy=origin_of(origin, j))
            for j, path in enumerate(paths)
        ])

    steps = instance.time_slice_count * steps_per_slice
    plan_count = len(plans)
    erase = tuple(erase) if erase is not None else ("none",) * plan_count
    if len(erase) != plan_count or any(e not in ("none", "slice", "step", "mean", "cell") for e in erase):
        raise ValueError("erase 는 계획마다 'none' | 'slice' | 'step' | 'mean' | 'cell' 하나씩이어야 한다")
    # "mean" 용: 슬라이스 t 에 항적이 셀 i 위에 실제로 뿌린 hazard 의 셀 평균.
    # 셀마다 격자점을 깔고 그 슬라이스의 창을 전부 더해 평균한다 (표적 배율 전).
    mean_hazard = [
        (_cell_mean_hazard(grid, tracks[index], instance.time_slice_count, steps_per_slice)
         if e == "mean" else None)
        for index, e in enumerate(erase)
    ]
    # "cell" 용: 셀 모형이 슬라이스 t 에 셀 i 에 적는 hazard (표적 배율 전). 배율은
    # detection_rate 와 swept_hazard 양쪽에 같은 스칼라로 걸리므로 곱하면 된다.
    cell_hazard = [
        (np.array([-np.log(joint_survival(instance.problem, plan[1], t))
                   for t in range(instance.time_slice_count)])
         if e == "cell" else None)
        for plan, e in zip(plans, erase)
    ]
    if any(e != "none" for e in erase) and not hasattr(grid, "half_side_m"):
        raise ValueError("셀 안 위치 소거는 정사각 격자에서만 지원한다")
    law = tuple(law) if law is not None else ("exponential",) * plan_count
    if len(law) != plan_count or any(l not in ("exponential", "probability") for l in law):
        raise ValueError("law 는 계획마다 'exponential' | 'probability' 하나씩이어야 한다")
    if any(l == "probability" and e in ("cell", "mean") for l, e in zip(law, erase)):
        raise ValueError("'cell'/'mean' 소거는 지수 법칙에서만 뜻이 있다")
    result = {
        "cells": instance.cell_count, "time_slices": instance.time_slice_count,
        "cell_side_m": side, "sweep_width_m": SUPPORT_HALF_M, "dt_s": dt,
        "particles": particle_count, "plans": [e[0] for e in plans],
        "target_stepping": target_stepping,
        "lane_m": [lane if np.isscalar(lane) else [list(row) for row in lane]
                   for lane in lanes],
        "erase": list(erase), "law": list(law),
        "dwell_from_centre": list(dwell), "entry": list(entry_modes),
        "start": [v if isinstance(v, str) else [list(map(float, xy)) for xy in v]
                  for v in start_modes],
        "cell_model_pd": cell_worst, "cell_model_pd_per_target": cell_per_target,
        "targets": [],
    }

    for k, target in enumerate(targets):
        multiplier = float(np.asarray(instance.target_models[k].hazard_multiplier, dtype=float))
        started = time.perf_counter()
        weights, positions, escaped_all = particle_trace(
            instance, target, k, mission=mission, prior=prior,
            terrain_coupling=terrain_coupling, particle_count=particle_count, seed=seed,
            steps=steps, dt=dt, belief_delay_s=belief_delay_s, cache_dir=trace_cache,
            stepping=target_stepping, slice_s=slice_s)
        survival = np.ones((plan_count, weights.size))
        erase_rng = np.random.default_rng((seed ^ 0x00E5_A5E0) + 7919 * k)
        slice_cell_xy = None                   # 슬라이스 시작 시점의 실제 위치 (셀 결정용)
        virtual = {}                           # erase 종류별 가상 위치
        for step in range(steps):
            px = positions[step, :, 0]
            py = positions[step, :, 1]
            escaped = escaped_all[step]
            if step % steps_per_slice == 0:
                slice_cell_xy = (px, py)
                if "slice" in erase:
                    virtual["slice"] = _virtual_positions(grid, *slice_cell_xy, erase_rng)
            if "step" in erase:
                virtual["step"] = _virtual_positions(grid, *slice_cell_xy, erase_rng)
            for index in range(plan_count):
                hazard = np.zeros(weights.size)
                if erase[index] in ("cell", "mean"):
                    if step % steps_per_slice == 0:      # 슬라이스에 한 번, 셀 모형처럼
                        table = (cell_hazard if erase[index] == "cell" else mean_hazard)[index]
                        cells = _cell_index(grid, *slice_cell_xy)
                        hazard = multiplier * table[step // steps_per_slice][cells]
                else:
                    qx, qy = virtual.get(erase[index], (px, py))
                    for j in range(len(tracks[index])):
                        accumulate_window(qx, qy, tracks[index][j][step], multiplier, hazard,
                                          law=law[index])
                hazard[escaped] = 0.0          # 격자 밖으로 나간 입자는 탐지 불가
                survival[index] *= np.exp(-hazard)
        nondetection = (weights[None, :] * survival).sum(axis=1)
        best = int(np.argmin(nondetection))
        paired_se = []
        for index in range(plan_count):
            diff = survival[index] - survival[best]
            mean = float((weights * diff).sum())
            paired_se.append(float(np.sqrt((weights ** 2 * (diff - mean) ** 2).sum())))
        name = instance.target_names[k].split("/")[-1]
        result["targets"].append({
            "target": name, "nondetection": nondetection.tolist(),
            "pd": (1.0 - nondetection).tolist(), "paired_se_vs_best": paired_se,
            "best_plan_index": best, "seconds": time.perf_counter() - started,
        })
        for index, plan in enumerate(plans):
            cell = cell_per_target[index][k]
            progress(f"  {name:17} {plan[0]:14} 셀모형 {cell:.5f} | 모형밖 "
                     f"{1 - nondetection[index]:.5f} | 차 {(1 - nondetection[index]) - cell:+.5f}")
    return result
