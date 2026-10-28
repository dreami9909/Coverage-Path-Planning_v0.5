"""탐색자 항적 — 계획된 셀 순서를 **연속 공간의 꺾은선**으로 편다.

셀 모형은 "이 셀에 얼마나 소인했나"만 알지만, 모형 밖 평가는 기체가 실제로
어디를 지났는지를 알아야 한다. 이 파일이 그 변환을 맡는다. 셀 안에서는 닫힌
보스트로페돈 회로를 돌고, 셀을 옮길 때는 중심을 잇는 직선으로 건너간다.

다섯 가지 결함을 고친 판이다 (2026-09~10 검토).

  1) 체류 중 사행이 슬라이스마다 처음부터 다시 시작해 같은 띠만 반복해 쓸었다.
     계획의 54~70 %가 제자리 호라 이 결함이 측정을 지배했다. 이제 이어서 깐다.
  2) 덮개를 열어 두어 한 바퀴 돌 때마다 끝점->시작점으로 셀을 가로지르는
     가짜 구간이 생겼다(소인 총량 +11 %). 닫힌 회로로 바꿨다.
  3) 등호길이 재표집이 모서리를 깎아 소인을 1~2 % 잃었다. 창 경계에서
     정확히 잘라 원래 꼭짓점을 그대로 쓴다.
  4) 같은 셀에 계속 머무는데도 슬라이스마다 셀 중심으로 돌아갔다. 셀 중심을
     매 슬라이스의 항법 지점으로 쓴 탓이며, 인증된 계획에서 체류 슬라이스의
     18~30 % 를 그 복귀 비행이 먹었다. 이제 있던 자리에서 회로를 이어 간다
     (2026-09-27). 소인 거리가 계획에 따라 2~36 % 늘고, 체류가 긴 계획일수록
     많이 늘어난다 (``tools/probes/dwell_probe.py``).

  5) 셀을 옮길 때 지금 위치에서 **다음 셀의 중심**까지 날아간 뒤 회로를 시작했다.
     회로 위 위치가 멀면 그 거리가 한 슬라이스에 날 수 있는 거리를 넘는데,
     넘친 만큼을 다음 슬라이스에서 갚지 않아 항적이 계획보다 계속 뒤처졌다
     (인증된 계획에서 이동의 18~27 % 가 지각, 최대 0.57 슬라이스). 이제 다음
     셀의 **가장 가까운 차선 끝**으로 들어가 그 자리에서 회로를 시작하고,
     슬라이스마다 정확히 한 슬라이스 분량만 난다 (2026-10-05,
     ``tools/audit/track_audit.py``). 옛 규약은 ``entry="centre"`` 로 남겨 둔다.

``planning/continuous_mc.py`` 가 이 항적으로 탐지를 계산한다. 항적이 계획대로
나는지는 ``tools/audit/track_audit.py`` 와 ``tests/test_searcher_track.py`` 가 확인한다.
"""
import numpy as np
from math import hypot


def cell_cover_polyline(cx, cy, side, lane, start, axis="x", wide_side_first=False):
    """셀을 덮는 닫힌 보스트로페돈 회로. ``start`` 에서 출발/복귀.

    ``axis`` 는 차선이 달리는 방향이다 ("x" 가로, "y" 세로). 첫 차선은 ``start``
    에서 **더 먼** 가장자리 쪽으로 깐다 — 가장자리에서 들어왔으면 차선 하나를
    온전히 쓸고, 중심에서 시작하면 오른쪽(위쪽)으로 반 차선을 쓴다.

    차선 순서는 ``start`` 쪽 차선에서 시작해 한쪽 끝까지 간 뒤 반대쪽으로 건너뛴다.
    ``wide_side_first`` 가 참이면 **남은 차선이 더 많은 쪽**으로 먼저 가서 건너뛰기를
    늦춘다 (``entry="nearest"`` 가 쓴다). 거짓이면 항상 위쪽 먼저다 — 2026-10-05
    이전 규약이 쓰던 순서이며 그 수치를 그대로 재현하려고 남긴다.
    """
    if axis == "y":
        return [(x, y) for (y, x) in
                cell_cover_polyline(cy, cx, side, lane, (start[1], start[0]), "x",
                                    wide_side_first)]
    half = 0.5 * side
    n = max(1, int(side // lane))
    step = side / n
    ys = [cy - half + (i + 0.5) * step for i in range(n)]
    i0 = min(range(n), key=lambda i: abs(ys[i] - start[1]))
    if not wide_side_first or n - 1 - i0 >= i0:
        seq = list(range(i0, n)) + list(range(i0 - 1, -1, -1))
    else:
        seq = list(range(i0, -1, -1)) + list(range(i0 + 1, n))
    pts = [(float(start[0]), float(start[1]))]
    xl, xr = cx - half, cx + half
    cur_x = float(start[0])
    # 더 먼 가장자리 쪽으로. 한가운데(반올림 오차 포함)면 오른쪽.
    toward_right = (xr - cur_x) >= (cur_x - xl) - 1e-6
    for j, i in enumerate(seq):
        y = ys[i]
        pts.append((cur_x, y))
        tgt = xr if ((j % 2 == 0) == toward_right) else xl
        pts.append((tgt, y))
        cur_x = tgt
    pts.append((float(start[0]), float(start[1])))
    return pts


def nearest_entry(cx, cy, side, lane, pos):
    """셀에서 ``pos`` 에 가장 가까운 **차선 끝**과 그때의 차선 방향.

    세로 변으로 들어가면 차선은 가로, 가로 변으로 들어가면 차선은 세로다 —
    들어온 방향 그대로 첫 차선을 깔 수 있게 한다. 이미 그 변 **위에** 서 있어도
    같은 규칙을 쓴다 (앞 셀의 차선을 끝까지 쓸고 나면 그 자리가 곧 다음 셀의
    변이다). 그래야 동쪽으로 가는 경로와 북쪽으로 가는 경로가 서로 거울상이 된다.
    셀 안쪽이면 그 자리에서 가로 차선으로 시작한다.
    """
    half = 0.5 * side
    n = max(1, int(side // lane))
    step = side / n
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    dx = max(x0 - pos[0], 0.0, pos[0] - x1)
    dy = max(y0 - pos[1], 0.0, pos[1] - y1)

    def lane_at(low, value):
        index = min(max(int((value - low) // step), 0), n - 1)
        return low + (index + 0.5) * step

    tolerance = 1e-6
    if dx <= tolerance and dy <= tolerance:
        on_vertical = min(abs(pos[0] - x0), abs(pos[0] - x1)) <= tolerance
        on_horizontal = min(abs(pos[1] - y0), abs(pos[1] - y1)) <= tolerance
        if on_vertical:
            return (float(pos[0]), lane_at(y0, pos[1])), "x"
        if on_horizontal:
            return (lane_at(x0, pos[0]), float(pos[1])), "y"
        return (float(pos[0]), float(pos[1])), "x"
    if dx >= dy:
        return ((x0 if pos[0] < x0 else x1), lane_at(y0, pos[1])), "x"
    return (lane_at(x0, pos[0]), (y0 if pos[1] < y0 else y1)), "y"


def _cum(pts):
    """길이 0 구간 제거. 남기면 cum 이 단조증가가 아니라 보간이 깨진다."""
    P = np.asarray(pts, dtype=float)
    if len(P) > 1:
        keep = np.concatenate([[True], np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])) > 1e-9])
        P = P[keep]
    if len(P) == 1:
        return P, np.array([0.0])
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    return P, np.concatenate([[0.0], np.cumsum(seg)])


def _at(P, cum, s):
    return np.array([np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])])


def _advance(P, cum, s0, length):
    """호길이 s0 에서 length 만큼 진행한 **정확한** 꺾은선. 회로를 돌면 감는다."""
    total = cum[-1]
    out = []
    if total <= 1e-9 or length <= 1e-12:
        return [tuple(P[0])] if total <= 1e-9 else [tuple(_at(P, cum, s0 % total))]
    s = s0 % total
    out.append(tuple(_at(P, cum, s)))
    left = length
    while left > 1e-9:
        take = min(left, total - s)
        end = s + take
        # s..end 사이의 원래 꼭짓점을 그대로 넣는다
        for idx in np.flatnonzero((cum > s + 1e-9) & (cum < end - 1e-9)):
            out.append((float(P[idx, 0]), float(P[idx, 1])))
        out.append(tuple(_at(P, cum, end)))
        left -= take
        s = 0.0 if end >= total - 1e-9 else end
        if s == 0.0 and left > 1e-9:
            out.append(tuple(P[0]))
    return out


def build_windows(path, start_state, grid, *, transit_speed, search_speed,
                  slice_s, lane_m, cell_side, sub, entry="nearest",
                  dwell_from_centre=False, ledger=None, start_xy=None):
    """입자 스텝 창마다 (점 배열) 목록. 창 하나가 dt = slice_s/sub 에 해당.

    ``entry`` 는 **새 셀에 어떻게 들어가는가**다.
        "nearest"  그 셀에서 가장 가까운 차선 끝으로 들어가 거기서 회로를 시작한다
                   (기본, 2026-10-05). 슬라이스마다 정확히 한 슬라이스 분량을 난다.
                   인접 셀이면 항상 제시간에 닿고, 같은 셀에 머물면 회로를 이어 간다.
        "centre-sync"  그 셀의 중심까지 간 뒤 중심에서 회로를 시작하되, 슬라이스마다
                   정확히 한 슬라이스 분량만 난다. 못 닿으면 다음 슬라이스가 남은
                   거리를 먼저 날고 그만큼 덜 쓴다. **경로는 이전 규약 그대로 두고
                   시간 계산만 고친 것**으로, 결함 수정의 몫과 진입점 변경의 몫을
                   가르는 대조용이다 (``tools/probes/entry_probe.py``).
        "centre"   그 셀의 중심까지 간 뒤 중심에서 회로를 시작한다. 2026-10-05
                   이전의 규약이며 **지각 결함**(머리말 5)이 그대로 들어 있다.
                   결함의 크기를 재는 대조용으로만 쓴다. 아래 ``dwell_from_centre``
                   는 이 규약에서만 뜻이 있다.

    ``lane_m`` 은 스칼라이거나 **슬라이스마다 하나씩**인 열이다. 열을 주면 셀에
    **들어가는 순간**의 값을 그 체류 동안 쓴다 — 셀 안 회로를 도는 중에 간격을
    바꿀 수는 없기 때문이다. 정책이 항로를 고르는 실험이 이 통로를 쓴다
    (``learning/continuous_env``).

    ``dwell_from_centre`` 는 **같은 셀에 계속 머물 때** 슬라이스마다 셀 중심으로
    돌아갈지다.
        False  돌아가지 않고 있던 자리에서 회로를 이어 간다 (기본, 2026-09-27).
        True   돌아간다. 2026-09-27 이전의 규약이며 부록 대조용으로 남겨 둔다.
    계획은 셀 안 항로를 정하지 않으므로 둘 다 평가기의 규약이다.

    ``start_xy`` 는 임무를 시작하는 **실제 위치**다. 주지 않으면 출발 셀의 중심에서
    시작한다 — 셀 모형이 그렇게 전제하기 때문이다. 실제 출발점(발사 고리 위의 점)은
    셀 중심에서 격자가 거칠수록 멀리 떨어져 있으므로, 그 반올림이 결과를 얼마나
    움직이는지 재려면 이 값을 준다 (``tools/probes/start_probe.py``). ``entry="centre"`` 는
    받지 않는다.

    ``ledger`` 에 빈 리스트를 주면 슬라이스마다 장부 한 줄을 적어 준다 — 그
    슬라이스에 깐 항적 길이와, 슬라이스 끝에서 계획보다 얼마나 뒤처졌는지
    (``lag_m``). 항적 자체는 바뀌지 않는다 (``tools/audit/track_audit.py``).
    """
    T = len(path)
    lanes = ([float(lane_m)] * T if np.isscalar(lane_m)
             else [float(v) for v in lane_m])
    if len(lanes) < T:
        raise ValueError("lane_m 열은 슬라이스마다 하나씩이어야 한다")
    if entry in ("nearest", "centre-sync"):
        if dwell_from_centre:
            raise ValueError("dwell_from_centre 는 entry='centre' 에서만 쓴다")
        return _fly_synchronised(path, start_state, grid, transit_speed=transit_speed,
                                 search_speed=search_speed, slice_s=slice_s, lanes=lanes,
                                 cell_side=cell_side, sub=sub, ledger=ledger,
                                 via_centre=(entry == "centre-sync"), start_xy=start_xy)
    if entry != "centre":
        raise ValueError(f"entry 는 'nearest' | 'centre-sync' | 'centre' 다: {entry!r}")
    if start_xy is not None:
        raise ValueError("start_xy 는 entry='centre' 에서 쓰지 않는다")
    dt_len = search_speed * slice_s / sub      # 창 하나의 항적 길이 (속도 일정)
    windows = []
    cover = None
    consumed = 0.0
    cur_cell = None
    buf = []                                    # 아직 창에 안 담긴 (점, 누적길이)
    a0 = grid.center_of(start_state)
    pos = (float(a0.x), float(a0.y))

    def emit(seg_pts):
        """seg_pts 를 창 단위로 잘라 windows 에 넣는다."""
        nonlocal buf
        for p in seg_pts:
            buf.append(p)
        # buf 를 길이 기준으로 창에 배분
        while True:
            Q, qc = _cum(buf)
            if len(Q) < 2 or qc[-1] < dt_len - 1e-9:
                break
            cut = dt_len
            idx = np.flatnonzero(qc < cut - 1e-9)
            head = [tuple(Q[i]) for i in idx] + [tuple(_at(Q, qc, cut))]
            windows.append(np.asarray(head, dtype=float))
            rest = [tuple(_at(Q, qc, cut))] + [tuple(Q[i]) for i in np.flatnonzero(qc > cut + 1e-9)]
            buf = rest

    buf = [pos]
    laid = 0.0                                  # 장부용: 지금까지 깐 항적 길이
    for t in range(T):
        prev = start_state if t == 0 else path[t - 1]
        dest = path[t]
        a = grid.center_of(prev); b = grid.center_of(dest)
        # 이동 거리는 **실제 현재 위치**에서 잰다. 셀 중심 기준으로 재면
        # UAV 가 사행 중간에 있을 때 총 항적이 v*slice_s 와 어긋난다.
        here = buf[-1]
        staying = dest == cur_cell and cover is not None
        # 같은 셀에 계속 머무는데 중심으로 돌아갈 이유는 없다. 이미 회로 위에 있다.
        dist = 0.0 if (staying and not dwell_from_centre) else hypot(
            b.x - here[0], b.y - here[1])
        transit_s = dist / transit_speed if transit_speed > 0 else 0.0
        dwell_len = max(0.0, search_speed * (slice_s - transit_s))
        if dist > 1e-9:
            emit([(float(b.x), float(b.y))])
        if dest != cur_cell or cover is None:
            cur_cell = dest
            # 간격은 **들어가는 순간**의 값으로 굳는다. 다시 들어온 셀도 중심에서 처음부터 깐다.
            cover = _cum(cell_cover_polyline(b.x, b.y, cell_side, lanes[t], (b.x, b.y)))
            consumed = 0.0
        if dwell_len > 1e-9:
            P, cum = cover
            pts = _advance(P, cum, consumed, dwell_len)
            emit(pts[1:])
            consumed += dwell_len
        if ledger is not None:
            laid += dist + dwell_len
            ledger.append({
                "slice": t, "cell": int(dest), "staying": bool(staying),
                "from_xy": (float(here[0]), float(here[1])),
                "transit_m": float(dist), "dwell_m": float(dwell_len),
                "late": bool(transit_s > slice_s + 1e-9),
                "lag_m": float(laid - search_speed * slice_s * (t + 1)),
            })
    while len(windows) < T * sub:
        windows.append(np.asarray(buf if len(buf) > 1 else [buf[0], buf[0]], dtype=float))
        buf = [buf[-1]]
    return windows[: T * sub]


def _split(points, times, parts, duration):
    """시각이 붙은 꺾은선을 ``parts`` 개의 **같은 시간** 창으로 자른다. 꼭짓점은 그대로."""
    P = np.asarray(points, dtype=float)
    tc = np.asarray(times, dtype=float)
    out = []
    for k in range(parts):
        t0, t1 = duration * k / parts, duration * (k + 1) / parts
        inner = np.flatnonzero((tc > t0 + 1e-9) & (tc < t1 - 1e-9))
        head = (np.interp(t0, tc, P[:, 0]), np.interp(t0, tc, P[:, 1]))
        tail = (np.interp(t1, tc, P[:, 0]), np.interp(t1, tc, P[:, 1]))
        out.append(np.asarray([head] + [tuple(P[i]) for i in inner] + [tail], dtype=float))
    return out


def _fly_synchronised(path, start_state, grid, *, transit_speed, search_speed, slice_s,
                      lanes, cell_side, sub, ledger, via_centre=False, start_xy=None):
    """슬라이스 하나가 정확히 ``slice_s`` 초인 항적 (``entry`` = "nearest" | "centre-sync").

    슬라이스마다 지금 위치에서 시작해 (1) 새 셀이면 진입점까지 직선으로 가고
    (2) 남은 시간 동안 회로를 돈다. 못 닿으면 닿는 데까지만 가고 다음 슬라이스가
    그 자리에서 이어받는다 — 넘친 시간이 쌓이지 않는다.

    진입점은 ``via_centre`` 가 거짓이면 가장 가까운 차선 끝, 참이면 셀 중심이다.
    중심으로 들어갈 때의 회로는 이전 규약(``entry="centre"``)과 **같은 모양**이다.
    """
    if start_xy is None:
        a0 = grid.center_of(start_state)
        pos = (float(a0.x), float(a0.y))
    else:
        pos = (float(start_xy[0]), float(start_xy[1]))
    cur_cell, cover, consumed = None, None, 0.0
    windows = []
    for t, dest in enumerate(path):
        pts, times = [pos], [0.0]
        clock = 0.0
        here = pos
        staying = dest == cur_cell and cover is not None
        transit_m = dwell_m = 0.0
        late = False
        if not staying:
            b = grid.center_of(dest)
            if via_centre:
                target, axis = (float(b.x), float(b.y)), "x"
            else:
                target, axis = nearest_entry(b.x, b.y, cell_side, lanes[t], pos)
            dist = hypot(target[0] - pos[0], target[1] - pos[1])
            need_s = dist / transit_speed if transit_speed > 0 else 0.0
            if need_s > slice_s + 1e-9:
                # 한 슬라이스에 못 닿는다. 닿는 데까지 가고 다음 슬라이스에 잇는다.
                share = slice_s / need_s
                pos = (pos[0] + share * (target[0] - pos[0]),
                       pos[1] + share * (target[1] - pos[1]))
                pts.append(pos); times.append(slice_s)
                clock = slice_s
                transit_m = dist * share
                cur_cell, cover, late = None, None, True
            else:
                if dist > 1e-9:
                    pts.append(target); times.append(need_s)
                pos, clock, transit_m = target, need_s, dist
                cur_cell = dest
                # 간격은 **들어가는 순간**의 값으로 굳는다.
                cover = _cum(cell_cover_polyline(b.x, b.y, cell_side, lanes[t],
                                                 target, axis,
                                                 wide_side_first=not via_centre))
                consumed = 0.0
        if slice_s - clock > 1e-9 and cover is not None:
            P, cum = cover
            dwell_m = search_speed * (slice_s - clock)
            seg = _advance(P, cum, consumed, dwell_m)
            run = clock
            for prev, nxt in zip(seg[:-1], seg[1:]):
                run += hypot(nxt[0] - prev[0], nxt[1] - prev[1]) / search_speed
                pts.append(nxt); times.append(run)
            times[-1] = slice_s                 # 반올림 오차를 슬라이스 끝에 맞춘다
            consumed += dwell_m
            pos = (float(seg[-1][0]), float(seg[-1][1]))
        if len(pts) == 1:                       # 제자리 (속도 0 등)
            pts.append(pos); times.append(slice_s)
        windows.extend(_split(pts, times, sub, slice_s))
        if ledger is not None:
            ledger.append({
                "slice": t, "cell": int(dest), "staying": bool(staying),
                "from_xy": (float(here[0]), float(here[1])),
                "transit_m": float(transit_m), "dwell_m": float(dwell_m),
                "late": bool(late), "lag_m": 0.0,
            })
    return windows
