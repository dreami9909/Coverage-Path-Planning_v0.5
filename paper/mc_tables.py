"""모형 밖 MC 에서 나온 표를 **결과 파일에서 직접** 다시 만든다.

손으로 옮겨 적으면 한 칸씩 어긋난다. 이 스크립트는 원고 HTML 에서 표를 번호로 찾아
머리줄과 몸통을 갈아 끼우고, 본문이 인용하는 파생 통계(범위·상관)를 화면에 찍는다.
문장은 사람이 고친다 — 숫자가 바뀌면 해석도 다시 봐야 하기 때문이다.

    표 1    인증 계획의 셀 모형 값과 MC 값 (본 표 규약 + 참고 규약 ± 출발점 폭)   mc2-*.json, start-*.json
    표 2    같은 것을 거동별로                                 mc2-*.json
    표 4    보상을 재는 자리에 따른 계획의 성능 (교차 채점)     entry-*.json
    표 5    셀 안 항로를 정책이 고를 때                         mappo3-*.json
    표 A.19 항적 규약 세 가지의 대조 (결함 수정 / 진입점)       entry-*.json, track-audit*-*.json
    표 A.20 차이의 네 항 분해                                  erase-*.json
    표 A.21 측방 곡선의 두 읽기                                law-*.json

없는 결과 파일이 있으면 그 표는 건드리지 않고 무엇이 없는지 말한다.

실행:  python3 paper/mc_tables.py            (원고를 고친다)
       python3 paper/mc_tables.py --dry      (통계만 찍는다)
"""
import json, pathlib, re, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from conditions import available_conditions                 # noqa: E402

CONDITIONS = available_conditions()         # 계획이 저장된 조건만 표에 올린다

ST = ROOT / "results/studies"
HTML = next((ROOT / "paper").glob("*초고.html"))
DRY = "--dry" in sys.argv


def load(name):
    path = ST / name
    return json.load(open(path)) if path.exists() else None


def cond_label(case, n):
    return f"{case} · {n} × {n}"


def num(value, digits=4, signed=False):
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return f'<td class="num">{text}</td>'


def cell(text=""):
    return f"<td>{text}</td>"


def row(cells):
    return "<tr>" + "".join(cells) + "</tr>"


def set_table(html, number, caption, head, rows):
    """번호가 ``number`` 인 표의 캡션·머리줄·몸통을 갈아 끼운다."""
    mark = f"<caption>표 {number}."
    at = html.find(mark)
    if at < 0:
        raise SystemExit(f"원고에 표 {number} 가 없다")
    start = html.rfind("<table>", 0, at)
    end = html.find("</table>", at) + len("</table>")
    body = "\n".join(rows)
    block = (f"<table>\n<caption>표 {number}. {caption}</caption>\n"
             f"<thead>{head}</thead>\n<tbody>\n{body}\n</tbody>\n</table>")
    return html[:start] + block + html[end:]


# ------------------------------------------------------------------ 자료
def gather():
    """조건마다 필요한 값을 한데 모은다. 탐지폭/셀 순으로 늘어놓는다."""
    out = []
    for case, n, plan_file in CONDITIONS:
        tag = f"{case}-{n}"
        plan = load(plan_file)
        item = {"case": case, "n": n, "tag": tag, "label": cond_label(case, n),
                "gap": float(plan["gap"]), "certified": float(plan["gap"]) <= 0.01,
                "mc2": load(f"mc2-{tag}.json"), "entry": load(f"entry-{tag}.json"),
                "start": load(f"start-{tag}.json"),
                "erase": load(f"erase-{tag}.json"), "law": load(f"law-{tag}.json"),
                "mappo3": load(f"mappo3-{tag}.json"),
                "audit": {key: load(f"track-audit{suffix}-{tag}.json")
                          for key, suffix in (("nearest", ""), ("centre-sync", "-centre-sync"),
                                              ("centre", "-centre"))}}
        out.append(item)
    with_wc = [c for c in out if c["mc2"]]
    for c in with_wc:
        c["wc"] = c["mc2"]["sweep_width_m"] / c["mc2"]["cell_side_m"]
    # 탐지폭/셀을 아는 조건을 그 순서로, 모르는 조건은 선언 순서로 뒤에
    return sorted(with_wc, key=lambda c: c["wc"]) + [c for c in out if not c["mc2"]]


def mc2_values(c):
    """mc2 의 (거동, 셀 모형, MC 본 표 규약 = 중심 경유, MC 참고 규약 = 최근접 진입) 목록."""
    d = c["mc2"]
    index = {name: i for i, name in enumerate(d["plans"])}
    base, alt = index["중심 경유"], index["최근접 진입"]
    assert d["entry"][base] == "centre-sync" and d["entry"][alt] == "nearest"
    assert d.get("target_stepping") == "model" and all(s == "true" for s in d["start"][:2])
    return [(t["target"], d["cell_model_pd_per_target"][base][k], t["pd"][base], t["pd"][alt])
            for k, t in enumerate(d["targets"])]


def start_spread(c, entry):
    """출발점을 출발 셀 안에서 흔들었을 때 최저 PD 의 표준편차 (tools/probes/start_probe.py). 없으면 None."""
    d = c["start"]
    if not d:
        return None
    return float(next(r for r in d["rows"] if r["entry"] == entry)["jitter_worst_std"])


# ------------------------------------------------------------------ 표 1·2
def tables_1_2(conds):
    rows1, rows2, stats = [], [], []
    for c in conds:
        values = mc2_values(c)
        lo_cell = min(values, key=lambda v: v[1])
        lo_base = min(values, key=lambda v: v[2])
        lo_alt = min(values, key=lambda v: v[3])
        gap = f"{100 * c['gap']:.3f} %" if c["certified"] else f"{100 * c['gap']:.1f} %"
        spread = start_spread(c, "nearest")
        alt_text = (f'<td class="num">{lo_alt[3]:.4f} ± {spread:.3f}</td>' if spread is not None
                    else num(lo_alt[3]))
        rows1.append(row([
            cell(c["label"]), f'<td class="num">{100 * c["wc"]:.1f} %</td>',
            num(lo_cell[1]), cell(lo_cell[0]), f'<td class="num">{gap}</td>',
            num(lo_base[2]), cell(lo_base[0]), num(lo_base[2] / lo_cell[1], 3),
            alt_text, cell(lo_alt[0]), num(lo_alt[3] / lo_cell[1], 3)]))
        for k, (name, cm, base, alt) in enumerate(values):
            marks = [label for label, low in (("MILP", lo_cell), ("MC", lo_base),
                                              ("MC 참고", lo_alt)) if low[0] == name]
            rows2.append(row([
                cell(c["label"] if k == 0 else ""),
                (f'<td class="num">{100 * c["wc"]:.1f} %</td>' if k == 0 else '<td class="num"></td>'),
                cell(name), num(cm), num(base), num(base / cm, 3), num(alt), num(alt / cm, 3),
                cell(" · ".join(marks))]))
        stats.append({
            "label": c["label"], "wc": c["wc"], "gap": c["gap"], "certified": c["certified"],
            "cell": lo_cell[1], "base": lo_base[2], "alt": lo_alt[3],
            "ratio_base": lo_base[2] / lo_cell[1], "ratio_alt": lo_alt[3] / lo_cell[1],
            "per_target": values,
            "mean_ratio_base": float(np.mean([v[2] / v[1] for v in values])),
            "mean_ratio_alt": float(np.mean([v[3] / v[1] for v in values])),
            "matched_base": lo_cell[2] / lo_cell[1], "matched_alt": lo_cell[3] / lo_cell[1],
            "spread_cell": max(v[1] for v in values) - min(v[1] for v in values),
            "spread_base": max(v[2] for v in values) - min(v[2] for v in values),
            "lowest_base": lo_base[0], "lowest_alt": lo_alt[0], "lowest_cell": lo_cell[0],
            "spread_base_jitter": start_spread(c, "centre-sync"),
            "spread_alt_jitter": start_spread(c, "nearest")})
    head1 = ('<tr><th rowspan="2">조건</th><th rowspan="2">탐지폭/셀</th>'
             '<th colspan="3">셀 모형 (MILP)</th><th colspan="3">MC · 중심 경유 (본 표)</th>'
             '<th colspan="3">MC · 최근접 진입 (참고)</th></tr>'
             '<tr><th>탐지확률</th><th>거동</th><th>최적성 간격</th>'
             '<th>탐지확률</th><th>거동</th><th>비</th>'
             '<th>탐지확률 ± 출발점 폭</th><th>거동</th><th>비</th></tr>')
    head2 = ('<tr><th>조건</th><th>탐지폭/셀</th><th>거동</th><th>MILP 탐지확률</th>'
             '<th>MC · 중심 경유</th><th>비</th><th>MC · 최근접 진입</th><th>비</th>'
             '<th>표 1 에 실린 값</th></tr>')
    return (head1, rows1), (head2, rows2), stats


def report_1_2(stats):
    print("\n[표 1·2] 조건별 (탐지폭/셀, 간격, 셀 모형, MC 중심 경유, 비, MC 최근접, 비, 세 거동 평균 비)")
    for s in stats:
        print(f"  {s['label']:<14}{100 * s['wc']:>5.1f}%  간격 {100 * s['gap']:>6.3f}%  "
              f"{s['cell']:.4f} | {s['base']:.4f} {s['ratio_base']:.3f} | "
              f"{s['alt']:.4f} {s['ratio_alt']:.3f} | 평균 비 {s['mean_ratio_base']:.3f} "
              f"{s['mean_ratio_alt']:.3f} | 같은 거동 {s['matched_base']:.3f} {s['matched_alt']:.3f}"
              f" | 최저 거동 {s['lowest_cell']} / {s['lowest_base']} / {s['lowest_alt']}")
    wc = np.array([s["wc"] for s in stats])
    gap = np.array([s["gap"] for s in stats])
    for key, name in (("base", "중심 경유"), ("alt", "최근접 진입")):
        ratio = np.array([s[f"ratio_{key}"] for s in stats])
        diff = np.array([s["cell"] - s[key] for s in stats])
        mean3 = np.array([s[f"mean_ratio_{key}"] for s in stats])
        matched = np.array([s[f"matched_{key}"] for s in stats])
        closed = np.array([s["certified"] for s in stats])
        print(f"  [{name}] 비 범위 {ratio.min():.3f}~{ratio.max():.3f} | "
              f"상관 탐지폭/셀↔비 {np.corrcoef(wc, ratio)[0, 1]:+.2f}, "
              f"↔세 거동 평균 비 {np.corrcoef(wc, mean3)[0, 1]:+.2f}, "
              f"↔같은 거동 비 {np.corrcoef(wc, matched)[0, 1]:+.2f} | "
              f"간격↔차 (전체) {np.corrcoef(gap, diff)[0, 1]:+.2f}, "
              f"(인증 닫힌 조건) {np.corrcoef(gap[closed], diff[closed])[0, 1]:+.2f}")
        ratios = [v[2 if key == "base" else 3] / v[1] for s in stats for v in s["per_target"]]
        print(f"      거동별 비 {min(ratios):.3f}~{max(ratios):.3f}, "
              f"STOP_HEAVY 가 최저인 조건 "
              f"{sum(s[f'lowest_{key}'] == 'STOP_HEAVY' for s in stats)} / {len(stats)}")
    print(f"  셀 모형 거동 간 폭 최대 {max(s['spread_cell'] for s in stats):.4f}, "
          f"MC(중심 경유) 거동 간 폭 {min(s['spread_base'] for s in stats):.4f}~"
          f"{max(s['spread_base'] for s in stats):.4f}")
    have = [s for s in stats if s["spread_alt_jitter"] is not None]
    if have:
        print("  출발점 흔들림(최저 PD 표준편차): 중심 경유 "
              f"{min(s['spread_base_jitter'] for s in have):.4f}~{max(s['spread_base_jitter'] for s in have):.4f}"
              f" (평균 {np.mean([s['spread_base_jitter'] for s in have]):.4f}), 최근접 진입 "
              f"{min(s['spread_alt_jitter'] for s in have):.4f}~{max(s['spread_alt_jitter'] for s in have):.4f}"
              f" (평균 {np.mean([s['spread_alt_jitter'] for s in have]):.4f})")


# ------------------------------------------------------------------ 표 4 (교차 채점)
GROUPS = ("절단평면", "셀 학습", "연속 학습·최근접", "연속 학습·중심")
GROUP_LABEL = {"절단평면": "절단평면", "셀 학습": "셀 학습",
               "연속 학습·최근접": "연속 학습 (최근접 진입 보상)",
               "연속 학습·중심": "연속 학습 (중심 경유 보상)"}


def table_4(conds):
    rows, stats = [], []
    for c in conds:
        d = c["entry"]
        block = {}
        for g in GROUPS:
            members = [r for r in d["rows"] if r["group"] == g]
            block[g] = {
                "cell": float(np.mean([min(r["cell_model_pd_per_target"]) for r in members])),
                "near": [r["worst_pd_nearest"] for r in members],
                "sync": [r["worst_pd_centre_sync"] for r in members]}
        for k, g in enumerate(GROUPS):
            b = block[g]
            best = (lambda xs: num(max(xs)) if len(xs) > 1 else '<td class="num">—</td>')
            rows.append(row([
                cell(c["label"] if k == 0 else ""),
                cell(("닫힘" if c["certified"] else "열림") if k == 0 else ""),
                cell(GROUP_LABEL[g]), num(b["cell"]),
                num(float(np.mean(b["near"]))), best(b["near"]),
                num(float(np.mean(b["sync"]))), best(b["sync"])]))
        stats.append({"label": c["label"], "certified": c["certified"], **{
            g: {"cell": block[g]["cell"], "near": float(np.mean(block[g]["near"])),
                "sync": float(np.mean(block[g]["sync"])),
                "near_all": block[g]["near"], "sync_all": block[g]["sync"]} for g in GROUPS}})
    head = ('<tr><th rowspan="2">조건</th><th rowspan="2">인증</th><th rowspan="2">계획</th>'
            '<th rowspan="2">셀 모형 PD</th><th colspan="2">최근접 진입으로 채점</th>'
            '<th colspan="2">중심 경유로 채점</th></tr>'
            '<tr><th>연속 PD</th><th>최고 seed</th><th>연속 PD</th><th>최고 seed</th></tr>')
    return (head, rows), stats


def report_4(stats):
    print("\n[표 4] 자기 규약에서: 절단평면 / 셀 학습 / 연속 학습 (평균), 연속−절단평면, seed 중 절단평면을 넘은 수")
    for key, own, name in (("near", "연속 학습·최근접", "최근접 진입"),
                           ("sync", "연속 학습·중심", "중심 경유")):
        print(f"  [{name}]")
        for s in stats:
            spx, cellp, cont = s["절단평면"][key], s["셀 학습"][key], s[own][key]
            wins = sum(v > spx for v in s[own][f"{key}_all"])
            print(f"    {s['label']:<14}{'닫힘' if s['certified'] else '열림'}  {spx:.4f} / {cellp:.4f} / "
                  f"{cont:.4f}  연속−절단 {cont - spx:+.4f}  연속−셀 {cont - cellp:+.4f}  "
                  f"절단평면을 넘은 seed {wins}/3  학습/절단 {cont / spx:.3f}")
    print("  [교차] 다른 규약으로 채점하면 연속 학습이 잃는 값 (자기 규약 값 − 다른 규약 값)")
    for s in stats:
        a, b = s["연속 학습·최근접"], s["연속 학습·중심"]
        print(f"    {s['label']:<14} 최근접 보상 정책: 최근접 {a['near']:.4f} → 중심 {a['sync']:.4f} "
              f"({a['sync'] - a['near']:+.4f}) | 중심 보상 정책: 중심 {b['sync']:.4f} → 최근접 "
              f"{b['near']:.4f} ({b['near'] - b['sync']:+.4f})")


# ------------------------------------------------------------------ 표 5 (항로 선택)
def table_5(conds):
    rows, stats = [], []
    for c in conds:
        d = c["mappo3"]
        if not d:
            continue
        pd = np.array(d["continuous_pd_per_target"])          # [계획, 표적]
        worst = pd.min(axis=1)
        names = d["plans"]
        spx = float(worst[names.index("절단평면")])
        fixed = float(np.mean([worst[i] for i, nm in enumerate(names) if nm.startswith("항로 고정")]))
        chosen = float(np.mean([worst[i] for i, nm in enumerate(names) if nm.startswith("항로 선택")]))
        counts = {}
        for r in d["lane_chosen"]:
            for lane, count in r["lane_histogram"].items():
                counts[lane] = counts.get(lane, 0) + count
        total = sum(counts.values())
        share = " · ".join(f"{lane} m {100 * count / total:.0f} %"
                           for lane, count in sorted(counts.items(), key=lambda kv: float(kv[0])))
        rows.append(row([cell(c["label"]), num(spx), num(fixed), num(chosen),
                         num(chosen - fixed, signed=True), cell(share)]))
        stats.append((c["label"], spx, fixed, chosen, share))
    head = ('<tr><th>조건</th><th>절단평면</th><th>항로 고정</th><th>항로 선택</th>'
            '<th>차</th><th>정책이 고른 간격</th></tr>')
    return (head, rows), stats


# ------------------------------------------------------------------ 표 A.19 (규약 대조)
def table_a19(conds):
    rows, shares = [], {"timing": [], "entry": [], "timing_spx": [], "entry_spx": []}
    late = {key: [0, 0] for key in ("centre", "centre-sync", "nearest")}
    lag = 0.0
    for c in conds:
        d = c["entry"]
        spx = next(r for r in d["rows"] if r["group"] == "절단평면")
        for r in d["rows"]:
            shares["timing"].append(r["delta_worst_timing"])
            shares["entry"].append(r["delta_worst_entry"])
        shares["timing_spx"].append(spx["delta_worst_timing"])
        shares["entry_spx"].append(spx["delta_worst_entry"])
        for key in late:
            audit = c["audit"][key]
            if audit:
                for plan in audit["plans"].values():
                    late[key][0] += plan["late_moves"]; late[key][1] += plan["move_slices"]
                if key == "centre":
                    lag = max(lag, max(p["max_lag_slices"] for p in audit["plans"].values()))
        rows.append(row([cell(c["label"]), num(spx["worst_pd_centre"]),
                         num(spx["worst_pd_centre_sync"]), num(spx["worst_pd_nearest"]),
                         num(spx["delta_worst_timing"], signed=True),
                         num(spx["delta_worst_entry"], signed=True)]))
    head = ('<tr><th>조건</th><th>이전 구현</th><th>중심 경유</th><th>최근접 진입</th>'
            '<th>시간 계산 수정의 몫</th><th>진입점 변경의 몫</th></tr>')
    return (head, rows), shares, late, lag


# ------------------------------------------------------------------ 표 A.20 (분해)
def table_a20(conds):
    rows, stats = [], []
    for c in conds:
        d = c["erase"]
        idx = {e: i for i, e in enumerate(d["erase"])}
        for k, t in enumerate(d["targets"]):
            pd = t["pd"]; cm = d["cell_model_pd_per_target"][0][k]
            r = {"label": c["label"], "wc": d["sweep_width_m"] / d["cell_side_m"],
                 "target": t["target"], "cell": cm, "none": pd[idx["none"]],
                 "d_pos": pd[idx["step"]] - pd[idx["none"]],
                 "d_strip": pd[idx["mean"]] - pd[idx["step"]],
                 "d_spill": pd[idx["cell"]] - pd[idx["mean"]],
                 "d_markov": cm - pd[idx["cell"]]}
            stats.append(r)
            rows.append(row([cell(c["label"] if k == 0 else ""), cell(t["target"]),
                             num(cm), num(r["none"]), num(r["d_pos"], signed=True),
                             num(r["d_strip"], signed=True), num(r["d_spill"], signed=True),
                             num(r["d_markov"], signed=True)]))
    head = ('<tr><th>조건</th><th>거동</th><th>셀 모형</th><th>MC</th><th>Δ위치</th>'
            '<th>Δ띠</th><th>Δ유출</th><th>ΔMarkov</th></tr>')
    return (head, rows), stats


def report_a20(stats):
    strip = np.array([r["d_strip"] / r["cell"] for r in stats])
    pos = np.array([r["d_pos"] / r["cell"] for r in stats])
    stop = np.array([r["target"] == "STOP_HEAVY" for r in stats])
    by_cond = {}
    for r in stats:
        by_cond.setdefault(r["label"], []).append(r)
    wc = [rs[0]["wc"] for rs in by_cond.values()]
    strip_mean = [float(np.mean([r["d_strip"] / r["cell"] for r in rs])) for rs in by_cond.values()]
    print(f"\n[표 A.20] Δ띠/셀 모형 {strip.min():.3f}~{strip.max():.3f} (평균 {strip.mean():.3f}), "
          f"탐지폭/셀과의 상관 {np.corrcoef(wc, strip_mean)[0, 1]:+.3f}")
    print("  조건별 Δ띠/셀 모형 평균: " + ", ".join(
        f"{label} {value:.3f}" for label, value in zip(by_cond, strip_mean)))
    print(f"  Δ위치/셀 모형: STOP_HEAVY {pos[stop].min():+.3f}~{pos[stop].max():+.3f}, "
          f"움직이는 거동 {pos[~stop].min():+.3f}~{pos[~stop].max():+.3f} (평균 {pos[~stop].mean():+.3f})")
    big = max(stats, key=lambda r: abs(r["d_pos"]))
    print(f"  |Δ위치| 최대: {big['label']} {big['target']} {big['d_pos']:+.4f} "
          f"({big['d_pos'] / big['cell']:+.1%})")
    print(f"  |Δ유출| 최대 {max(abs(r['d_spill']) for r in stats):.4f}, "
          f"|ΔMarkov| 최대 {max(abs(r['d_markov']) for r in stats):.4f}")
    print(f"  Δ위치 거동 평균(조건별): " + ", ".join(
        f"{label} {np.mean([r['d_pos'] for r in rs]):+.3f}" for label, rs in by_cond.items()))


# ------------------------------------------------------------------ 표 A.21 (법칙)
def table_a21(conds):
    rows, stats = [], []
    for c in conds:
        d = c["law"]
        idx = {law: i for i, law in enumerate(d["law"])}
        values = [(t["target"], d["cell_model_pd_per_target"][0][k],
                   t["pd"][idx["exponential"]], t["pd"][idx["probability"]])
                  for k, t in enumerate(d["targets"])]
        lo_cell = min(values, key=lambda v: v[1])
        expo, prob = min(v[2] for v in values), min(v[3] for v in values)
        rows.append(row([cell(c["label"]), num(lo_cell[1]), cell(lo_cell[0]),
                         num(expo), num(expo / lo_cell[1], 3), num(prob), num(prob / lo_cell[1], 3)]))
        stats.append((c["label"], values))
    head = ('<tr><th>조건</th><th>셀 모형</th><th>셀 모형 거동</th><th>MC · hazard 읽기</th>'
            '<th>비</th><th>MC · 통과당 확률 읽기</th><th>비</th></tr>')
    return (head, rows), stats


def report_a21(stats):
    gain = [v[3] - v[2] for _, values in stats for v in values]
    moving = [v[3] / v[1] for _, values in stats for v in values if v[0] != "STOP_HEAVY"]
    stop = [(label, v[3] / v[1]) for label, values in stats for v in values if v[0] == "STOP_HEAVY"]
    print(f"\n[표 A.21] 확률 읽기 − hazard 읽기 {min(gain):+.3f}~{max(gain):+.3f} | "
          f"움직이는 거동 비 {min(moving):.2f}~{max(moving):.2f} | STOP_HEAVY 비 "
          + ", ".join(f"{label} {value:.3f}" for label, value in stop))


# ------------------------------------------------------------------ 실행
def main():
    conds = gather()
    html = HTML.read_text(encoding="utf-8")
    todo = []

    def ready(key, name):
        """그 결과가 있는 조건들. 빠진 조건이 있으면 말하고, 원고는 고치지 않게 표시한다."""
        have = [c for c in conds if c[key]]
        missing = [c["tag"] for c in conds if not c[key]]
        if missing:
            print(f"\n{name}: 아직 없는 조건 {missing} — 있는 것만으로 통계를 찍고 원고는 두었다")
        return have, not missing

    have, complete = ready("mc2", "표 1·2")
    if have:
        (h1, r1), (h2, r2), stats = tables_1_2(have)
        report_1_2(stats)
    if have and complete:
        todo += [("1", "인증된 계획의 MILP 탐지확률과 모형 밖 MC 탐지확률 — 각 모형에서 세 거동 중 "
                       "가장 낮은 값. 본 표 규약은 중심 경유, 참고 규약(최근접 진입)은 출발점을 출발 셀 "
                       "안에서 12번 흔들었을 때의 표준편차를 ± 로 붙였다", h1, r1),
                 ("2", "거동별 MILP 탐지확률과 모형 밖 MC 탐지확률", h2, r2)]

    have, complete = ready("entry", "표 4·A.19")
    if have:
        (h4, r4), stats4 = table_4(have)
        report_4(stats4)
    if have and complete:
        todo.append(("4", "보상을 어디서 재는가에 따른 계획의 성능 — 모형 밖 MC 로 채점, 학습 seed 3 개 "
                          "평균. 연속 학습 정책은 보상을 받은 규약이 둘이고, 모든 계획을 두 규약으로 채점하였다",
                     h4, r4))
    if have:
        (h19, r19), shares, late, lag = table_a19(have)
        t, e = np.array(shares["timing"]), np.array(shares["entry"])
        print(f"\n[표 A.19] 계획 {t.size}개: 시간 계산 수정의 몫 {t.min():+.4f}~{t.max():+.4f} "
              f"(평균 절댓값 {np.abs(t).mean():.4f}), 진입점의 몫 {e.min():+.4f}~{e.max():+.4f} "
              f"(평균 절댓값 {np.abs(e).mean():.4f})")
        print(f"  절단평면 8개: 시간 {min(shares['timing_spx']):+.4f}~{max(shares['timing_spx']):+.4f} "
              f"(평균 {np.mean(shares['timing_spx']):+.4f}), 진입점 {min(shares['entry_spx']):+.4f}~"
              f"{max(shares['entry_spx']):+.4f} (평균 {np.mean(shares['entry_spx']):+.4f})")
        print(f"  지각(슬라이스 안에 진입점 미도착): " + ", ".join(
            f"{key} {a}/{b}" for key, (a, b) in late.items()) + f" | 이전 구현 최대 밀림 {lag:.2f} 슬라이스")
    if have and complete:
        todo.append(("A.19", "같은 계획을 항적 규약 셋으로 날린 최저 탐지확률 (인증 계획, 입자 4만, "
                             "같은 난수)", h19, r19))

    have = [c for c in conds if c["mappo3"]]
    if have:
        (h5, r5), stats5 = table_5(have)
        print("\n[표 5] 조건 / 절단평면 / 항로 고정 / 항로 선택 / 고른 간격")
        for label, spx, fixed, chosen, share in stats5:
            print(f"  {label:<14}{spx:.4f} / {fixed:.4f} / {chosen:.4f} ({chosen - fixed:+.4f}, "
                  f"절단평면 대비 {chosen / spx:.2f}) | {share}")
        todo.append(("5", "셀 안 항로를 정책이 고를 때 (중심 경유 규약, 채점은 모형 밖 MC, "
                          "학습 seed 3 개 평균)", h5, r5))

    have, complete = ready("erase", "표 A.20")
    if have:
        (h20, r20), stats20 = table_a20(have)
        report_a20(stats20)
    if have and complete:
        todo.append(("A.20", "두 평가의 차이를 네 항으로 분해한 결과 (입자 4만, 중심 경유 규약)",
                     h20, r20))

    have, complete = ready("law", "표 A.21")
    if have:
        (h21, r21), stats21 = table_a21(have)
        report_a21(stats21)
    if have and complete:
        todo.append(("A.21", "측방 곡선의 두 읽기에 따른 최저 탐지확률 (각 모형에서 세 거동 중 최저, "
                             "중심 경유 규약)", h21, r21))

    if DRY:
        print("\n(--dry: 원고는 고치지 않았다)")
        return
    for number, caption, head, rows in todo:
        html = set_table(html, number, caption, head, rows)
    HTML.write_text(html, encoding="utf-8")
    print("\n원고 갱신:", ", ".join(f"표 {t[0]}" for t in todo))


if __name__ == "__main__":
    main()
