"""2단계 계획들을 **다른 난수로 다시** 채점하고, 실행가능성도 감사한다.

왜 필요한가
-----------
연속 보상은 입자 4천 개를 학습 내내 고정해 쓴다 (공통 난수). 그러면 정책이 그
표본에 맞춰 버릴 수 있다. 본 채점은 입자 4만 개로 하지만 seed 가 같으므로,
"다른 난수에서도 유지되는가"는 따로 보여야 한다.

이 스크립트는 저장된 계획(``mappo2-*.json`` 의 세 종류)을 **새 seed** 로 다시
채점한다. 아울러 각 계획이 인접·점유·분리 규칙 안에 있는지 감사한다 — 학습 중
누적 위반수는 최종 계획의 실행가능성을 말해주지 않는다.

실행:  python3 tools/mappo2_rescore.py [Tank-8 ...]
       RESCORE_SEED (기본 20260202), MC_PARTICLES (기본 40000),
       RESCORE_ENTRY (항적 규약: nearest 기준 | centre-sync 대조)
"""
import json, os, pathlib, sys
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.chapters.ch2_stone_spx import (
    case_specs, case_target_specs, case_mission, cue_prior)
from cpp_search.chapters.ch2_studies import published_instance
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, evaluate
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL

sys.path.insert(0, str(ROOT / "tools"))
from run_mappo_stage1 import audit_paths           # noqa: E402
from conditions import (                           # noqa: E402
    convention_label, convention_suffix, stage2_results)

PLANNING_SEED, DT_S = 20260201, 5.0


def load_plans(saved: dict) -> list[tuple[str, tuple]]:
    """``mappo2-*.json`` 이 채점한 계획들을 이름 그대로 다시 모은다."""

    case, n = saved["case"], saved["n"]
    stored_plan = json.load(open(ROOT / "results/studies" / saved["plan_file"]))
    plans = [("절단평면", tuple(tuple(int(c) for c in p)
                             for p in stored_plan["assignments"]))]
    cell_file = ROOT / f"results/studies/mappo1-{case}-{n}.json"
    if cell_file.exists():
        for row in json.load(open(cell_file))["rows"]:
            name = f"셀 학습 s{row['policy_seed']}"
            if name in saved["plans"]:
                plans.append((name, tuple(tuple(int(c) for c in p)
                                          for p in row["assignments"])))
    for row in saved["continuous_trained"]:
        plans.append((f"연속 학습 s{row['policy_seed']}",
                      tuple(tuple(int(c) for c in p) for p in row["assignments"])))
    return plans


def combined_plans(tag: str) -> tuple[dict, list[tuple[str, str, tuple]]]:
    """한 조건의 계획 열 개 — (기준 규약 결과 파일, [(이름, 묶음, 경로)]).

    절단평면 하나, 셀 학습 셋, 그리고 **두 규약 각각의 연속 보상으로 학습한** 정책 셋씩.
    교차 채점(``entry_probe``)과 항적 감사(``track_audit``)가 같은 묶음을 쓴다.
    """
    trained = {entry: dict(stage2_results(entry, {tag}))
               for entry in ("nearest", "centre-sync")}
    if tag not in trained["nearest"] or tag not in trained["centre-sync"]:
        raise SystemExit(f"{tag}: 두 규약의 2단계 결과가 모두 있어야 한다")
    saved = trained["nearest"][tag]
    if saved["fingerprint"] != trained["centre-sync"][tag]["fingerprint"]:
        raise SystemExit(f"{tag}: 두 규약의 2단계가 서로 다른 인스턴스에서 나왔다")
    out = []
    for name, paths in load_plans(saved):
        group = ("절단평면" if name == "절단평면" else
                 "셀 학습" if name.startswith("셀 학습") else "연속 학습·최근접")
        out.append((name.replace("연속 학습", "연속 학습·최근접"), group, paths))
    for row in trained["centre-sync"][tag]["continuous_trained"]:
        out.append((f"연속 학습·중심 s{row['policy_seed']}", "연속 학습·중심",
                    tuple(tuple(int(c) for c in p) for p in row["assignments"])))
    return saved, out


RESCORE_SEED = int(os.environ.get("RESCORE_SEED", 20260202))
#: 어느 규약의 2단계 결과를 다시 채점할지. 채점도 그 규약으로 한다.
ENTRY = os.environ.get("RESCORE_ENTRY", "nearest")
MC_PARTICLES = int(os.environ.get("MC_PARTICLES", 40_000))


def main(argv: list[str]) -> None:
    wanted = set(argv) or None
    cfg = load_chapter_config("3")
    delay = float(cfg.get("stone_spx.initial_belief_delay_s", 0.0))
    report = []

    print(f"항적 규약: {convention_label(ENTRY)}", flush=True)
    for tag, saved in stage2_results(ENTRY, wanted):
        case, n = saved["case"], saved["n"]
        block = next(c for c in case_specs(cfg) if c["name"] == case)
        targets = case_target_specs(cfg, block)
        mission = case_mission(cfg.mission(), block)
        instance = published_instance(cfg, case=block, width=n, seed=PLANNING_SEED)
        if instance.fingerprint != saved["fingerprint"]:
            raise SystemExit(f"{tag}: fingerprint 불일치 — 다른 인스턴스다")
        prior = cue_prior(cfg, mission.search_radius_m)[0]

        plans = load_plans(saved)

        print(f"\n[{case} {n}×{n}]  계획 {len(plans)}개, 채점 seed {RESCORE_SEED}, "
              f"입자 {MC_PARTICLES:,}", flush=True)
        audits = {name: audit_paths(instance, paths) for name, paths in plans}
        bad = [name for name, audit in audits.items() if not audit["feasible"]]
        print(f"  실행가능성 감사: {'전부 통과' if not bad else '위반 ' + ', '.join(bad)}",
              flush=True)

        started = perf_counter()
        exact = evaluate(
            instance, targets,
            [(name, paths, SUPPORT_HALF_M) for name, paths in plans],
            mission=mission, prior=prior, terrain_coupling=BELIEF_TERRAIN_FULL,
            particle_count=MC_PARTICLES, dt_s=DT_S, seed=RESCORE_SEED,
            belief_delay_s=delay, entry=(ENTRY,) * len(plans),
            progress=lambda *_: None)
        pd = np.array([t["pd"] for t in exact["targets"]])          # [표적, 계획]
        old = np.array(saved["continuous_pd_per_target"]).T         # [표적, 계획]
        names = [t["target"] for t in exact["targets"]]
        print(f"  평가기 {perf_counter() - started:.0f} s")
        print(f"  {'계획':<16}{'최저(새 seed)':>14}{'최저(본 채점)':>14}{'차':>8}   "
              + "".join(f"{name[:9]:>11}" for name in names))
        for index, (name, _) in enumerate(plans):
            fresh, before = pd[:, index].min(), old[:, index].min()
            print(f"  {name:<16}{fresh:>14.4f}{before:>14.4f}{fresh - before:>+8.4f}   "
                  + "".join(f"{pd[k, index]:>11.4f}" for k in range(len(names))),
                  flush=True)
        report.append({
            "case": case, "n": n, "rescore_seed": RESCORE_SEED, "entry": ENTRY,
            "particles": MC_PARTICLES, "targets": names,
            "plans": [name for name, _ in plans],
            "continuous_pd_per_target": pd.T.tolist(),
            "audits": audits,
        })

    for block in report:                   # 조건마다 파일 하나 — 조건별로 따로 돌릴 수 있다
        dst = ROOT / (f"results/studies/mappo2-rescore{convention_suffix(ENTRY)}"
                      f"-{block['case']}-{block['n']}.json")
        json.dump(block, open(dst, "w"), ensure_ascii=False, indent=2)
        print("저장:", dst.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
