"""측방 곡선을 hazard 로 읽느냐 통과당 탐지확률로 읽느냐 — 같은 계획·같은 입자로 둘 다 잰다.

셀 모형은 Stone 의 지수 탐지함수다: 셀에 쌓인 소인 a 에 대해 P_D = 1 − e^{−a}
(Koopman 의 랜덤탐색식). 모형 밖 MC 도 지금까지는 같은 법칙으로 측방 곡선 p(u) 를
hazard 로 더했다 — 그래서 두 평가의 차이는 오직 **소인이 어디에 놓이는가**였다.

그런데 1장의 소인폭은 W = ∫P_D(u) du, 중심선 값 PD0 로 정의된 **통과당 탐지확률**
곡선에서 나온다. 그 정의를 글자 그대로 따르면 한 번 지나갈 때 생존은 (1 − p) 이지
e^{−p} 가 아니다. 두 읽기는 완전 소인에서도 다르다 (간격 = W 한 바퀴, 균등 정지 표적:
지수 0.58 대 확률 0.78). 이 스크립트는 그 차이가 실제 계획에서 얼마인지 잰다.

실행:  python3 tools/probes/law_probe.py [조건...]   MC_PARTICLES (기본 40000),
       LAW_ENTRY (기본 centre-sync, 본 표 규약), LAW_OUT (저장 접두어, 기본 law)
저장:  results/studies/{접두어}-{케이스}-{N}.json
"""
import json, os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

from cpp_search.common.config import load_chapter_config
from cpp_search.common.options import RunOptions
from cpp_search.experiments.ch2_stone_spx import (
    build_instance, case_mission, case_specs, case_target_specs, cue_prior)
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL
from cpp_search.solvers.milp.continuous_mc import evaluate, selftest, SUPPORT_HALF_M
from tools.erasure_probe import CONDITIONS, SEED, H
from tools.conditions import BASELINE

PARTICLES = int(os.environ.get("MC_PARTICLES", 40_000))
OUT_TAG = os.environ.get("LAW_OUT", "law")
ENTRY = os.environ.get("LAW_ENTRY", BASELINE)         # 항적 진입 규약 (core/searcher_track)
LAWS = ("exponential", "probability")


def main(argv):
    if not selftest():
        raise SystemExit("selftest 실패 — 소인 규약이 어긋났다")
    wanted = set(argv) or None
    cfg = load_chapter_config("2")
    m0, sensor = cfg.mission(), cfg.sensor()
    base = next(g for g in cfg.get("stone_spx.grid_conditions", []) if g.get("grid_kind") == "square")
    out_dir = ROOT / "results/studies"
    for case, n, plan_file in CONDITIONS:
        tag = f"{case}-{n}"
        if wanted and tag not in wanted:
            continue
        block = next(c for c in case_specs(cfg) if c["name"] == case)
        targets = case_target_specs(cfg, block)
        mission = case_mission(m0, block)
        opts = RunOptions(sample_count=10, particle_count=5000, episode_count=1,
                          target_profile="both", seed=1, map_error=False,
                          communication_loss_probability=0.0,
                          communication_latency_slices=0, mission_time_s=H)
        inst = build_instance(cfg, opts, mission, sensor,
                              grid={**base, "grid_width": n, "grid_height": n},
                              seed=SEED, terrain_weighting=True, targets=targets)
        paths = tuple(tuple(int(c) for c in p)
                      for p in json.load(open(out_dir / plan_file))["assignments"])
        print(f"\n[{case} {n}×{n}]  입자 {PARTICLES:,}", flush=True)
        plans = [(f"법칙 {l[:4]}", paths, SUPPORT_HALF_M) for l in LAWS]
        res = evaluate(inst, targets, plans, mission=mission,
                       prior=cue_prior(cfg, mission.search_radius_m)[0],
                       terrain_coupling=BELIEF_TERRAIN_FULL,
                       particle_count=PARTICLES, dt_s=5.0, seed=SEED, law=LAWS,
                       entry=(ENTRY,) * len(LAWS),
                       belief_delay_s=float(cfg.get("stone_spx.initial_belief_delay_s", 0.0)))
        res.update({"case": case, "n": n, "seed": SEED, "plan_file": plan_file})
        dst = out_dir / f"{OUT_TAG}-{tag}.json"
        json.dump(res, open(dst, "w"), ensure_ascii=False, indent=2)
        print("  저장:", dst.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
