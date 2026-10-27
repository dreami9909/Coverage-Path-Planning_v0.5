"""셀 모형과 모형 밖 MC 의 차이를 **셀 안 위치의 소거**로 분해한다.

셀 모형은 표적이 어느 셀에 있는지만 기억하고 셀 안 위치는 매 단계 새로 뽑힌다고
전제한다. 그 전제를 모형 밖 MC 에 단계적으로 심어 넣으면 차이가 어디서 오는지
갈린다. 같은 입자 집합·같은 항적으로 세 가지를 나란히 잰다(공통 난수).

    none   실제 위치            — 본문 표 1·2 의 MC
    slice  슬라이스마다 셀 안에서 위치를 새로 뽑음  → 슬라이스 사이의 위치 지속성을 지운다
    step   입자 스텝마다 새로 뽑음                 → 슬라이스 안의 띠 모양 불균일까지 지운다
    mean   위치 대신, 항적이 그 슬라이스에 슬라이스 시작 셀에 뿌린 hazard 의 셀 평균을 줌
           → 띠 모양이 사라지고 hazard 가 셀 안에 고르게 퍼진 것
    cell   위치 대신 슬라이스 시작 셀에 셀 모형의 hazard Y_t[cell] 을 그대로 줌
           → 표적의 **셀 열**만 실제 궤적에서 오고 나머지는 전부 셀 모형

    none → step  : 셀 안 위치가 실제 위치냐 매번 새로 뽑은 위치냐의 차 (거동에 따라 갈릴 것)
    step → mean  : 소인이 띠로 쌓이는 것과 셀 안에 고르게 퍼진 것의 차 (Jensen, A.7.3)
    mean → cell  : 항적이 실제로 그 셀에 뿌린 양과 셀 모형이 그 셀에 적은 양의 차 (경계 밖 유출)
    cell → 셀 모형: 실제 셀 열을 Markov 사슬로 줄인 몫 (+ 입자 5천 대 4만, 시간전개)

실행:  python3 tools/erasure_probe.py [조건...]      (기본: 저장된 계획이 있는 여덟 조건)
       MC_PARTICLES (기본 40000), ERASE_ENTRY (기본 centre-sync, 본 표 규약), ERASE_OUT (저장 접두어, 기본 erase)
저장:  results/studies/{접두어}-{케이스}-{N}.json
"""
import json, os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpp_search.config import load_chapter_config
from cpp_search.options import RunOptions
from cpp_search.chapters.ch2_stone_spx import (
    build_instance, case_mission, case_specs, case_target_specs, cue_prior)
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL
from cpp_search.solvers.milp.continuous_mc import evaluate, selftest, SUPPORT_HALF_M

SEED, H = 20260201, 600.0
PARTICLES = int(os.environ.get("MC_PARTICLES", 40_000))
OUT_TAG = os.environ.get("ERASE_OUT", "erase")
sys.path.insert(0, str(ROOT / "tools"))
from conditions import CONDITIONS, BASELINE                 # noqa: E402
ENTRY = os.environ.get("ERASE_ENTRY", BASELINE)       # 항적 진입 규약 (core/searcher_track)
ERASE = ("none", "slice", "step", "mean", "cell")


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
        stored = json.load(open(out_dir / plan_file))
        if stored.get("fingerprint") not in (None, inst.fingerprint):
            raise SystemExit(f"{tag}: fingerprint 불일치 — 저장된 계획과 다른 인스턴스다")
        paths = tuple(tuple(int(c) for c in p) for p in stored["assignments"])
        print(f"\n[{case} {n}×{n}]  입자 {PARTICLES:,}", flush=True)
        plans = [(f"소거 {e}", paths, SUPPORT_HALF_M) for e in ERASE]
        res = evaluate(inst, targets, plans, mission=mission,
                       prior=cue_prior(cfg, mission.search_radius_m)[0],
                       terrain_coupling=BELIEF_TERRAIN_FULL,
                       particle_count=PARTICLES, dt_s=5.0, seed=SEED, erase=ERASE,
                       entry=(ENTRY,) * len(ERASE),
                       belief_delay_s=float(cfg.get("stone_spx.initial_belief_delay_s", 0.0)))
        res.update({"case": case, "n": n, "seed": SEED, "plan_file": plan_file})
        dst = out_dir / f"{OUT_TAG}-{tag}.json"
        json.dump(res, open(dst, "w"), ensure_ascii=False, indent=2)
        print("  저장:", dst.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
