"""3단계 — 정책이 **셀 안 항로까지** 고르게 한다.

지금까지 계획은 "단계 t 에 셀 j 에 있으라"까지만 말했고, 셀 안을 어떻게 날지는
평가기의 규약(간격 = 탐지폭)이 정했다. 부록 A.7.4 가 보였듯 그 선택은 결과를
몇 %p 움직인다. 3단계는 그 자유도를 **정책의 행동**으로 옮긴다.

    2단계 행동   인접 셀 아홉 개 중 하나
    3단계 행동   (인접 셀, 셀 안 항로 간격) 쌍 — 간격 후보 세 개

간격은 셀에 **들어가는 순간** 굳는다 (회로를 도는 중에는 바꿀 수 없다).
실행가능집합은 셀만이 정하므로 절단평면과의 비교는 그대로 성립한다.

채점은 승인된 평가기가 하며, 정책이 고른 간격 열을 그대로 받아 날린다.

실행:  python3 tools/run_mappo_stage3.py [Tank-8 ...]
       MAPPO_SEEDS (기본 1,2,3), MAPPO_ITERATIONS (기본: config 의 mappo.study_iterations), MC_PARTICLES (기본 40000),
       LANE_OPTIONS (기본 200,400,800), ENV_PARTICLES·ENV_PIXEL_M·ENV_SUB
"""
import json, os, pathlib, sys
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.experiments.ch2_stone_spx import (
    case_specs, case_target_specs, case_mission, cue_prior)
from cpp_search.experiments.ch2_studies import published_instance
from cpp_search.solvers.mappo.continuous_env import (
    ContinuousRewardConfig, ContinuousSPXEnvironment)
from cpp_search.solvers.mappo.spx_policy import train_stone_mappo
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, evaluate, selftest
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL

from run_mappo_stage1 import audit_paths                   # noqa: E402
from run_mappo_stage2 import CONDITIONS, _settings         # noqa: E402
from conditions import stage2_results                      # noqa: E402

SEED, DT_S = 20260201, 5.0
POLICY_SEEDS = [int(v) for v in os.environ.get("MAPPO_SEEDS", "1,2,3").split(",")]
MC_PARTICLES = int(os.environ.get("MC_PARTICLES", 40_000))
LANE_OPTIONS = tuple(float(v) for v in
                     os.environ.get("LANE_OPTIONS", "200,400,800").split(","))


def main(argv: list[str]) -> None:
    if not selftest():
        raise SystemExit("selftest 실패 — 소인 규약이 어긋났다")
    wanted = set(argv) or None
    cfg = load_chapter_config("3")
    delay = float(cfg.get("stone_spx.initial_belief_delay_s", 0.0))
    settings, env_config = _settings(cfg)
    reward_template = dict(
        particle_count=int(os.environ.get("ENV_PARTICLES", 4000)),
        pixel_m=float(os.environ.get("ENV_PIXEL_M", 100.0)),
        sub_slices=int(os.environ.get("ENV_SUB", 3)),
        lane_m=SUPPORT_HALF_M, seed=SEED, belief_delay_s=delay,
        lane_options=LANE_OPTIONS,
    )

    for case, n, plan_file in CONDITIONS:
        tag = f"{case}-{n}"
        if wanted and tag not in wanted:
            continue
        block = next(c for c in case_specs(cfg) if c["name"] == case)
        targets = case_target_specs(cfg, block)
        mission = case_mission(cfg.mission(), block)
        instance = published_instance(cfg, case=block, width=n, seed=SEED)
        prior = cue_prior(cfg, mission.search_radius_m)[0]
        stored = json.load(open(ROOT / "results/studies" / plan_file))
        spx_paths = tuple(tuple(int(c) for c in p) for p in stored["assignments"])

        environment = ContinuousSPXEnvironment(
            instance, env_config, targets=targets, mission=mission, prior=prior,
            terrain_coupling=BELIEF_TERRAIN_FULL,
            reward=ContinuousRewardConfig(**reward_template))
        print(f"\n[{case} {n}×{n}]  셀 {instance.cell_count}, 단계 "
              f"{instance.time_slice_count}, 항로 후보 "
              f"{', '.join(f'{v:.0f} m' for v in environment.lane_choices)}, "
              f"행동 {environment.action_count} 개", flush=True)

        # 대조군: 절단평면 계획과 2단계(항로 고정) 계획. 2단계는 **같은 항적 규약**
        # 으로 학습한 것이어야 한다 — 없거나 다른 규약이면 대조가 성립하지 않는다.
        entries = [("절단평면", spx_paths, SUPPORT_HALF_M)]
        fixed_rows = stage2_results("nearest", {tag})
        if not fixed_rows:
            raise SystemExit(f"{tag}: 2단계 결과(mappo2-{tag}.json)가 없다 — 먼저 2단계를 돌린다")
        for row in fixed_rows[0][1]["continuous_trained"]:
            if row["policy_seed"] in POLICY_SEEDS:
                entries.append((f"항로 고정 s{row['policy_seed']}",
                                tuple(tuple(int(c) for c in p) for p in row["assignments"]),
                                SUPPORT_HALF_M))

        rows = []
        for policy_seed in POLICY_SEEDS:
            started = perf_counter()
            learned = train_stone_mappo(instance, settings=settings,
                                        environment=environment, seed=policy_seed)
            paths = tuple(tuple(int(c) for c in p) for p in learned.best_paths)
            lanes = [list(map(float, row))
                     for row in learned.best_plan_metadata.get("lane_m", [])]
            if len(lanes) != len(paths):
                raise SystemExit("학습 결과에 탐색자별 항로 열이 없다")
            audit = audit_paths(instance, paths)
            counts = {v: sum(row.count(v) for row in lanes)
                      for v in environment.lane_choices}
            total = max(sum(counts.values()), 1)
            rows.append({
                "policy_seed": policy_seed,
                "training_reward_pd": float(learned.best_worst_target_detection_probability),
                "cell_model_pd_per_target": [float(v) for v in instance.score_paths(paths)],
                "lane_histogram": {f"{k:.0f}": v for k, v in counts.items()},
                "reported_plan_audit": audit,
                "runtime_s": float(learned.runtime_s),
                "assignments": [list(p) for p in paths],
                "lane_m": lanes,
            })
            entries.append((f"항로 선택 s{policy_seed}", paths, lanes))
            share = ", ".join(f"{k:.0f} m {100*v/total:.0f} %"
                              for k, v in counts.items())
            print(f"  seed {policy_seed}: 학습 보상 {learned.best_worst_target_detection_probability:.4f}"
                  f" | 셀 모형 {min(instance.score_paths(paths)):.4f} | 항로 {share}"
                  f" | 감사 {'통과' if audit['feasible'] else audit} | "
                  f"{perf_counter() - started:.0f} s", flush=True)

        started = perf_counter()
        exact = evaluate(instance, targets, entries, mission=mission, prior=prior,
                         terrain_coupling=BELIEF_TERRAIN_FULL,
                         particle_count=MC_PARTICLES, dt_s=DT_S, seed=SEED,
                         belief_delay_s=delay, progress=lambda *_: None)
        pd = np.array([t["pd"] for t in exact["targets"]])
        names = [t["target"] for t in exact["targets"]]
        print(f"  평가기 {perf_counter() - started:.0f} s (입자 {MC_PARTICLES:,}, "
              f"계획 {len(entries)}개)")
        print(f"  {'계획':<16}{'셀 모형':>9}{'연속':>9}   "
              + "".join(f"{name[:9]:>11}" for name in names))
        cell = np.array(exact["cell_model_pd_per_target"])
        for index, (name, _, _) in enumerate(entries):
            print(f"  {name:<16}{cell[index].min():>9.4f}{pd[:, index].min():>9.4f}   "
                  + "".join(f"{pd[k, index]:>11.4f}" for k in range(len(names))),
                  flush=True)

        dst = ROOT / f"results/studies/mappo3-{tag}.json"
        json.dump({
            "case": case, "n": n, "planning_seed": SEED, "plan_file": plan_file,
            "fingerprint": instance.fingerprint,
            "lane_options": list(LANE_OPTIONS),
            "action_count": environment.action_count,
            "reward_config": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in reward_template.items()},
            "mappo_settings": {k: getattr(settings, k) for k in settings.__slots__},
            "mc_particles": MC_PARTICLES, "targets": names,
            "plans": [name for name, _, _ in entries],
            "continuous_pd_per_target": pd.T.tolist(),
            "cell_model_pd_per_target": cell.tolist(),
            "lane_chosen": rows,
        }, open(dst, "w"), ensure_ascii=False, indent=2)
        print("  저장:", dst.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
