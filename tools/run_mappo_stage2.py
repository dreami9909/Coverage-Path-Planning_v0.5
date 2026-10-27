"""2단계 — **연속 공간에서 보상을 받은 정책**이 셀 모형 정책보다 실제로 나은가.

무엇을 비교하나
---------------
같은 인스턴스, 같은 행동공간, 같은 학습 규칙에서 **보상을 재는 자리만** 다른 두
정책을 학습시키고, 셋을 같은 자로 채점한다.

    절단평면    인증된 계획 (셀 모형에서 최적, 저장된 결과)
    셀 학습     셀 모형에서 보상을 받은 정책          (1단계, mappo1-*.json)
    연속 학습   연속 공간에서 보상을 받은 정책         (이 스크립트)

채점은 승인된 평가기(``planning/continuous_mc.evaluate``)가 한다. 학습에 쓴 빠른
환경이 아니라 평가기가 최종 수치를 내야 "빠른 환경에 과적합했다"는 반론이 닫힌다.
계획을 한 번에 넘기므로 **같은 입자 집합**으로 짝지어 비교된다.

셀 모형 값도 같이 싣는다. 연속 학습 정책이 셀 모형 기준으로는 더 나쁘면서 실제로
더 좋다면, 그것이 곧 "셀 모형의 전제가 계획을 어디로 끌고 갔는가"의 증거다.

실행:  python3 tools/run_mappo_stage2.py [Tank-8 ...]
       MAPPO_SEEDS (기본 1,2,3), MAPPO_ITERATIONS (기본: config 의 mappo.study_iterations), MC_PARTICLES (기본 40000),
       ENV_PARTICLES (학습 보상용, 기본 4000), ENV_PIXEL_M (기본 100), ENV_SUB (기본 3),
       ENV_ENTRY (항적 규약: nearest 기준 | centre-sync 대조. 결과는 mappo2[-centre]-*.json),
       STAGE2_VARIANT (곁가지 실행의 이름표. 예: MAPPO_ITERATIONS=400 STAGE2_VARIANT=it400)
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
from cpp_search.solvers.mappo.continuous_env import (
    ContinuousRewardConfig, ContinuousSPXEnvironment)
from cpp_search.solvers.mappo.spx_env import SPXEnvironmentConfig
from cpp_search.solvers.mappo.spx_policy import StoneMAPPOSettings, train_stone_mappo
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, evaluate, selftest
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL

SEED, DT_S = 20260201, 5.0
sys.path.insert(0, str(ROOT / "tools"))
from conditions import (                                    # noqa: E402
    CONDITIONS, convention_label, stage2_path)

#: 학습 보상과 최종 채점이 **같은 항적 규약**을 쓴다. 다르면 정책이 최적화한 것과
#: 채점받는 것이 어긋난다 (tools/entry_probe.py 가 그 크기를 잰다).
ENTRY = os.environ.get("ENV_ENTRY", "nearest")
#: 본 실행과 설정이 다른 곁가지 실행의 이름표 (학습 예산 민감도 등). 결과 파일이 따로 저장된다.
VARIANT = os.environ.get("STAGE2_VARIANT", "")
POLICY_SEEDS = [int(v) for v in os.environ.get("MAPPO_SEEDS", "1,2,3").split(",")]
MC_PARTICLES = int(os.environ.get("MC_PARTICLES", 40_000))


def _settings(cfg) -> tuple[StoneMAPPOSettings, SPXEnvironmentConfig]:
    block = dict(cfg.get("mappo", {}) or {})
    env = dict(block.get("environment", {}) or {})
    # 연구 측정의 반복 수는 설정 파일의 ``study_iterations`` 다. ``iterations`` 는 챕터
    # 실행용 빠른 값이라 여기서 쓰면 학습이 모자란다 (2026-10-05 에 실제로 그랬다).
    if "study_iterations" not in block:
        raise SystemExit("config 의 mappo.study_iterations 가 없다")
    iterations = int(os.environ.get("MAPPO_ITERATIONS", block["study_iterations"]))
    return (
        StoneMAPPOSettings(
            iterations=iterations,
            rollouts_per_iteration=int(block.get("rollouts_per_iteration", 8)),
            hidden_size=int(block.get("hidden_size", 64)),
            actor_learning_rate=float(block.get("actor_learning_rate", 5e-4)),
            critic_learning_rate=float(block.get("critic_learning_rate", 1e-3)),
            gamma=float(block.get("gamma", 0.99)),
            gae_lambda=float(block.get("gae_lambda", 0.95)),
            clip_ratio=float(block.get("clip_ratio", 0.2)),
            entropy_coefficient=float(block.get("entropy_coefficient", 0.01)),
            epochs=int(block.get("epochs", 4)),
        ),
        SPXEnvironmentConfig(
            action_count=int(env.get("action_count", 9)),
            worst_target_temperature=float(env.get("worst_target_temperature", 0.05)),
            terminal_weight=float(env.get("terminal_weight", 10.0)),
            slice_weight=float(env.get("slice_weight", 20.0)),
            repair_penalty=float(env.get("repair_penalty", 0.05)),
        ),
    )


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
        entry=ENTRY,
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
        if abs(min(instance.score_paths(spx_paths)) - float(stored["worst_pd"])) > 1e-9:
            raise SystemExit("저장된 계획을 다시 채점한 값이 다르다 — 다른 인스턴스다")

        print(f"\n[{case} {n}×{n}]  항적 규약 {convention_label(ENTRY)}, "
              f"셀 {instance.cell_count}, 단계 "
              f"{instance.time_slice_count}, 학습 보상 입자 "
              f"{reward_template['particle_count']:,}, 픽셀 "
              f"{reward_template['pixel_m']:.0f} m, 조각 "
              f"{reward_template['sub_slices']}", flush=True)

        started = perf_counter()
        environment = ContinuousSPXEnvironment(
            instance, env_config, targets=targets, mission=mission, prior=prior,
            terrain_coupling=BELIEF_TERRAIN_FULL,
            reward=ContinuousRewardConfig(**reward_template))
        print(f"  입자 궤적 준비 {perf_counter() - started:.0f} s", flush=True)

        plans = [("절단평면", spx_paths)]
        cell_file = ROOT / f"results/studies/mappo1-{case}-{n}.json"
        cell_rows = json.load(open(cell_file))["rows"] if cell_file.exists() else []
        for row in cell_rows:
            if row["policy_seed"] in POLICY_SEEDS:
                plans.append((f"셀 학습 s{row['policy_seed']}",
                              tuple(tuple(int(c) for c in p) for p in row["assignments"])))

        learned_rows = []
        for policy_seed in POLICY_SEEDS:
            started = perf_counter()
            learned = train_stone_mappo(instance, settings=settings,
                                        environment=environment, seed=policy_seed)
            paths = tuple(tuple(int(c) for c in p) for p in learned.best_paths)
            learned_rows.append({
                "policy_seed": policy_seed,
                "training_reward_pd": float(learned.best_worst_target_detection_probability),
                "untrained_training_reward_pd": float(
                    learned.untrained_worst_target_detection_probability),
                "cell_model_pd_per_target": [float(v) for v in instance.score_paths(paths)],
                "repaired_moves": int(learned.repaired_moves),
                "occupancy_violations": int(learned.occupancy_violations),
                "separation_violations": int(learned.separation_violations),
                "runtime_s": float(learned.runtime_s),
                "assignments": [list(p) for p in paths],
            })
            plans.append((f"연속 학습 s{policy_seed}", paths))
            print(f"  seed {policy_seed}: 학습 보상 기준 "
                  f"{learned.best_worst_target_detection_probability:.4f} "
                  f"(미학습 {learned.untrained_worst_target_detection_probability:.4f}) | "
                  f"셀 모형 {min(instance.score_paths(paths)):.4f} | "
                  f"{perf_counter() - started:.0f} s", flush=True)

        # 최종 채점은 **승인된 평가기**가 한다. 계획을 한 번에 넘겨 짝지어 잰다.
        started = perf_counter()
        exact = evaluate(
            instance, targets,
            [(name, paths, SUPPORT_HALF_M) for name, paths in plans],
            mission=mission, prior=prior, terrain_coupling=BELIEF_TERRAIN_FULL,
            particle_count=MC_PARTICLES, dt_s=DT_S, seed=SEED,
            belief_delay_s=delay, entry=(ENTRY,) * len(plans),
            progress=lambda *_: None)
        print(f"  평가기 {perf_counter() - started:.0f} s "
              f"(입자 {MC_PARTICLES:,}, 계획 {len(plans)}개)", flush=True)

        names = [t["target"] for t in exact["targets"]]
        pd = np.array([t["pd"] for t in exact["targets"]])          # [표적, 계획]
        cell = np.array(exact["cell_model_pd_per_target"]).T        # [표적, 계획]
        print(f"  {'계획':<16}{'셀 모형 최저':>12}{'연속 최저':>10}"
              + "".join(f"{name[:9]:>11}" for name in names))
        for index, (name, _) in enumerate(plans):
            print(f"  {name:<16}{cell[:, index].min():>12.4f}{pd[:, index].min():>10.4f}"
                  + "".join(f"{pd[k, index]:>11.4f}" for k in range(len(names))),
                  flush=True)

        dst = stage2_path(case, n, ENTRY, VARIANT)
        json.dump({
            "case": case, "n": n, "planning_seed": SEED, "plan_file": plan_file,
            "entry": ENTRY,
            "cells": int(instance.cell_count),
            "time_slices": int(instance.time_slice_count),
            "fingerprint": instance.fingerprint,
            "reward_config": reward_template,
            "mappo_settings": {k: getattr(settings, k) for k in settings.__slots__},
            "mc_particles": MC_PARTICLES,
            "targets": names,
            "plans": [name for name, _ in plans],
            "continuous_pd_per_target": pd.T.tolist(),
            "cell_model_pd_per_target": cell.T.tolist(),
            "continuous_trained": learned_rows,
        }, open(dst, "w"), ensure_ascii=False, indent=2)
        print("  저장:", dst.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
