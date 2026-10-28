"""1단계 — 학습 정책(MAPPO)을 **인증된 최적해와 같은 자**로 잰다.

무엇을 재나
-----------
저장된 계획을 낸 것과 같은 ``StoneGridInstance`` 를 다시 만들고, 그 위에
MAPPO 를 학습시킨 뒤 두 계획을 ``instance.score_paths`` 하나로 채점한다.

    도달률      = MAPPO 최악표적 PD / 절단평면 incumbent PD
    보장 하한   = MAPPO 최악표적 PD / (1 - L),   L = 쌍대 하한(미탐지)

두 번째 값이 이 측정의 요점이다. ``1 - L`` 은 그 격자에서 **어떤 계획도 넘을 수
없는** PD 의 상한이므로, 이 비는 "진짜 최적 대비 최소한 이만큼"이라는 뜻이다.
인증이 닫히지 않은 격자(Tank 12x12, 14x14)에서도 성립한다 — incumbent 와의
비는 거기서 뜻이 흐려지지만 이 비는 흐려지지 않는다.

같은 문제인가
-------------
``fingerprint`` 를 저장된 값과 대조하고, 저장된 계획을 다시 채점해 저장된
``worst_pd`` 와 같은지도 확인한다. 둘 중 하나라도 어긋나면 즉시 멈춘다 —
다른 문제를 비교하느니 아무것도 보고하지 않는 편이 낫다.

실행:  python3 tools/runs/run_mappo_stage1.py [Tank-4 ...]   (기본: 아래 일곱 조건)
       MAPPO_SEEDS      학습 seed 목록 (기본 1,2,3)
       MAPPO_ITERATIONS 학습 반복 (기본: config 의 mappo.study_iterations)
"""
import json, os, pathlib, sys
from time import perf_counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cpp_search.common.config import load_chapter_config
from cpp_search.experiments.ch2_studies import published_instance
from cpp_search.solvers.mappo.spx_env import SPXEnvironmentConfig
from cpp_search.solvers.mappo.spx_policy import StoneMAPPOSettings, train_stone_mappo

SEED = 20260201
sys.path.insert(0, str(ROOT / "tools"))
from conditions import CONDITIONS                           # noqa: E402

POLICY_SEEDS = [int(v) for v in os.environ.get("MAPPO_SEEDS", "1,2,3").split(",")]


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


def _check_same_problem(instance, stored: dict) -> str:
    """저장된 계획이 **이 인스턴스의 것**임을 보인다. 증거를 문자열로 남긴다."""

    paths = tuple(tuple(int(c) for c in p) for p in stored["assignments"])
    scored = instance.score_paths(paths)
    if abs(min(scored) - float(stored["worst_pd"])) > 1e-9:
        raise SystemExit(
            f"저장된 계획을 다시 채점한 값이 다르다: {min(scored)!r} vs {stored['worst_pd']!r}"
        )
    if "fingerprint" in stored:
        if instance.fingerprint != stored["fingerprint"]:
            raise SystemExit("fingerprint 불일치 — 다른 인스턴스다")
        return "fingerprint + rescore"
    # occupancy-1 은 fingerprint 이전 형식이다. 표적별 값이 전부 맞으면
    # 같은 인스턴스라는 증거로 충분하다 (세 값이 우연히 맞을 수 없다).
    # 인스턴스의 이름은 "Tank-nominal/STOP_HEAVY" 꼴이고 저장된 쪽은 거동
    # 이름만 갖는다 (fingerprint 이전 형식).
    names = [str(item).split("/")[-1] for item in instance.target_names]
    for name, value in (stored.get("per_target") or {}).items():
        if abs(scored[names.index(name)] - float(value)) > 1e-9:
            raise SystemExit(f"표적 {name} 재채점 불일치")
    return "per-target rescore"


def audit_paths(instance, paths) -> dict:
    """**보고하는 계획 하나**가 SPX 실행가능집합 안에 있는지 직접 센다.

    학습 중 누적 위반수(``occupancy_violations``)는 탐색 중 깨진 횟수라
    최종 계획의 실행가능성을 말해주지 않는다. 비교의 전제는 "같은
    실행가능집합"이므로 보고하는 계획을 따로 감사한다.
    """

    import numpy as np

    adjacency = np.asarray(instance.problem.searchers[0].adjacency, dtype=bool)
    limit = max(1, int(instance.config.occupancy_limit))
    reservations = instance.reservation_neighborhoods
    starts = [int(c) for c in instance.starts]
    counts = {"adjacency": 0, "occupancy": 0, "separation": 0}
    for searcher, path in enumerate(paths):
        previous = starts[searcher]
        for cell in path:
            if not adjacency[previous, int(cell)]:
                counts["adjacency"] += 1
            previous = int(cell)
    for step in range(len(paths[0])):
        cells = [int(path[step]) for path in paths]
        for cell in set(cells):
            if cells.count(cell) > limit:
                counts["occupancy"] += 1
        if reservations is not None:
            for i, left in enumerate(cells):
                for right in cells[i + 1:]:
                    if right != left and right in reservations[left]:
                        counts["separation"] += 1
    counts["feasible"] = (
        counts["adjacency"] == 0
        and counts["occupancy"] == 0
        and counts["separation"] == 0
    )
    return counts


def _pd_upper_bound(stored: dict) -> tuple[float, str]:
    """그 격자에서 어떤 계획도 넘을 수 없는 PD. 쌍대 하한 L 에서 나온다."""

    if stored.get("lower_bound_nondetection") is not None:
        return 1.0 - float(stored["lower_bound_nondetection"]), "저장된 쌍대 하한"
    # gap = (U - L) / |U| 이므로 L = U (1 - gap) 로 되돌린다.
    nondetection = 1.0 - float(stored["worst_pd"])
    return 1.0 - nondetection * (1.0 - float(stored["gap"])), "gap 에서 역산"


def main(argv: list[str]) -> None:
    wanted = set(argv) or None
    cfg = load_chapter_config("3")
    settings, environment = _settings(cfg)
    out_dir = ROOT / "results/studies"

    for case, n, plan_file in CONDITIONS:
        tag = f"{case}-{n}"
        if wanted and tag not in wanted:
            continue
        stored = json.load(open(out_dir / plan_file))
        overrides = {
            k: v for k, v in (stored.get("overrides") or {}).items()
            if k in {"master_time_limit_s", "max_iterations"}
        }
        instance = published_instance(cfg, case=case, width=n, seed=SEED,
                                      grid_overrides=overrides or None)
        evidence = _check_same_problem(instance, stored)
        spx_paths = tuple(tuple(int(c) for c in p) for p in stored["assignments"])
        spx_audit = audit_paths(instance, spx_paths)
        if not spx_audit["feasible"]:
            raise SystemExit(f"저장된 절단평면 계획이 감사에 걸린다: {spx_audit}")
        bound, bound_source = _pd_upper_bound(stored)
        incumbent = float(stored["worst_pd"])
        print(f"\n[{case} {n}×{n}]  셀 {instance.cell_count}, 단계 "
              f"{instance.time_slice_count}, 같은 문제 확인: {evidence}", flush=True)
        print(f"  절단평면 incumbent {incumbent:.4f} | 달성 가능 상한 {bound:.4f} "
              f"({bound_source}) | 인증 {'닫힘' if stored.get('certified') else '열림'}",
              flush=True)

        rows = []
        for policy_seed in POLICY_SEEDS:
            started = perf_counter()
            learned = train_stone_mappo(instance, settings=settings,
                                        environment_config=environment,
                                        seed=policy_seed)
            worst = float(learned.best_worst_target_detection_probability)
            audit = audit_paths(instance, learned.best_paths)
            rows.append({
                # 보고하는 계획 자체의 감사. 학습 중 누적 위반과 다르다.
                "reported_plan_audit": audit,
                "policy_seed": policy_seed,
                "worst_target_detection_probability": worst,
                "untrained": float(learned.untrained_worst_target_detection_probability),
                "learning_effect": float(learned.learning_effect),
                "achievement_ratio": worst / incumbent,
                "certified_ratio_lower_bound": worst / bound,
                "repaired_moves": int(learned.repaired_moves),
                "occupancy_violations": int(learned.occupancy_violations),
                "separation_violations": int(learned.separation_violations),
                "action_candidate_coverage": float(learned.action_candidate_coverage),
                "runtime_s": float(learned.runtime_s),
                "assignments": [list(p) for p in learned.best_paths],
                "learning_curve": [
                    {k: v for k, v in point.items()
                     if k in {"iteration", "worst_target_detection_probability",
                              "best_worst_target_detection_probability"}}
                    for point in learned.learning_curve
                ],
            })
            print(f"  seed {policy_seed}: MAPPO {worst:.4f} | 도달률 "
                  f"{worst / incumbent:.3f} | 보장 하한 {worst / bound:.3f} | "
                  f"미학습 대비 {learned.learning_effect:+.4f} | "
                  f"계획 감사 {'통과' if audit['feasible'] else audit} | "
                  f"학습 중 점유위반 {learned.occupancy_violations} | "
                  f"{perf_counter() - started:.0f} s", flush=True)

        dst = out_dir / f"mappo1-{tag}.json"
        json.dump({
            "case": case, "n": n, "planning_seed": SEED,
            "plan_file": plan_file, "same_problem_evidence": evidence,
            "cells": int(instance.cell_count),
            "time_slices": int(instance.time_slice_count),
            "fingerprint": instance.fingerprint,
            "spx_worst_pd": incumbent,
            "spx_plan_audit": spx_audit,
            "spx_certified": bool(stored.get("certified", False)),
            "spx_relative_gap": float(stored.get("gap", float("nan"))),
            "achievable_pd_upper_bound": bound,
            "bound_source": bound_source,
            "mappo_settings": settings.__dict__ if hasattr(settings, "__dict__")
                              else {k: getattr(settings, k) for k in settings.__slots__},
            "rows": rows,
        }, open(dst, "w"), ensure_ascii=False, indent=2)
        print("  저장:", dst.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
