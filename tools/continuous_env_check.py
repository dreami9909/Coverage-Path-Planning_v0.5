"""학습용 연속 환경이 **승인된 평가기와 같은 값**을 내는지 확인한다.

학습 환경(``learning/continuous_env``)은 속도를 위해 두 가지를 근사한다.

    1. hazard 를 입자마다 재지 않고 항적 둘레 래스터에 그린 뒤 읽는다  (픽셀 크기)
    2. 슬라이스를 평가기만큼 잘게 나누지 않는다                        (조각 수)

조각 수를 평가기와 같게 두면(``sub = round(slice_s / dt_s)``) 입자 표본까지
똑같아진다. 그 줄의 차이가 곧 **래스터 오차**이고, 조각 수를 줄인 줄과의 차이가
**시간 해상도 오차**다. 순위가 뒤집히는지도 함께 본다 — 학습이 보는 것은 값이
아니라 순위다.

실행:  python3 tools/continuous_env_check.py [Tank-8 ...]
       CHECK_PARTICLES (기본 4000)
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
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M, evaluate
from cpp_search.solvers.milp.terrain_belief import BELIEF_TERRAIN_FULL

SEED, DT_S = 20260201, 5.0
CONDITIONS = [("Tank", 4, "anchor-Tank-4-20260201.json"),
              ("Tank", 8, "occupancy-1.json"),
              ("Tank", 10, "scale-Tank-10-300x60.json")]
PARTICLES = int(os.environ.get("CHECK_PARTICLES", 4000))


def _plans(case: str, n: int, plan_file: str):
    """대조에 쓸 계획들 — 절단평면 계획과 1단계 학습 계획 둘."""

    stored = json.load(open(ROOT / "results/studies" / plan_file))
    out = [("절단평면", tuple(tuple(int(c) for c in p) for p in stored["assignments"]))]
    learned = ROOT / f"results/studies/mappo1-{case}-{n}.json"
    if learned.exists():
        rows = json.load(open(learned))["rows"]
        for row in (rows[0], rows[-1]):
            out.append((f"MAPPO-{row['policy_seed']}",
                        tuple(tuple(int(c) for c in p) for p in row["assignments"])))
    return out


def _random_plans(instance, count: int, seed: int) -> list[tuple[str, tuple]]:
    """무작위 정책으로 **실행가능한** 계획을 여러 개 만든다.

    순위 검사용이다. 계획 셋으로는 순위가 우연히 맞을 수 있으므로, 학습이 실제로
    훑는 범위에 가까운 계획들을 모아 두 자의 순위를 견준다.
    """

    from cpp_search.solvers.mappo.spx_env import StoneSPXEnvironment

    environment = StoneSPXEnvironment(instance)
    rng = np.random.default_rng(seed)
    plans = []
    for index in range(count):
        environment.reset()
        done = False
        while not done:
            actions = rng.integers(0, environment.action_count, environment.searcher_count)
            *_, done, metrics = environment.step(actions)
        plans.append((f"무작위-{index}", metrics.paths))
    return plans


def main(argv: list[str]) -> None:
    wanted = set(argv) or None
    cfg = load_chapter_config("3")
    delay = float(cfg.get("stone_spx.initial_belief_delay_s", 0.0))
    report = []

    for case, n, plan_file in CONDITIONS:
        tag = f"{case}-{n}"
        if wanted and tag not in wanted:
            continue
        block = next(c for c in case_specs(cfg) if c["name"] == case)
        targets = case_target_specs(cfg, block)
        mission = case_mission(cfg.mission(), block)
        instance = published_instance(cfg, case=block, width=n, seed=SEED)
        plans = _plans(case, n, plan_file)
        prior = cue_prior(cfg, mission.search_radius_m)[0]
        matched = max(1, int(round(float(instance.slice_s) / DT_S)))
        print(f"\n[{case} {n}×{n}]  셀 {instance.cell_count}, 단계 "
              f"{instance.time_slice_count}, 입자 {PARTICLES:,}, 계획 {len(plans)}개, "
              f"평가기와 같은 조각 수 {matched}", flush=True)

        started = perf_counter()
        exact = evaluate(
            instance, targets,
            [(name, paths, SUPPORT_HALF_M) for name, paths in plans],
            mission=mission, prior=prior, terrain_coupling=BELIEF_TERRAIN_FULL,
            particle_count=PARTICLES, dt_s=DT_S, seed=SEED,
            belief_delay_s=delay, progress=lambda *_: None)
        exact_pd = np.array([t["pd"] for t in exact["targets"]])   # [표적, 계획]
        names = [t["target"] for t in exact["targets"]]
        print(f"  평가기 {perf_counter() - started:.0f} s", flush=True)
        print(f"  {'해상도':<26}{'최대 |차|':>10}{'평균 차':>9}{'최악표적 차':>12}"
              f"{'순위':>6}{'준비':>7}{'채점':>7}")

        for pixel_m, sub in ((100.0, matched), (50.0, matched), (200.0, matched),
                             (100.0, 3), (100.0, 1)):
            started = perf_counter()
            reward = ContinuousRewardConfig(
                particle_count=PARTICLES, pixel_m=pixel_m, sub_slices=sub,
                lane_m=SUPPORT_HALF_M, seed=SEED, belief_delay_s=delay)
            env = ContinuousSPXEnvironment(
                instance, targets=targets, mission=mission, prior=prior,
                terrain_coupling=BELIEF_TERRAIN_FULL, reward=reward)
            build_s = perf_counter() - started
            started = perf_counter()
            fast = np.array([env.detection_probabilities(paths)
                             for _, paths in plans]).T              # [표적, 계획]
            score_s = (perf_counter() - started) / max(len(plans), 1)
            diff = fast - exact_pd
            worst_fast, worst_exact = fast.min(axis=0), exact_pd.min(axis=0)
            order_ok = bool((np.argsort(worst_fast) == np.argsort(worst_exact)).all())
            print(f"  픽셀 {pixel_m:>5.0f} m  조각 {sub:<3}        "
                  f"{np.abs(diff).max():>10.4f}{diff.mean():>9.4f}"
                  f"{np.abs(worst_fast - worst_exact).max():>12.4f}"
                  f"{'같음' if order_ok else '다름':>6}{build_s:>6.0f}s{score_s:>6.1f}s",
                  flush=True)
            report.append({
                "case": case, "n": n, "pixel_m": pixel_m, "sub_slices": sub,
                "particles": PARTICLES, "max_abs_diff": float(np.abs(diff).max()),
                "mean_diff": float(diff.mean()),
                "worst_target_max_abs_diff": float(np.abs(worst_fast - worst_exact).max()),
                "ranking_matches": order_ok,
                "prepare_s": build_s, "score_s": score_s,
                "fast_pd": fast.tolist(), "exact_pd": exact_pd.tolist(),
                "targets": names, "plans": [p[0] for p in plans],
            })
            if (pixel_m, sub) == (100.0, matched):
                for k, name in enumerate(names):
                    row = "  ".join(f"{plans[i][0]}: {fast[k, i]:.4f}/{exact_pd[k, i]:.4f}"
                                    for i in range(len(plans)))
                    print(f"      {name:<17} {row}")

        # 순위 검사 — 계획 여러 개에서 두 자가 같은 순서를 매기는가.
        count = int(os.environ.get("CHECK_RANDOM_PLANS", 12))
        if count > 0:
            sample = plans + _random_plans(instance, count, SEED)
            exact_many = evaluate(
                instance, targets,
                [(name, paths, SUPPORT_HALF_M) for name, paths in sample],
                mission=mission, prior=prior, terrain_coupling=BELIEF_TERRAIN_FULL,
                particle_count=PARTICLES, dt_s=DT_S, seed=SEED,
                belief_delay_s=delay, progress=lambda *_: None)
            exact_worst = np.array([t["pd"] for t in exact_many["targets"]]).min(axis=0)
            reward = ContinuousRewardConfig(
                particle_count=PARTICLES, pixel_m=100.0, sub_slices=3,
                lane_m=SUPPORT_HALF_M, seed=SEED, belief_delay_s=delay)
            env = ContinuousSPXEnvironment(
                instance, targets=targets, mission=mission, prior=prior,
                terrain_coupling=BELIEF_TERRAIN_FULL, reward=reward)
            fast_worst = np.array([min(env.detection_probabilities(paths))
                                   for _, paths in sample])
            order_fast = np.argsort(np.argsort(fast_worst))
            order_exact = np.argsort(np.argsort(exact_worst))
            spearman = float(np.corrcoef(order_fast, order_exact)[0, 1])
            top_same = bool(np.argmax(fast_worst) == np.argmax(exact_worst))
            print(f"  순위 검사 (계획 {len(sample)}개, 픽셀 100 m·조각 3): "
                  f"Spearman {spearman:+.3f}, 최선 계획 {'같음' if top_same else '다름'}, "
                  f"최대 |차| {np.abs(fast_worst - exact_worst).max():.4f}", flush=True)
            report.append({
                "case": case, "n": n, "check": "ranking", "plan_count": len(sample),
                "pixel_m": 100.0, "sub_slices": 3, "particles": PARTICLES,
                "spearman": spearman, "top_plan_matches": top_same,
                "fast_worst": fast_worst.tolist(), "exact_worst": exact_worst.tolist(),
                "plans": [p[0] for p in sample],
            })

    dst = ROOT / "results/studies/continuous-env-check.json"
    json.dump(report, open(dst, "w"), ensure_ascii=False, indent=2)
    print("\n저장:", dst.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
