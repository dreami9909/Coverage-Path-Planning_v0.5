"""전이행렬의 적분 간격만 바꿔 셀 모형 값이 얼마나 움직이는지 잰다.

셀 모형은 슬라이스당 ``predict(Δ)`` 를 한 번 부르고, 모형 밖 MC 는
``predict(Δ/sub)`` 를 sub 번 부른다. IMM5 는 모드를 갈아타며 움직이므로 이 둘이
같은 분포를 주지 않는다. **같은 계획·같은 hazard 를 두 전이행렬로만 바꿔
평가**하면 그 차이가 순수하게 분리된다.
"""
import pathlib, sys, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.options import RunOptions
from cpp_search.experiments.ch2_stone_spx import (
    build_instance, case_mission, case_specs, case_target_specs, cue_prior)
from cpp_search.solvers.milp.terrain_belief import TerrainParticleBelief, BELIEF_TERRAIN_FULL
from cpp_search.solvers.milp.stone_spx import _target_path_detection_probabilities
from cpp_search.theory.stone_path import StoneTargetModel

CASE, N, SEED, H, SUB = "Tank", 8, 20260201, 600.0, 11
cfg = load_chapter_config("2")
m0, sensor = cfg.mission(), cfg.sensor()
base = next(g for g in cfg.get("stone_spx.grid_conditions", []) if g.get("grid_kind") == "square")
block = next(c for c in case_specs(cfg) if c["name"] == CASE)
targets = case_target_specs(cfg, block)
mission = case_mission(m0, block)
opts = RunOptions(sample_count=10, particle_count=5000, episode_count=1,
                  target_profile="both", seed=1, map_error=False,
                  communication_loss_probability=0.0, communication_latency_slices=0,
                  mission_time_s=H)
inst = build_instance(cfg, opts, mission, sensor,
                      grid={**base, "grid_width": N, "grid_height": N},
                      seed=SEED, terrain_weighting=True, targets=targets)
grid, slice_s, T = inst.grid, float(inst.slice_s), inst.time_slice_count
prior = cue_prior(cfg, mission.search_radius_m)[0]
paths = tuple(tuple(int(c) for c in p)
              for p in json.load(open(ROOT / "results/studies/occupancy-1.json"))["assignments"])

outside = grid.cell_count
def states(f):
    out = []
    for point, esc in zip(f.points, f.escaped):
        c = None if esc else grid.cell_of(point)
        out.append(outside if c is None else c)
    return out

def transitions_for(index, sub):
    """sub 번으로 쪼개 전진시키며 슬라이스 단위 전이행렬을 만든다."""
    b = TerrainParticleBelief(mission, targets[index].profile, prior=prior,
        particle_count=opts.particle_count, seed=SEED ^ (0x5350_5800 + index),
        radial_step_m=max(mission.search_radius_m, 1.0), angular_bin_count=8,
        terrain=inst.terrain, terrain_coupling=BELIEF_TERRAIN_FULL, roughening=False)
    f = b.filter
    initial = np.zeros(outside + 1)
    for s, w in zip(states(f), f.weights):
        initial[s] += w
    initial /= initial.sum()
    steps = max(T - 1, 0)
    P = np.zeros((steps, outside + 1, outside + 1))
    for t in range(steps):
        before = states(f); w = tuple(f.weights)
        for _ in range(sub):
            f.predict(slice_s / sub)
        after = states(f)
        for s, d, wt in zip(before, after, w):
            P[t, s, d] += wt
        row = P[t].sum(axis=1)
        for s in range(outside + 1):
            if row[s] > 0: P[t, s] /= row[s]
            else: P[t, s, s] = 1.0
    return initial, P

print(f"Tank {N}×{N}  슬라이스 {slice_s:.1f} s,  단계 {T},  계획은 인증된 것 그대로\n")
print(f"{'표적':<18}{'셀모형(Δ 한번)':>16}{'셀모형(Δ/11 쪼갬)':>19}{'차':>10}")
rows = []
for sub, label in ((1, "coarse"), (SUB, "fine")):
    models = []
    for k in range(len(targets)):
        init, P = transitions_for(k, sub)
        models.append(StoneTargetModel(init, P, inst.target_models[k].hazard_multiplier))
    pds = _target_path_detection_probabilities(inst.problem, tuple(models), paths)
    rows.append([float(v) for v in pds])
for k in range(len(targets)):
    name = inst.target_names[k].split("/")[-1]
    print(f"{name:<18}{rows[0][k]:>16.4f}{rows[1][k]:>19.4f}{rows[1][k]-rows[0][k]:>10.4f}")
print(f"{'최악(minimax)':<18}{min(rows[0]):>16.4f}{min(rows[1]):>19.4f}{min(rows[1])-min(rows[0]):>10.4f}")
print(f"\n참고: 저장된 인증값 0.7474, 모형 밖 MC 최악 0.5688")
