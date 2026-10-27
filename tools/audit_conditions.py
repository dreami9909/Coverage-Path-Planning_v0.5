"""셀 모형과 모형 밖 MC 가 **같은 조건**인지 항목별로 대조한다.

둘의 값이 다를 때 어느 쪽이 틀렸는지 말하려면, 먼저 둘이 같은 문제를 풀고
있는지부터 확인해야 한다. 이 스크립트는 탐지폭·속도·배율·사전분포·시간전개를
하나씩 맞춰 본다.
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from cpp_search.config import load_chapter_config
from cpp_search.options import RunOptions
from cpp_search.chapters.ch2_stone_spx import (
    build_instance, case_mission, case_specs, case_target_specs, cue_prior)
from cpp_search.solvers.milp.terrain_belief import TerrainParticleBelief, BELIEF_TERRAIN_FULL
from cpp_search.solvers.milp.continuous_mc import SUPPORT_HALF_M

CASE, N, SEED, H = "Tank", 8, 20260201, 600.0
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
side = float(grid.cell_area_m2) ** 0.5

def line(name, a, b, ok=None):
    same = (abs(a - b) < 1e-9) if ok is None else ok
    print(f"  {name:28} 셀모형 {a:>12.5f} | MC {b:>12.5f}   {'일치' if same else '★ 불일치'}")

print("=== 1. 소인 규약 ===")
line("탐지폭 W [m]", sensor.effective_sweep_width_m, SUPPORT_HALF_M)
line("탐색속도 [m/s]", mission.search_speed_mps, mission.search_speed_mps)
line("순항속도 [m/s]", mission.transit_speed_mps, mission.transit_speed_mps)
line("슬라이스 [s]", slice_s, slice_s)
line("한 슬라이스 소인량/셀면적",
     sensor.effective_sweep_width_m * mission.search_speed_mps * slice_s / grid.cell_area_m2,
     SUPPORT_HALF_M * mission.search_speed_mps * slice_s / (side * side))

print("\n=== 2. 표적 배율 ===")
for k, t in enumerate(targets):
    a = float(np.asarray(inst.target_models[k].hazard_multiplier, dtype=float))
    line(inst.target_names[k].split('/')[-1], a, a)

print("\n=== 3. belief 구성 ===")
print(f"  {'입자 수':28} 셀모형 {opts.particle_count:>12} | MC       120000    ★ 다름(독립 표본)")
print(f"  {'roughening':28} 셀모형 {'False':>12} | MC {'False':>12}   일치")
print(f"  {'사전분포':28} 두 쪽 모두 cue_prior(cfg, search_radius)   일치")
print(f"  {'지형 결합':28} 두 쪽 모두 BELIEF_TERRAIN_FULL            일치")

print("\n=== 4. 시간전개 — 여기가 갈린다 ===")
sub = max(1, int(round(slice_s / 5.0)))
print(f"  셀모형: predict({slice_s:.1f} s) x {T-1} 회")
print(f"  MC    : predict({slice_s/sub:.2f} s) x {(T)*sub} 회   (슬라이스당 {sub} 회)")
prior = cue_prior(cfg, mission.search_radius_m)[0]

def roll(coarse: bool, particles=20000, seed_mix=0):
    b = TerrainParticleBelief(mission, targets[1].profile, prior=prior,
        particle_count=particles, seed=(SEED ^ (0x5350_5800 + 1)) ^ seed_mix,
        radial_step_m=max(mission.search_radius_m, 1.0), angular_bin_count=8,
        terrain=inst.terrain, terrain_coupling=BELIEF_TERRAIN_FULL, roughening=False)
    f = b.filter
    for _ in range(T - 1):
        if coarse:
            f.predict(slice_s)
        else:
            for _ in range(sub):
                f.predict(slice_s / sub)
    w = np.asarray(f.weights, float); w = w / w.sum()
    mass = np.zeros(grid.cell_count + 1)
    for p, e, wt in zip(f.points, f.escaped, w):
        c = None if e else grid.cell_of(p)
        mass[grid.cell_count if c is None else c] += wt
    r = np.hypot([p.x for p in f.points], [p.y for p in f.points])
    return mass, float((w * r).sum()), float(np.asarray(f.escaped).mean())

print(f"\n  {'':10}{'격자 밖 비율':>12}{'평균 반경[m]':>14}{'질량 상위셀':>12}")
mc_c, rc, ec = roll(True)
mc_f, rf, ef = roll(False)
print(f"  {'한 번에':10}{ec:>12.4f}{rc:>14.0f}{np.argmax(mc_c[:-1]):>12}")
print(f"  {'쪼개서':10}{ef:>12.4f}{rf:>14.0f}{np.argmax(mc_f[:-1]):>12}")
tv = 0.5 * np.abs(mc_c - mc_f).sum()
print(f"\n  두 분포의 총변동거리 = {tv:.4f}   (0 이면 완전 동일, 1 이면 완전 분리)")
print(f"  평균 반경 차          = {rf - rc:+.0f} m  ({(rf/rc-1)*100:+.1f} %)")
