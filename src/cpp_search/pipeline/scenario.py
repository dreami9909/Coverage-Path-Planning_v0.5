"""시나리오 입력 — TOML 하나가 실험 하나다.

검증은 실행 전에 끝낸다. 잘못된 값으로 수십 분을 돌고 나서 죽는 것보다
읽는 즉시 거부하는 것이 싸다. 형식이 TOML 인 이유: 주석이 되고(모든 값에
근거를 적는다), 표준 라이브러리 ``tomllib`` 로 읽혀 의존성이 없다.

``template`` 은 프리셋이다 — 템플릿이 기본값을 채우고, 시나리오 파일에
적은 키가 템플릿보다 우선한다.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cpp_search.config import ChapterConfig, load_chapter_config
from cpp_search.options import RunOptions


#: 탐색기 프리셋. 새 센서는 여기 한 항목으로 등록한다.
SEEKER_TEMPLATES: dict[str, dict[str, Any]] = {
    "SR-Z50": {
        "altitude_m": 600.0,
        "frame_rate_hz": 30.0,
        "detection_probability": 0.86,
        "false_alarm_probability": 0.05,
        "eo": {"resolution": [1280, 720], "hfov_deg": 18.0},
        "ir": {"resolution": [1280, 720], "hfov_deg": 18.0},
        "gimbal": {
            "max_angle_deg": 45.0,
            "slew_rate_dps": 60.0,
            "scan_rate_dps": 30.0,
        },
    },
}

_CASES = ("Tank", "TEL")
_SOLVERS = ("milp", "mappo")
_GRID_KINDS = ("square", "operational-polar")


class ScenarioError(ValueError):
    """시나리오 파일이 선언한 값이 실행 불가능하다."""


def _positive(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ScenarioError(f"{name} 은 양수여야 한다 (지금 {value!r})")
    return value


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ScenarioError(f"{name} 은 (0, 1] 이어야 한다 (지금 {value!r})")
    return value


@dataclass(frozen=True)
class ChannelSpec:
    resolution: tuple[int, int]
    hfov_deg: float
    #: 선언만 받는다 — 현행 공간 탐지모델은 HFOV·고도 기반이라 VFOV 를
    #: 쓰지 않는다. 값을 적으면 스냅샷에 남고 경고 목록에 올라간다.
    vfov_deg: float | None = None


@dataclass(frozen=True)
class GimbalSpec:
    max_angle_deg: float
    slew_rate_dps: float
    scan_rate_dps: float


@dataclass(frozen=True)
class SeekerSpec:
    template: str
    altitude_m: float
    frame_rate_hz: float
    detection_probability: float
    false_alarm_probability: float
    eo: ChannelSpec
    ir: ChannelSpec
    gimbal: GimbalSpec


@dataclass(frozen=True)
class VehicleSpec:
    count: int
    search_speed_kph: float
    transit_speed_kph: float


@dataclass(frozen=True)
class MissionSpec:
    ingress_s: float
    search_s: float


@dataclass(frozen=True)
class TargetSpec:
    case: str                      # Tank | TEL — 케이스(최악 거동 집합)를 정한다
    max_speed_kph: float
    tp_error_sigma_m: float        # 위치(제보) 오차 — 초기 belief 폭
    aoi_radius_m: float | None     # 생략하면 케이스 확정값(순변위 99%)


@dataclass(frozen=True)
class TerrainSpec:
    weighting: bool                # 지형 가중(사전·전이) on/off
    bias_strength: float           # 이동 편향 (지형 가중치)
    offroad_probability: float
    halt_probability_boost: float
    corridor_count: int
    barrier_count: int
    concealment_count: int         # 은닉 지대 수 (은닉 가중치)
    background: float


@dataclass(frozen=True)
class GridSpec:
    kind: str                      # square | operational-polar
    n: int                         # square: N×N 외접 (셀 크기·슬라이스 자동 유도)
    radial_step_m: float           # polar 전용
    angular_bins: int              # polar 전용


@dataclass(frozen=True)
class SolverSpec:
    kind: str                      # milp (mappo 는 v0.5 2단계)
    certificate_gap: float
    time_limit_s: float | None


@dataclass(frozen=True)
class EvaluationSpec:
    mc_samples: int
    particles: int


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    seeker: SeekerSpec
    vehicle: VehicleSpec
    mission: MissionSpec
    target: TargetSpec
    terrain: TerrainSpec
    grid: GridSpec
    solver: SolverSpec
    evaluation: EvaluationSpec
    source_path: Path | None = None
    warnings: tuple[str, ...] = field(default=())

    # ------------------------------------------------- 기존 실행 경로로 번역
    def chapter_config(self, *, config_root: Path | None = None) -> ChapterConfig:
        """Chapter 2 기준 조건 위에 시나리오를 덮어쓴 설정을 만든다.

        기존 config JSON 이 기본값 창고 역할을 한다 — 시나리오 파일은
        바꾸고 싶은 값만 적으면 된다.
        """

        base = load_chapter_config("2", config_root=config_root)
        common = _deep_merge(
            base.common,
            {
                "sensor": {
                    "template": self.seeker.template,
                    "altitude_m": self.seeker.altitude_m,
                    "eo_resolution": list(self.seeker.eo.resolution),
                    "ir_resolution": list(self.seeker.ir.resolution),
                    "frame_rate_hz": self.seeker.frame_rate_hz,
                    "fused_search_hfov_deg": self.seeker.eo.hfov_deg,
                    "eo_search_hfov_deg": self.seeker.eo.hfov_deg,
                    "ir_search_hfov_deg": self.seeker.ir.hfov_deg,
                    "gimbal_max_angle_deg": self.seeker.gimbal.max_angle_deg,
                    "gimbal_max_rate_dps": self.seeker.gimbal.slew_rate_dps,
                    "gimbal_scan_rate_dps": self.seeker.gimbal.scan_rate_dps,
                },
                "detection": {
                    "reference_detection_probability": self.seeker.detection_probability,
                    "false_alarm_probability": self.seeker.false_alarm_probability,
                },
                "mission": {
                    "uav_count": self.vehicle.count,
                    "search_speed_kph": self.vehicle.search_speed_kph,
                    "transit_speed_kph": self.vehicle.transit_speed_kph,
                    "target_max_speed_kph": self.target.max_speed_kph,
                    "ingress_s": self.mission.ingress_s,
                    "mission_time_s": self.mission.search_s,
                },
                "aoi": {"cue_ring": {"sigma_m": self.target.tp_error_sigma_m}},
                "target_motion": {
                    "terrain_coupling": {
                        "bias_strength": self.terrain.bias_strength,
                        "offroad_probability": self.terrain.offroad_probability,
                        "halt_probability_boost": self.terrain.halt_probability_boost,
                    }
                },
                "monte_carlo": {
                    "sample_count": self.evaluation.mc_samples,
                    "base_seed": self.seed,
                },
            },
        )
        data = _deep_merge(
            base.data,
            {
                "stone_spx": {
                    "initial_belief_delay_s": self.mission.ingress_s,
                    "certificate": {
                        "required_relative_gap": self.solver.certificate_gap
                    },
                }
            },
        )
        return ChapterConfig("pipeline", data, common, base.source_paths)

    def run_options(self) -> RunOptions:
        return RunOptions(
            sample_count=self.evaluation.mc_samples,
            particle_count=self.evaluation.particles,
            episode_count=1,
            target_profile=self.target.case.lower(),
            seed=self.seed,
            map_error=False,
            communication_loss_probability=0.0,
            communication_latency_slices=0,
            mission_time_s=self.mission.search_s,
        )

    def grid_declaration(self, base_grid: dict[str, Any]) -> dict[str, Any]:
        """config 의 격자 선언(기본 한도 포함)에 시나리오 격자를 덮어쓴다."""

        declared: dict[str, Any] = {
            **base_grid,
            "name": f"{self.grid.kind}-{self.grid.n}",
            "grid_kind": self.grid.kind,
            "square_fit": "circumscribed",
            "grid_width": self.grid.n,
            "grid_height": self.grid.n,
            "time_slice_count": "auto",
            "particle_count": self.evaluation.particles,
            "radial_step_m": self.grid.radial_step_m,
            "angular_bin_count": self.grid.angular_bins,
        }
        if self.solver.time_limit_s is not None:
            declared["master_time_limit_s"] = self.solver.time_limit_s
        return declared

    def snapshot(self) -> dict[str, Any]:
        """결과 디렉터리에 남길 입력 전문 (재현의 근거)."""

        from dataclasses import asdict

        body = asdict(self)
        body["source_path"] = str(self.source_path) if self.source_path else None
        body["warnings"] = list(self.warnings)
        return body


# ------------------------------------------------------------------ 로딩/검증

def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return build_scenario(raw, source_path=path)


def build_scenario(raw: dict[str, Any], *, source_path: Path | None = None) -> Scenario:
    warnings: list[str] = []

    name = str(raw.get("name", "")).strip()
    if not name or any(ch in name for ch in "/\\ "):
        raise ScenarioError(
            "name 은 빈칸·슬래시 없는 실행 이름이어야 한다 (결과 폴더명이 된다)"
        )
    seed = int(raw.get("seed", 20260830))

    seeker = _seeker(dict(raw.get("seeker", {}) or {}), warnings)

    vehicle_raw = dict(raw.get("vehicle", {}) or {})
    vehicle = VehicleSpec(
        count=int(vehicle_raw.get("count", 6)),
        search_speed_kph=_positive(
            vehicle_raw.get("search_speed_kph", 160.0), "vehicle.search_speed_kph"
        ),
        transit_speed_kph=_positive(
            vehicle_raw.get("transit_speed_kph", 160.0), "vehicle.transit_speed_kph"
        ),
    )
    if not 1 <= vehicle.count <= 12:
        raise ScenarioError(f"vehicle.count 는 1~12 (지금 {vehicle.count})")

    mission_raw = dict(raw.get("mission", {}) or {})
    mission = MissionSpec(
        ingress_s=_positive(mission_raw.get("ingress_s", 480.0), "mission.ingress_s"),
        search_s=_positive(mission_raw.get("search_s", 600.0), "mission.search_s"),
    )

    target_raw = dict(raw.get("target", {}) or {})
    case = str(target_raw.get("case", "TEL"))
    if case not in _CASES:
        raise ScenarioError(f"target.case 는 {_CASES} 중 하나 (지금 {case!r})")
    aoi_radius = target_raw.get("aoi_radius_m")
    target = TargetSpec(
        case=case,
        max_speed_kph=_positive(
            target_raw.get("max_speed_kph", 40.0), "target.max_speed_kph"
        ),
        tp_error_sigma_m=_positive(
            target_raw.get("tp_error_sigma_m", 500.0), "target.tp_error_sigma_m"
        ),
        aoi_radius_m=float(aoi_radius) if aoi_radius is not None else None,
    )

    terrain_raw = dict(raw.get("terrain", {}) or {})
    terrain = TerrainSpec(
        weighting=bool(terrain_raw.get("weighting", True)),
        bias_strength=_probability(
            terrain_raw.get("bias_strength", 0.6), "terrain.bias_strength"
        ),
        offroad_probability=_probability(
            terrain_raw.get("offroad_probability", 0.2), "terrain.offroad_probability"
        ),
        halt_probability_boost=_probability(
            terrain_raw.get("halt_probability_boost", 0.35),
            "terrain.halt_probability_boost",
        ),
        corridor_count=int(terrain_raw.get("corridor_count", 2)),
        barrier_count=int(terrain_raw.get("barrier_count", 1)),
        concealment_count=int(terrain_raw.get("concealment_count", 3)),
        background=float(terrain_raw.get("background", 0.06)),
    )

    grid_raw = dict(raw.get("grid", {}) or {})
    kind = str(grid_raw.get("kind", "square"))
    if kind not in _GRID_KINDS:
        raise ScenarioError(f"grid.kind 는 {_GRID_KINDS} 중 하나 (지금 {kind!r})")
    grid = GridSpec(
        kind=kind,
        n=int(grid_raw.get("n", 10)),
        radial_step_m=_positive(
            grid_raw.get("radial_step_m", 600.0), "grid.radial_step_m"
        ),
        angular_bins=int(grid_raw.get("angular_bins", 24)),
    )
    if kind == "square" and not 2 <= grid.n <= 40:
        raise ScenarioError(f"grid.n 은 2~40 (지금 {grid.n})")

    solver_raw = dict(raw.get("solver", {}) or {})
    solver_kind = str(solver_raw.get("kind", "milp"))
    if solver_kind not in _SOLVERS:
        raise ScenarioError(f"solver.kind 는 {_SOLVERS} 중 하나 (지금 {solver_kind!r})")
    if solver_kind == "mappo":
        raise ScenarioError(
            "solver.kind = 'mappo' 는 v0.5 2단계(MAPPO implementation)에서 열린다"
        )
    time_limit = solver_raw.get("time_limit_s")
    solver = SolverSpec(
        kind=solver_kind,
        certificate_gap=_probability(
            solver_raw.get("certificate_gap", 0.01), "solver.certificate_gap"
        ),
        time_limit_s=float(time_limit) if time_limit is not None else None,
    )

    evaluation_raw = dict(raw.get("evaluation", {}) or {})
    evaluation = EvaluationSpec(
        mc_samples=int(evaluation_raw.get("mc_samples", 1000)),
        particles=int(evaluation_raw.get("particles", 5000)),
    )
    if evaluation.mc_samples < 1:
        raise ScenarioError("evaluation.mc_samples 는 1 이상")
    if not 100 <= evaluation.particles <= 40_000:
        raise ScenarioError(
            "evaluation.particles 는 100~40000 (궤적 캐시 한도가 4만 입자다)"
        )

    return Scenario(
        name=name,
        seed=seed,
        seeker=seeker,
        vehicle=vehicle,
        mission=mission,
        target=target,
        terrain=terrain,
        grid=grid,
        solver=solver,
        evaluation=evaluation,
        source_path=source_path,
        warnings=tuple(warnings),
    )


def _seeker(raw: dict[str, Any], warnings: list[str]) -> SeekerSpec:
    template_name = str(raw.get("template", "SR-Z50"))
    if template_name not in SEEKER_TEMPLATES:
        raise ScenarioError(
            f"seeker.template {template_name!r} 미등록 — "
            f"등록된 템플릿: {sorted(SEEKER_TEMPLATES)}"
        )
    merged = _deep_merge(SEEKER_TEMPLATES[template_name], raw)

    def channel(key: str) -> ChannelSpec:
        block = dict(merged.get(key, {}) or {})
        resolution = block.get("resolution", [1280, 720])
        if len(resolution) != 2 or any(int(v) <= 0 for v in resolution):
            raise ScenarioError(f"seeker.{key}.resolution 은 [가로, 세로] 양수 쌍")
        vfov = block.get("vfov_deg")
        if vfov is not None:
            warnings.append(
                f"seeker.{key}.vfov_deg 는 선언만 기록된다 — 현행 공간 탐지모델은 "
                "HFOV·고도 기반이라 VFOV 를 쓰지 않는다 (모형 변경 결정 대기)"
            )
        hfov = float(block.get("hfov_deg", 18.0))
        if not 1.0 <= hfov <= 120.0:
            raise ScenarioError(f"seeker.{key}.hfov_deg 는 1~120도 (지금 {hfov})")
        return ChannelSpec(
            resolution=(int(resolution[0]), int(resolution[1])),
            hfov_deg=hfov,
            vfov_deg=float(vfov) if vfov is not None else None,
        )

    gimbal_raw = dict(merged.get("gimbal", {}) or {})
    gimbal = GimbalSpec(
        max_angle_deg=_positive(
            gimbal_raw.get("max_angle_deg", 45.0), "seeker.gimbal.max_angle_deg"
        ),
        slew_rate_dps=_positive(
            gimbal_raw.get("slew_rate_dps", 60.0), "seeker.gimbal.slew_rate_dps"
        ),
        scan_rate_dps=_positive(
            gimbal_raw.get("scan_rate_dps", 30.0), "seeker.gimbal.scan_rate_dps"
        ),
    )
    return SeekerSpec(
        template=template_name,
        altitude_m=_positive(merged.get("altitude_m", 600.0), "seeker.altitude_m"),
        frame_rate_hz=_positive(
            merged.get("frame_rate_hz", 30.0), "seeker.frame_rate_hz"
        ),
        detection_probability=_probability(
            merged.get("detection_probability", 0.86), "seeker.detection_probability"
        ),
        false_alarm_probability=_probability(
            merged.get("false_alarm_probability", 0.05),
            "seeker.false_alarm_probability",
        ),
        eo=channel("eo"),
        ir=channel("ir"),
        gimbal=gimbal,
    )


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
