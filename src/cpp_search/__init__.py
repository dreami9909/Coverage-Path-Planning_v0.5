"""협동 표적탐색 연구 시뮬레이터.

최상위에는 모듈을 두지 않는다. 모든 코드는 역할별 하위 패키지에 속한다.

```
cpp_search/
├── models/       도메인 계층 — 센서·표적 운동·지형·belief·시뮬레이션 (세계 기술)
├── common/       공용 지원 — 실험조건 로딩(config)·실행 옵션(options)
├── theory/       경로제약 최적화 이론 — Stone SP1/SPX 절단평면·Markov 모델
├── solvers/      milp/(절단평면 계획) · mappo/(학습 정책)
├── reporting/    공통 KPI 블록·신뢰성 계약·CSV/PNG 산출
├── experiments/  챕터 실험 ch1~ch4 + runner(CLI)
└── pipeline/     시나리오 TOML -> Model -> Solver -> Output
```

의존은 한 방향이다:
``experiments -> solvers.mappo -> solvers.milp -> theory -> models``.
``common``·``reporting``은 사다리 밖 지원 계층으로, 실행 계층을 import 하지
않는다. ``tests/test_architecture.py``가 이 규칙을 코드로 강제한다.

여기서 재수출하는 이름은 도메인 계층(models)의 공개 API다.
"""

from cpp_search.models.models import MissionConfig, Point2D, Route, SensorChannelSpec, SensorSpec
from cpp_search.models.profiles import (
    NOMINAL_TARGET_PROFILES,
    TANK_OPERATIONAL_PROFILE,
    TEL_OPERATIONAL_PROFILE,
    TargetOperationalProfile,
)
from cpp_search.models.search_envelope import CompositeSearchEnvelope
from cpp_search.models.sensor_observation import (
    SENSOR_PERFORMANCE_SR_Z50,
    SpatialDetectionModel,
    TANK_NOMINAL,
    TEL_NOMINAL,
)

__all__ = [
    "CompositeSearchEnvelope",
    "MissionConfig",
    "NOMINAL_TARGET_PROFILES",
    "Point2D",
    "Route",
    "SENSOR_PERFORMANCE_SR_Z50",
    "SensorChannelSpec",
    "SensorSpec",
    "SpatialDetectionModel",
    "TANK_NOMINAL",
    "TANK_OPERATIONAL_PROFILE",
    "TEL_NOMINAL",
    "TEL_OPERATIONAL_PROFILE",
    "TargetOperationalProfile",
]
