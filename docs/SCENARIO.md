# 시나리오 작성 매뉴얼 (v0.5 pipeline)

**한 파일 = 한 실험.** 시나리오 TOML 하나가 실험의 입력 전부를 선언하고,
실행하면 `results/runs/<이름>/` 에 입력 스냅샷·지표·그림이 남는다.
그 파일을 그대로 다시 돌리면 재현된다.

## 실행

```bash
# 저장소 루트에서
python3 -m cpp_search.pipeline config/scenarios/baseline_tel.toml

# 검증만 (실행 없이 입력 오류 확인)
python3 -m cpp_search.pipeline my_scenario.toml --validate-only
```

잘못된 값은 **실행 전에 거부**된다 — 수십 분 돌고 나서 죽는 일이 없도록
모든 한도를 로딩 시점에 검사한다.

## 전체 예시 (주석이 곧 설명)

```toml
name = "baseline_tel"        # 결과 폴더명 — 빈칸·슬래시 금지
seed = 20260201              # 모든 난수의 뿌리: 지형·belief·MC·탐지

[seeker]
template = "SR-Z50"          # 프리셋이 기본값을 채우고, 아래 키가 우선
altitude_m = 600.0
detection_probability = 0.86
false_alarm_probability = 0.05

[seeker.eo]
resolution = [1280, 720]
hfov_deg = 18.0
# vfov_deg = 10.1            # 선언만 기록됨 (아래 '유의' 참조)

[seeker.ir]
resolution = [1280, 720]
hfov_deg = 18.0

[seeker.gimbal]
max_angle_deg = 45.0         # 김발 제어각
slew_rate_dps = 60.0         # 김발 전환 속도
scan_rate_dps = 30.0         # 탐색 스캔 속도

[vehicle]
count = 6
search_speed_kph = 160.0
transit_speed_kph = 160.0

[mission]
ingress_s = 480.0            # 이송 — 이 동안의 표적 확산이 belief에 반영됨
search_s = 600.0             # 탐색

[target]
case = "TEL"                 # Tank | TEL — 최악 거동 집합·AOI·인증 격자를 정함
max_speed_kph = 40.0
tp_error_sigma_m = 500.0     # 위치(제보) 오차 = 초기 belief 폭
# aoi_radius_m = 9950.0      # 생략하면 케이스 확정값(순변위 99%)

[terrain]
weighting = true             # 지형 가중(사전·전이) on/off
bias_strength = 0.6          # 이동 편향 (지형 가중치)
offroad_probability = 0.2
halt_probability_boost = 0.35
concealment_count = 3        # 은닉 지대 수 (은닉 가중치)

[grid]
kind = "square"              # square(N×N 외접, 기본) | operational-polar
n = 10                       # 셀 크기·슬라이스는 AOI에서 자동 유도

[solver]
kind = "milp"                # mappo는 v0.5 2단계에서 열림
certificate_gap = 0.01       # 사전선언 인증 기준 1%
# time_limit_s = 300         # master 시간 한도

[evaluation]
mc_samples = 1000
particles = 5000
```

모든 키는 생략 가능하다 — 생략하면 위 값(= v0.4 확정값)이 기본으로 들어간다.
최소 시나리오는 `name` 과 `[target] case` 두 줄이면 된다.

## seed 설정

`seed` 하나가 지형 생성·belief 입자·Monte Carlo·탐지 난수를 전부 결정한다.

| 목적 | 권장 seed |
|---|---|
| v0.4 인증 실적과 비교 | **20260201, 20260202, 20260203** (인증이 열린 planning seed 3개) |
| 새 조건 탐색 | 아무 값 — 단, 인증(gap ≤ 1%)이 열린다는 보장은 없음 |

주의: 인증 집합 밖 seed에서 gap이 1%를 넘는 것은 결함이 아니라 그 seed의
지형·belief에서 증명이 그만큼 어렵다는 뜻이다 (예: seed 20260830 → gap 2.6%).

## 필드별 한도 (어기면 로딩 시 거부)

| 필드 | 허용 범위 | 비고 |
|---|---|---|
| `name` | 빈칸·`/` 없는 문자열 | 결과 폴더명 |
| `vehicle.count` | 1~12 | |
| `seeker.*.hfov_deg` | 1~120° | |
| 확률류 (P_D, P_F, bias 등) | (0, 1] | |
| `grid.n` | 2~40 | square 전용 |
| `evaluation.particles` | 100~40,000 | 궤적 캐시 한도 4만 입자 |
| `solver.kind` | `milp` | `mappo`는 2단계 개방 예정 |
| `target.case` | `Tank`, `TEL` | 케이스가 AOI·거동 집합을 정함 |

## 자동 유도되는 값 (입력 아님)

- **AOI 반경**: 케이스 확정값 (TEL 9,950 m / Tank 8,884 m — 이송+탐색 1,080 s 순변위 99%)
- **셀 크기**: 외접 정사각형을 N등분 (TEL 10×10 → 1,990 m)
- **슬라이스 수**: T = 탐색시간 × 중심선속도 ÷ 셀 한 변 ("한 슬라이스에 한 칸")
- **탐색 시작 belief**: σ(제보 오차) + 이송 480 s 확산
- **탐색폭 W**: 고도·센서 스펙에서 적분

## 결과물 — `results/runs/<name>/`

| 파일 | 내용 |
|---|---|
| `metrics.json` | 모형 PD(최악표적·표적별), optimality gap, **인증 판정**, 실비행 MC PD(거동 평균·최악), 실패율, 조건부 평균 탐지시간, 면적 커버리지, model−flown, 해소된 격자 값 전부 |
| `figure1_weights.png` | 가중치 지도 3면 — 기동성·은닉·초기 belief 셀 질량 |
| `figure2_tracks.png` | step별 1~N 비행체 항적 (셀 중심 경유 본 표, ×=실제 발진 지점) |
| `scenario.toml` | 입력 스냅샷 — 이 파일이 재현의 근거 |

지표 읽는 법: 모형 PD(셀 모형) > 실비행 PD(센서 MC)인 것은 정상이다 —
v0.4 gap-decomposition 연구가 기록한 모형 계층(셀 → 연속 MC → 센서 비행)
그대로다. 탐지율과 평균 탐지시간은 **짝으로** 읽는다.

## 유의

1. **VFOV는 선언만 기록된다** — 현행 공간 탐지모델이 HFOV·고도 기반이라,
   실제 반영은 인증 결과가 바뀌는 모형 변경이어서 별도 결정 대기 중.
2. `solver.kind = "mappo"` 는 v0.5 2단계(MAPPO implementation)에서 열린다.
3. 새 탐색기는 `src/cpp_search/pipeline/scenario.py` 의 `SEEKER_TEMPLATES`
   에 한 항목으로 등록한다.
4. 동시 실행은 6~7개까지 (디스크 여유 2~5 GB).

참고 시나리오: `config/scenarios/baseline_tel.toml`(인증 조건),
`config/scenarios/smoke.toml`(수십 초 연기 시험),
`config/scenarios/demo_custom.toml`(입력 확장 4블록 시연 — 아래 절).

---

## 입력 확장 — 선언적 명시 입력 (2026-10-31 구현)

seed 추첨 대신 지형 배치·초기 belief·거동·평가 가설을 **직접 지정**하는
선택 블록 4개. 설계 원칙은 둘이다 — ① 모든 블록은 생략 가능하고, 생략하면
현행(케이스 확정값 + seed 추첨) 그대로다. ② 프로파일은 단일 출처다:
`[target.behaviour]` 는 config 오버레이(`behaviour_profiles.CUSTOM` + 배정
교체) 하나로 주입되어 belief 전개·셀 모형 전이행렬·진리 생성이 **같은
프로파일**을 보게 된다 (한 군데만 바꾸면 "계획이 상대한 표적과 다른 표적으로
채점"하는 사고가 재발한다). 계약은 `tests/test_scenario_extensions.py` 가
코드로 지킨다.

```toml
[target.behaviour]           # 배정 거동 오버라이드 — "움직임만 내 정의로"
base = "RELOCATION_HEAVY"    # 출발 프리셋, 아래 키가 그 위에 덮임
mode_probabilities = [0.05, 0.10, 0.65, 0.15, 0.05]  # π: HALT·LOW·HIGH·CTRV·MANV
persistence_alpha = 0.65     # α: 모드 유지 끈기
process_noise_scale = 0.75
turn_rate_scale = 0.75
mode_speed_bounds_kph = [[0,0],[0,20],[20,40],[2,32],[0,40]]

[belief]                     # 초기 belief — 생략 시 케이스 확정값
kind = "tp-centered"         # tp-centered | ring
sigma_m = 500.0              # (기존 target.tp_error_sigma_m 자리 이동, 호환 유지)
# center_m = [1500, -800]    # 선언만 기록 + 경고 — 현행 사전분포가 AOI 중심
#                              대칭이라 중심 이동은 모형 변경(결정 대기).
#                              지형 대비 상대 위치가 목적이면 terrain.layout
#                              좌표를 반대로 옮기면 같은 효과다.

[terrain.layout]             # 배치 명시 — 있으면 추첨 대신 사용, 가중 세기는 [terrain] 그대로
corridors    = [ { from = [-6000, 2000], to = [7000, -1000], width_m = 800 } ]
obstacles    = [ { center = [3000, 4000], radius_m = 1500 } ]
concealments = [ { center = [-2000, -3000], radius_m = 1200 } ]

[truth]                      # 평가 가설 명시 — 생략 시 아래 기본 규칙
hypotheses = [
  { x = 4200,  y = -1300, behaviour = "STOP_HEAVY" },
  { x = -2500, y = 3100,  behaviour = "custom" },   # custom = [target.behaviour] 프로파일
]
```

### 의미 규칙

| 블록 | 바꾸는 것 | 안 바꾸는 것 |
|---|---|---|
| `[target.behaviour]` | 배정 거동 (belief·계획·전이행렬·진리가 공통으로 쓰는 프로파일) | 센서 시그니처·AOI (그건 `case` 소관) |
| `[truth]` hypotheses | 평가 가설 집합 — 각 판 따로 채점, 최악 보고 | 배정 거동·belief |
| `[terrain.layout]` | 지형 배치 | 가중 세기 (`[terrain]` 키가 계속 적용) |
| `[belief]` | 초기 belief 종류·중심·폭 | 이송 480 s 확산 (여전히 자동) |

- **가설 기본 규칙 (확정)**: `[truth]` 를 생략하면 가설 = 케이스 최악 거동 집합.
  단 `[target.behaviour]` 가 있으면 **custom 프로파일을 그 집합에 추가**한다
  (기존 3개 + custom) — 보수성은 유지하면서 "내 거동이 실제였다면"도 채점된다.
- **seed 와의 관계**: 명시한 성분은 seed 소관에서 빠진다. seed(생략 시 기본값)는
  남은 무작위 — 입자 추첨·모드 전환 타이밍·탐지 주사위 — 만 결정하므로,
  시나리오 비교는 seed 를 건드리지 않고 입력만 바꾸는 것이 짝지은 비교가 된다.
- **표적 타입**: `case` 는 그대로 필수 선언 — 시그니처·AOI·최악 거동 집합을 정한다.
  behaviour 오버라이드는 "TEL 로 보이는데 움직임은 내 정의"를 만든다.
  새 표적 타입(새 시그니처)은 이 설계 범위 밖 — 표적 템플릿 등록이 따로 필요하다.

### 로딩 시 검증 (어기면 실행 전 거부)

| 필드 | 규칙 |
|---|---|
| `mode_probabilities` | 길이 5, 각 [0,1], 합 = 1 (±1e-6) |
| `persistence_alpha` | (0, 1] |
| `*_scale` | > 0 |
| `mode_speed_bounds_kph` | 5쌍, lo ≤ hi, HALT(첫 쌍) = [0,0], hi ≤ `target.max_speed_kph` |
| layout 좌표·반경 | AOI 원 안, `width_m`·`radius_m` > 0 |
| hypotheses | 좌표 AOI 안, `behaviour` 는 알려진 프리셋 또는 "custom"("custom" 은 `[target.behaviour]` 있을 때만) |

### 출력 규약

- 신설 블록을 하나라도 쓰면 `metrics.json` 에
  `provenance.custom_inputs = ["target.behaviour", ...]` 를 기록하고
  **"v0.4 인증 실적과 비교 불가"** 라벨을 남긴다 — 기준 표에 섞이는 사고 방지.
- 커스텀 프로파일의 결과 표기 이름은 `CUSTOM(base=RELOCATION_HEAVY)`.
- 재현성은 현행 그대로 — `scenario.toml` 스냅샷이 입력 전부다.

### 구현 메모 (2026-10-31)

- `pipeline/scenario.py` — 파싱·거부 규칙 전부와 config 오버레이(CUSTOM 주입).
  좌표 검증은 AOI 가 해소되는 `pipeline/run.py` 초입에서 실행 전에 거부한다.
- `models/terrain.py::build_declared_terrain` — 선언 배치 그대로 지형 구성.
- `models/simulation.py` / `models/truth.py` — `initial_points` 통로로 명시
  가설이 그 좌표에서 출발한다 (궤적별 난수열 규약은 추첨 경로와 동일).
- 결과 표기: 계획 표적에 `…/CUSTOM(base=<프리셋>)`, 평가 가설에 `<거동>@(x,y)`.
- **편차 1건**: `belief.center_m` 은 선언만 기록된다 (위 주석 참조).
