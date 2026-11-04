# 시나리오 작성 매뉴얼 (v0.5 pipeline)

**한 파일 = 한 실험.** 시나리오 TOML 하나가 실험의 입력 전부를 선언하고,
실행하면 `results/runs/<이름>/` 에 입력 스냅샷·지표·그림이 남는다.
그 파일을 그대로 다시 돌리면 재현된다.

## 실행

저장소 루트에서, Python 3.11 이상 (`tomllib` 필요):

```bash
# make 타깃이 PYTHONPATH=src 를 스스로 붙인다
make scenario S=config/scenarios/baseline_tel.toml

# 직접 실행하려면 PYTHONPATH=src 가 필요하다 (make install 뒤에는 불필요)
PYTHONPATH=src python3 -m cpp_search.pipeline config/scenarios/baseline_tel.toml

# 검증만 — 실행 없이 입력 오류 확인 (한도·오타·config 해소·좌표·거동 이름 전부)
PYTHONPATH=src python3 -m cpp_search.pipeline my_scenario.toml --validate-only
```

플래그: `--out <루트>` 결과 루트 변경(기본 `results/runs`), `--resolve`
저장된 plan.json 을 무시하고 다시 solve (아래 결과물 절).

잘못된 값은 **실행 전에 거부**된다 — 수십 분 돌고 나서 죽는 일이 없도록
한도·config 해소·좌표·가설 거동 이름을 solver 전에 전부 검사하고,
**목록에 없는 키는 오타로 보고 거부**한다 (오타가 기본값으로 조용히
떨어져 쓴 것과 다른 실험이 도는 사고 방지, 2026-11-04).

대략의 실행 시간 (Apple Silicon): `smoke.toml` 수십 초,
10×10 인증 조건(`baseline_tel` 류)은 **solve 가 수십 분**이고 gap 이
기준을 넘으면 자동 재시도로 **두 배**까지 간다. 같은 입력의 재실행은
plan.json 재사용으로 MC 평가만 돌아 수 분이다.

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
search_s = 600.0             # 탐색 — 480+600에서 벗어나면 provenance에
                             # 남고 v0.4 앵커와 비교 불가 (아래 출력 규약).
                             # AOI는 케이스 확정값이라 따라 커지지 않으니
                             # 시간을 크게 바꾸면 aoi_radius_m도 함께 조정

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
background = 0.06            # 지형 특징 없는 셀의 바닥 가중
corridor_count = 2           # ↓ 셋은 seed 추첨 개수 — [terrain.layout] 을
barrier_count = 1            #   쓰면 추첨 자체가 없어 무시된다
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
필수는 `name` 하나다 (`target.case` 생략 시 TEL). 반대로 **위 목록에 없는
키를 적으면 거부**된다 — 받는 키의 전부가 `scenario.py` 의 `_SCHEMA` 다.

## seed 설정

`seed` 하나가 지형 생성·belief 입자·Monte Carlo·탐지 난수를 전부 결정한다.
생략하면 **기본 seed 20260830** 이 들어간다 — 주의: seed 는 provenance
감시 대상이 아니므로(같은 코드의 다른 추첨일 뿐), seed 를 생략한 실행도
`comparable_to_v04_anchors = true` 로 나올 수 있다. anchor(20260201~03)
실적 표에 나란히 실으려면 seed 까지 같아야 한다는 것은 사람이 지켜야 한다.

| 목적 | 권장 seed |
|---|---|
| v0.4 인증 실적과 비교 | **20260201, 20260202, 20260203** (인증이 열린 planning seed 3개) |
| 새 조건 탐색 | 아무 값 — 단, 인증(gap ≤ 1%)이 열린다는 보장은 없음 |

주의: 인증 집합 밖 seed에서 gap이 1%를 넘는 것은 결함이 아니라 그 seed의
지형·belief에서 증명이 그만큼 어렵다는 뜻이다 (예: seed 20260830 → gap 2.6%).
또한 같은 인스턴스라도 solve 실행마다 gap이 흔들릴 수 있다 — tel_0201_layout
에서 같은 입력이 4.07%(미인증) → 0.91%(인증)로 갈린 관측 (2026-11-03).

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
| `metrics.json` | 모형 PD(최악표적·표적별), optimality gap, **인증 판정**, MC PD(표적별·평균·최악), 실패율, 면적 커버리지(경로 기하), model−MC, 해소된 격자 값 전부 |
| `plan.json` | 셀 경로·인증 경계 + 입력 fingerprint + solver 설정 — **같은 입력의 재실행은 다시 풀지 않고 이 계획을 재사용**한다 (solve 는 실행마다 다른 계획을 낼 수 있음 — 위 비결정성 관측). solver 설정(`time_limit_s`·`certificate_gap`)만 바꿔도 다시 푼다. 강제로 다시 풀려면 `--resolve` |
| `figure1_weights.png` | 가중치 지도 3면 — 기동성·은닉·초기 belief 셀 질량 |
| `figure2_tracks.png` | step별 1~N 비행체 항적 (셀 중심 경유 본 표, ×=실제 발진 지점) |
| `scenario.toml` | 입력 스냅샷 — 이 파일이 재현의 근거 |

실행 중 피드백·인증 재시도 (2026-11-03): solve 는 SPX 반복마다
`SPX 반복 n 비탐지 하한 · 상한 · gap` 한 줄을 찍는다 — 수십 분 실행이
살아 있는지 밖에서 보인다. 그리고 **인증 재시도 정책**: 첫 solve 의 gap 이
기준(1%)을 넘으면 자동으로 **한 번만** 다시 풀고 둘 중 gap 이 작은 쪽을
쓴다 (`metrics.json` 의 `solver.solve_attempts` 에 횟수가 남는다).
비결정성 때문에 인증이 실행 운에 갈리는 것을 줄이는 공표 절차다.
실측 사례 (tel_0201_layout, 2026-11-04): 1차 solve gap 1.073% 미인증 →
자동 재시도 → 0.943% 인증.

### metrics.json 읽는 법 (핵심 키)

| 키 | 뜻 |
|---|---|
| `provenance.custom_inputs` | v0.4 기본형에서 벗어난 입력의 전체 목록 (기본형과의 전수 diff). 비어 있지 않으면 `comparable_to_v04_anchors = false` |
| `resolved.*` | 입력에서 자동 유도된 값 전부 — AOI·셀 크기·슬라이스 수·발사 지점·표적 이름·인스턴스 fingerprint |
| `solver.model_detection_probability_by_target` | **셀 모형** 표적별 PD (minimax 계획의 값) |
| `solver.relative_optimality_gap` / `certified` | SPX 인증폭과 기준(1%) 판정 |
| `solver.lower/upper_bound_nondetection` | 절단평면 하한 / 정수 master 상한 (비탐지확률) — gap 의 두 끝 |
| `solver.plan_reused` / `solve_attempts` | plan.json 재사용 여부 / 이번 실행의 solve 횟수 (재사용이면 0, 재시도면 2). 재사용 시 `runtime_s` 는 **저장 당시** solve 시간 |
| `mc.kind` | MC 장치 — `continuous-centre-sync`(기본) 또는 `flight-hypotheses`([truth] 있을 때) |
| `mc.by_target` / `..._mean` / `..._worst` | MC 표적별·평균·최악 PD. 최악에는 누가 최악인지 이름이 같이 남는다 |
| `mc.model_minus_mc` | 기본 장치: 모형 최악 − MC 최악 (= 셀 모형 가정의 몫). 실비행 장치: 모형 최악 − MC **평균** (v0.4 model−flown 정의) |
| `mc.*detection_time*` | **실비행 장치에만 있다** — 연속 MC 는 생존곡선 적분이라 개별 탐지시각이 없다 |
| `geometry.unique_area_coverage_ratio` | 계획 경로 기하의 AOI 면적 커버리지 (MC 아님) |

**MC 장치 (2026-11-03 B안 확정)**: MC 열은 **규약 맞춤 연속 MC** 다 —
셀 모형과 항적 규약(centre-sync)·지수 소인 법칙·이송 처리(표적만 480 s
확산)를 전부 맞추고, 표적이 셀 안 실제 위치를 기억한다는 것만 현실로
되돌린 평가. 그래서 `model−MC` 가 곧 셀 모형 가정(주로 Markov 섞임)의
몫이다 (erasure 연구: 섞임이 격차의 93~100%). 모형 PD > MC PD 는 정상.
예외: `[truth]` 고정 출발점 가설이 있으면 그 판만 **실비행 MC**(이송·실궤적·
센서 주사위 포함, `kind = "flight-hypotheses"`, 탐지시간 포함)로 채점된다 —
고정 출발점을 지원하는 유일한 장치라서다. `evaluation.mc_samples` 는 이
실비행 경로에만 쓰이고, 표준 경로의 입자 수는 `evaluation.particles` 다.

## 유의

1. **VFOV는 선언만 기록된다** — 현행 공간 탐지모델이 HFOV·고도 기반이라,
   실제 반영은 인증 결과가 바뀌는 모형 변경이어서 별도 결정 대기 중.
2. `solver.kind = "mappo"` 는 v0.5 2단계(MAPPO implementation)에서 열린다.
3. 새 탐색기는 `src/cpp_search/pipeline/scenario.py` 의 `SEEKER_TEMPLATES`
   에 한 항목으로 등록한다. 단, **실비행 MC**(`[truth]` 가설 채점)의 공간
   탐지모델은 아직 SR-Z50 성능표가 박혀 있다 — 다른 탐색기 스펙은 계획
   모형·규약 맞춤 MC 에는 반영되지만 실비행 채점에는 반영되지 않는다.
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

- **provenance 전수 감시 (2026-11-03)**: v0.4 확정값(= 모든 키 생략의
  기본형)에서 벗어난 입력 키를 **전부** 자동 감지해 `metrics.json` 의
  `provenance.custom_inputs = ["vehicle.count", "target.behaviour", ...]` 에
  키 경로로 기록하고 **"v0.4 인증 실적과 비교 불가"** 라벨을 남긴다 —
  기준 표에 섞이는 사고 방지. 예외는 셋뿐: `name`(폴더명), `seed`(같은
  실험의 다른 추첨), `evaluation.mc_samples`(평가 정밀도만). 기본값을
  명시로 적은 것은 같은 세계라 올라가지 않는다.
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
