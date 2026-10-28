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
`config/scenarios/smoke.toml`(수십 초 연기 시험).
