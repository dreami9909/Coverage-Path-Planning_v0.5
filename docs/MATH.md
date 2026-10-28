# 수식 색인 — 어떤 식이 어디에 구현되어 있나

각 행의 위치는 저장소 기준 `파일:줄`이다. 코드를 고치면 줄 번호는 바뀌므로,
찾을 때는 줄 번호보다 **함수 이름**을 먼저 보는 편이 안전하다.
이 표는 `make docs`로 다시 생성할 수 있다.


## Ch1 센서·탐색범위

| 수식 / 개념 | 구현 위치 |
|---|---|
| swath = 2 H tan(HFOV/2) | `src/cpp_search/models/models.py:190` |
| s = W (1 - overlap) | `src/cpp_search/models/models.py:204` |
| PD(x) = e + (1-e)(1+cos(pi u))/2 | `src/cpp_search/models/search_envelope.py:192` |
| W = ∫ PD(x) dx  (Simpson) | `src/cpp_search/models/search_envelope.py:221` |
| weave 호 길이 ∫ sqrt(1+(A k cos)^2) ds | `src/cpp_search/models/search_envelope.py:84` |
| v_centerline = v_actual / (L(lambda)/lambda) | `src/cpp_search/models/search_envelope.py:99` |

## Ch1 확률 계약

| 수식 / 개념 | 구현 위치 |
|---|---|
| l = ln(p/(1-p)), p = sigmoid(l) | `src/cpp_search/models/belief.py:53` |
| l += ln(PD/PF) / ln((1-PD)/(1-PF)) | `src/cpp_search/models/belief.py:121` |
| POC / POS / POD | `src/cpp_search/models/belief.py:157` |

## Ch1 자원 상한

| 수식 / 개념 | 구현 위치 |
|---|---|
| A_max = n v T W, ceiling = A_max/(pi R^2) | `src/cpp_search/experiments/ch1_evaluation.py:193` |

## Ch1 AOI 사이징

| 수식 / 개념 | 구현 위치 |
|---|---|
| 포함률 기반 탐색반경 | `src/cpp_search/models/aoi.py:185` |

## 표적 운동

| 수식 / 개념 | 구현 위치 |
|---|---|
| P = alpha I + (1-alpha) 1 pi^T | `src/cpp_search/models/motion.py:135` |
| CTRV 전파 (v/omega)(sin psi' - sin psi) | `src/cpp_search/models/motion.py:548` |
| 원 밖 흡수 처리 | `src/cpp_search/models/motion.py:435` |

## 사전확률·지도

| 수식 / 개념 | 구현 위치 |
|---|---|
| f(r) = exp(-(r-mu)^2 / 2 sigma^2) | `src/cpp_search/models/probability.py:86` |
| A_cell = (r_out^2 - r_in^2) dtheta / 2 | `src/cpp_search/models/probability.py:149` |

## 입자필터

| 수식 / 개념 | 구현 위치 |
|---|---|
| ESS = 1 / sum w_i^2 | `src/cpp_search/models/particle_filter.py:765` |
| systematic / stratified 재표본추출 | `src/cpp_search/models/particle_filter.py:827` |
| roughening sigma = K E N^(-1/2) | `src/cpp_search/models/particle_filter.py:902` |
| 음성관측 w <- w exp(-Lambda) | `src/cpp_search/models/particle_filter.py:583` |
| 입자 -> 셀 질량 투영 | `src/cpp_search/models/particle_filter.py:748` |

## EO/IR 탐지

| 수식 / 개념 | 구현 위치 |
|---|---|
| GSD = 2R tan(HFOV/2)/W_px, n_px | `src/cpp_search/models/sensor_observation.py:657` |
| SNR = SNR_ref (R_ref/R)^2 contrast | `src/cpp_search/models/sensor_observation.py:667` |
| P_fused = 1 - (1-P_eo)(1-P_ir) | `src/cpp_search/models/sensor_observation.py:774` |
| lambda = -ln(1-PD)/t_scan | `src/cpp_search/models/sensor_observation.py:882` |
| P(노출 t) = 1 - exp(-lambda t) | `src/cpp_search/models/sensor_observation.py:884` |

## 지형 가중

| 수식 / 개념 | 구현 위치 |
|---|---|
| 통로 가중 exp(-d^2/2sigma^2) | `src/cpp_search/models/terrain.py:86` |
| 이동 편향 v <- (1-b)v + b grad(mobility) | `src/cpp_search/models/terrain.py:366` |
| 관측성 가중치 (셀 hazard 배율) | `src/cpp_search/models/terrain.py:386` |
| 합성지형 생성 (회랑·장애물·은폐) | `src/cpp_search/models/terrain.py:696` |
| 대조군 지형 (회랑 없음) | `src/cpp_search/models/terrain.py:751` |

## 경로 KPI

| 수식 / 개념 | 구현 위치 |
|---|---|
| 임무시간 클리핑 t = L_flown / v | `src/cpp_search/models/evaluation.py:207` |
| 고유 면적률 / 중복률 | `src/cpp_search/models/evaluation.py:315` |
| 확률질량 탐색률 | `src/cpp_search/models/evaluation.py:172` |

## 탐지시간 평가

| 수식 / 개념 | 구현 위치 |
|---|---|
| 상대운동 노출구간 (2차 방정식) | `src/cpp_search/models/simulation.py:984` |
| 누적 위험률 역변환 표집 | `src/cpp_search/models/simulation.py:936` |
| Lambda(t) = threshold 선형 보간 | `src/cpp_search/models/simulation.py:970` |

## Ch2 경로제약 기준식

| 수식 / 개념 | 구현 위치 |
|---|---|
| 미탐지 재귀 u_{t+1} = (u_t e^{-a}) P_t | `src/cpp_search/theory/path_constrained.py:361` |
| hazard 합 exp(-sum_j a_j) | `src/cpp_search/theory/path_constrained.py:335` |
| 유효 hazard a_eff = a(t,k) f[i,k] | `src/cpp_search/theory/path_constrained.py:190` |
| PD(경로묶음) = 1 - sum u_{T-1}^+ | `src/cpp_search/theory/path_constrained.py:406` |

## Ch2 warm start

| 수식 / 개념 | 구현 위치 |
|---|---|
| team H1 후퇴지평 (SPX 초기해 전용) | `src/cpp_search/theory/path_constrained.py:794` |
| team H2 후보 PD 비교 | `src/cpp_search/theory/path_constrained.py:981` |

## Ch2 Markov 추정

| 수식 / 개념 | 구현 위치 |
|---|---|
| pi_0(x) = sum_{i in x} w_i | `src/cpp_search/theory/markov.py:127` |
| P_t(x,y) = N_t(x,y)/sum_y N_t(x,y) | `src/cpp_search/theory/markov.py:191` |
| 도달가능 A(x,y) = [d <= reach] | `src/cpp_search/theory/markov.py:211` |

## Ch2 Stone SP1/SPX

| 수식 / 개념 | 구현 위치 |
|---|---|
| Markov PND f(Y) 와 gradient | `src/cpp_search/theory/stone_path.py:1768` |
| SP1 동질·단일표적 (4.23)-(4.28) | `src/cpp_search/theory/stone_path.py:134` |
| SPX 이질·다중표적 minimax (4.48)-(4.56) | `src/cpp_search/theory/stone_path.py:200` |
| 접평면 f(Yi)+grad^T(Y-Yi) <= eta (4.3.1) | `src/cpp_search/theory/stone_path.py:469` |
| 셀 점유한도 (4.53) | `src/cpp_search/theory/stone_path.py:418` |
| 양립불가 이동 (4.54) | `src/cpp_search/theory/stone_path.py:443` |
| 분리반경 clique 절단 | `src/cpp_search/theory/stone_path.py:1491` |
| 정확 생존 MILP (곱항 선형화) | `src/cpp_search/theory/stone_path.py:1108` |

## Ch2 지형결합 belief

| 수식 / 개념 | 구현 위치 |
|---|---|
| 지형 3성분 주입 지점 | `src/cpp_search/solvers/milp/terrain_belief.py:134` |

## Ch2 인스턴스

| 수식 / 개념 | 구현 위치 |
|---|---|
| 입자 -> 축약 다중표적 Markov 투영 | `src/cpp_search/solvers/milp/stone_spx.py:352` |
| 공통 입력 지문 (두 계획법 동일성 증거) | `src/cpp_search/solvers/milp/stone_spx.py:627` |
| hazard = W v dt / A_cell x scale x observability | `src/cpp_search/solvers/milp/stone_spx.py:885` |

## Ch2 경로변환

| 수식 / 개념 | 구현 위치 |
|---|---|
| n_track = round(sqrt(L/s)) 정사각 블록 | `src/cpp_search/solvers/milp/routes.py:52` |
| 시간예산 회계 (실제 소요시간으로 축소) | `src/cpp_search/solvers/milp/routes.py:112` |

## Ch3 SPX 환경

| 수식 / 개념 | 구현 위치 |
|---|---|
| 행동 후보 = belief 점수 상위 K 인접셀 | `src/cpp_search/solvers/mappo/spx_env.py:194` |
| 점유·분리 복구 (실행가능집합 유지) | `src/cpp_search/solvers/mappo/spx_env.py:350` |
| 온라인 미탐지 재귀 (= nondetection_trace) | `src/cpp_search/solvers/mappo/spx_env.py:417` |
| minimax 성형 w_i ∝ exp(-PD_i/tau) | `src/cpp_search/solvers/mappo/spx_env.py:447` |
| 종단 보상 = min_i PD_i | `src/cpp_search/solvers/mappo/spx_env.py:341` |

## Ch3 MAPPO

| 수식 / 개념 | 구현 위치 |
|---|---|
| TD 오차 delta_t | `src/cpp_search/solvers/mappo/mappo.py:925` |
| GAE 역방향 누적 A_t = delta_t + gamma lambda A_{t+1} | `src/cpp_search/solvers/mappo/mappo.py:930` |
| PPO clipped surrogate | `src/cpp_search/solvers/mappo/mappo.py:858` |

## Ch3 비교

| 수식 / 개념 | 구현 위치 |
|---|---|
| 학습효과 = 학습 - 미학습 (결정론적 평가) | `src/cpp_search/solvers/mappo/spx_policy.py:202` |
| SPX - MAPPO (같은 인스턴스) | `src/cpp_search/solvers/mappo/spx_policy.py:250` |

## Ch4 실비행 평가

| 수식 / 개념 | 구현 위치 |
|---|---|
| 계획모형 - 실비행 차이 (부호는 측정 결과) | `src/cpp_search/experiments/ch4_synthetic_terrain.py:307` |
| 표적 계층 가중 합성 | `src/cpp_search/experiments/ch4_synthetic_terrain.py:95` |

## Ch4 통계

| 수식 / 개념 | 구현 위치 |
|---|---|
| 짝지은 seed 군집 부트스트랩 백분위 구간 | `src/cpp_search/reporting/reliability.py:149` |
| 짝지은 차이 + 부호 방향 명시 | `src/cpp_search/reporting/reliability.py:188` |
| Wilson 점수 구간 | `src/cpp_search/reporting/reliability.py:112` |

## Ch4 KPI

| 수식 / 개념 | 구현 위치 |
|---|---|
| 공통 KPI 블록 (5 순위지표 + 진단) | `src/cpp_search/reporting/kpi.py:81` |
| seed 군집 요약 + 신뢰성 판정 | `src/cpp_search/reporting/kpi.py:218` |


# 의존 그래프 (import 기준, 자동 추출)

`쓰는 것`은 그 모듈이 import하는 저장소 내부 모듈, `쓰이는 곳`은 그 모듈을
import하는 모듈이다. 패키지 `__init__.py`의 재수출은 제외하지 않았으므로,
`theory`/`planning`/`learning` 같은 패키지 이름이 보이면 그 패키지의
`__init__.py`를 통해 들어온 것이다.


| 모듈 | 쓰는 것 | 쓰이는 곳 |
|---|---|---|
| `cpp_search` | `models.models`, `models.profiles`, `models.search_envelope`, `models.sensor_observation` | — |
| `__main__` | `experiments.runner` | — |
| `common` | — | — |
| `common.config` | `models.models`, `models.motion`, `models.profiles`, `models.search_envelope`, `models.sensor_observation` | `experiments.ch1_evaluation`, `experiments.ch2_stone_spx`, `experiments.ch2_studies`, `experiments.ch3_mappo_stone`, `experiments.ch4_synthetic_terrain`, `experiments.runner`, `models.aoi`, `models.truth`, `pipeline.scenario`, `reporting.reliability` |
| `common.options` | — | `experiments.ch1_evaluation`, `experiments.ch2_stone_spx`, `experiments.ch2_studies`, `experiments.ch3_mappo_stone`, `experiments.ch4_synthetic_terrain`, `experiments.runner`, `pipeline.scenario` |
| `experiments` | `experiments` | `experiments`, `experiments.ch2_stone_spx`, `experiments.runner` |
| `experiments.ch1_evaluation` | `common.config`, `common.options`, `models.aoi`, `models.belief`, `models.models`, `models.profiles`, `models.sensor_observation` | `experiments.ch2_stone_spx` |
| `experiments.ch2_stone_spx` | `common.config`, `common.options`, `experiments`, `experiments.ch1_evaluation`, `models.models`, `models.motion`, `models.terrain`, `models.truth`, `reporting.reliability`, `solvers.milp.stone_spx` | `experiments.ch2_studies`, `experiments.ch3_mappo_stone`, `experiments.ch4_synthetic_terrain`, `pipeline.run` |
| `experiments.ch2_studies` | `common.config`, `common.options`, `experiments.ch2_stone_spx`, `models.models`, `solvers.milp.stone_spx`, `theory.stone_path` | `experiments.ch3_mappo_stone` |
| `experiments.ch3_mappo_stone` | `common.config`, `common.options`, `experiments.ch2_stone_spx`, `experiments.ch2_studies`, `models.models`, `reporting.reliability`, `solvers.mappo.spx_env`, `solvers.mappo.spx_policy`, `solvers.milp.stone_spx` | `experiments.ch4_synthetic_terrain` |
| `experiments.ch4_synthetic_terrain` | `common.config`, `common.options`, `experiments.ch2_stone_spx`, `experiments.ch3_mappo_stone`, `models.evaluation`, `models.models`, `models.sensor_observation`, `models.simulation`, `models.terrain`, `models.truth`, `reporting.kpi`, `reporting.reliability`, `solvers.mappo.spx_policy`, `solvers.milp.stone_spx` | — |
| `experiments.runner` | `common.config`, `common.options`, `experiments`, `reporting.figures`, `reporting.kpi` | `__main__` |
| `models` | — | — |
| `models.aoi` | `common.config`, `models.models`, `models.simulation`, `models.terrain`, `models.truth` | `experiments.ch1_evaluation` |
| `models.belief` | — | `experiments.ch1_evaluation` |
| `models.evaluation` | `models.models` | `experiments.ch4_synthetic_terrain`, `models.simulation`, `pipeline.run`, `reporting.kpi` |
| `models.geometry` | `models.models` | `models.probability` |
| `models.models` | `models.search_envelope` | `cpp_search`, `common.config`, `experiments.ch1_evaluation`, `experiments.ch2_stone_spx`, `experiments.ch2_studies`, `experiments.ch3_mappo_stone`, `experiments.ch4_synthetic_terrain`, `models.aoi`, `models.evaluation`, `models.geometry`, `models.motion`, `models.particle_filter`, `models.probability`, `models.sensor_observation`, `models.simulation`, `models.terrain`, `models.truth`, `solvers.milp.routes`, `solvers.milp.stone_spx`, `solvers.milp.terrain_belief`, `theory.markov` |
| `models.motion` | `models.models` | `common.config`, `experiments.ch2_stone_spx`, `models.particle_filter`, `models.profiles`, `models.simulation`, `models.truth` |
| `models.particle_filter` | `models.models`, `models.motion`, `models.probability` | `solvers.milp.terrain_belief` |
| `models.probability` | `models.geometry`, `models.models` | `models.particle_filter`, `models.simulation`, `models.truth`, `solvers.milp.stone_spx`, `solvers.milp.terrain_belief`, `theory.markov` |
| `models.profiles` | `models.motion`, `models.sensor_observation` | `cpp_search`, `common.config`, `experiments.ch1_evaluation`, `models.truth`, `solvers.milp.stone_spx`, `solvers.milp.terrain_belief` |
| `models.search_envelope` | — | `cpp_search`, `common.config`, `models.models` |
| `models.searcher_track` | — | `solvers.mappo.continuous_env`, `solvers.milp.continuous_mc` |
| `models.sensor_observation` | `models.models` | `cpp_search`, `common.config`, `experiments.ch1_evaluation`, `experiments.ch4_synthetic_terrain`, `models.profiles`, `pipeline.run` |
| `models.simulation` | `models.evaluation`, `models.models`, `models.motion`, `models.probability` | `experiments.ch4_synthetic_terrain`, `models.aoi`, `models.truth`, `pipeline.run`, `reporting.kpi` |
| `models.terrain` | `models.models` | `experiments.ch2_stone_spx`, `experiments.ch4_synthetic_terrain`, `models.aoi`, `pipeline.figures`, `pipeline.run` |
| `models.truth` | `common.config`, `models.models`, `models.motion`, `models.probability`, `models.profiles`, `models.simulation` | `experiments.ch2_stone_spx`, `experiments.ch4_synthetic_terrain`, `models.aoi`, `pipeline.run` |
| `pipeline` | `pipeline.scenario` | — |
| `pipeline.__main__` | `pipeline.run`, `pipeline.scenario` | — |
| `pipeline.figures` | `models.terrain`, `reporting.figures`, `solvers.milp.stone_spx` | `pipeline.run` |
| `pipeline.run` | `experiments.ch2_stone_spx`, `models.evaluation`, `models.sensor_observation`, `models.simulation`, `models.terrain`, `models.truth`, `pipeline.figures`, `pipeline.scenario`, `reporting.kpi`, `solvers.milp.stone_spx` | `pipeline.__main__` |
| `pipeline.scenario` | `common.config`, `common.options` | `pipeline`, `pipeline.__main__`, `pipeline.run` |
| `reporting` | — | — |
| `reporting.figures` | `reporting.kpi` | `experiments.runner`, `pipeline.figures` |
| `reporting.kpi` | `models.evaluation`, `models.simulation`, `reporting.reliability` | `experiments.ch4_synthetic_terrain`, `experiments.runner`, `pipeline.run`, `reporting.figures` |
| `reporting.reliability` | `common.config` | `experiments.ch2_stone_spx`, `experiments.ch3_mappo_stone`, `experiments.ch4_synthetic_terrain`, `reporting.kpi` |
| `solvers` | — | — |
| `solvers.mappo` | `solvers.mappo.mappo`, `solvers.mappo.spx_env`, `solvers.mappo.spx_policy` | — |
| `solvers.mappo.continuous_env` | `models.searcher_track`, `solvers.mappo.spx_env`, `solvers.milp.continuous_mc` | — |
| `solvers.mappo.mappo` | `solvers.mappo.nets` | `solvers.mappo`, `solvers.mappo.spx_policy` |
| `solvers.mappo.nets` | — | `solvers.mappo.mappo` |
| `solvers.mappo.spx_env` | `solvers.milp.stone_spx`, `theory` | `experiments.ch3_mappo_stone`, `solvers.mappo`, `solvers.mappo.continuous_env`, `solvers.mappo.spx_policy` |
| `solvers.mappo.spx_policy` | `solvers.mappo.mappo`, `solvers.mappo.spx_env`, `solvers.milp.stone_spx` | `experiments.ch3_mappo_stone`, `experiments.ch4_synthetic_terrain`, `solvers.mappo` |
| `solvers.milp` | `solvers.milp.routes`, `solvers.milp.stone_spx`, `solvers.milp.terrain_belief` | — |
| `solvers.milp.continuous_mc` | `models.searcher_track`, `solvers.milp.terrain_belief`, `theory.path_constrained` | `solvers.mappo.continuous_env` |
| `solvers.milp.routes` | `models.models` | `solvers.milp`, `solvers.milp.stone_spx` |
| `solvers.milp.stone_spx` | `models.models`, `models.probability`, `models.profiles`, `solvers.milp.routes`, `solvers.milp.terrain_belief`, `theory`, `theory.markov`, `theory.path_constrained`, `theory.stone_path` | `experiments.ch2_stone_spx`, `experiments.ch2_studies`, `experiments.ch3_mappo_stone`, `experiments.ch4_synthetic_terrain`, `pipeline.figures`, `pipeline.run`, `solvers.mappo.spx_env`, `solvers.mappo.spx_policy`, `solvers.milp` |
| `solvers.milp.terrain_belief` | `models.models`, `models.particle_filter`, `models.probability`, `models.profiles` | `solvers.milp`, `solvers.milp.continuous_mc`, `solvers.milp.stone_spx` |
| `theory` | `theory.markov`, `theory.path_constrained`, `theory.stone_path` | `solvers.mappo.spx_env`, `solvers.milp.stone_spx`, `theory.path_constrained`, `theory.stone_path` |
| `theory.markov` | `models.models`, `models.probability` | `solvers.milp.stone_spx`, `theory` |
| `theory.path_constrained` | `theory` | `solvers.milp.continuous_mc`, `solvers.milp.stone_spx`, `theory`, `theory.stone_path` |
| `theory.stone_path` | `theory`, `theory.path_constrained` | `experiments.ch2_studies`, `solvers.milp.stone_spx`, `theory` |
| `theory.transitions` | — | — |
