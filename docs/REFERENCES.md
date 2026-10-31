# 탐지모델 계수의 문헌 근거

`src/cpp_search/models/sensor_observation.py`의 위장 계수(`CamouflageSpec`)와
채널 알고리즘 효율(`algorithm_efficiency`)은 지금까지 "A급 모델링 가정"으로만
선언되어 있었다. 이 문서는 각 수치를 공개 문헌의 측정값·모델 표준값으로
괄호치기(bracketing)한 근거를 정리한다. **계수를 문헌이 직접 지정하는 것은
아니다** — 아래 근거는 "이 값이 문헌 보고 범위 안에 있다"는 수준의 정당화이며,
민감도 분석 대상이라는 기존 입장은 유지한다.

## 1. 채널 확률 수식 구조 — Johnson 기준과 TTPF

코드의 픽셀 항

    P_px = sigma( (n_px - n50) / dn )

은 미 육군 NVESD(Night Vision and Electronic Sensors Directorate) 계열 모델의
표준 수식인 **TTPF(Target Transfer Probability Function)** 의 로지스틱 근사다.

* TTPF 원형 (NVTherm 사용자 매뉴얼, NVESD):

      P(N) = (N/N50)^E / (1 + (N/N50)^E),   E = 3.76

  ACQUIRE 모델 계열은 E = 2.7 + 0.7·(N/N50)을 쓴다 (미 특허 7,149,652,
  "Method for modeling detection of camouflaged targets").
* **N50 (50% 과업 성공에 필요한 표적 횡단 사이클 수, 1 cycle = 2 px)**
  — NVTherm 매뉴얼 §2.8.4~2.8.6:

  | 과업 | N50 (cycles) | 픽셀 환산 |
  |---|---|---|
  | 탐지(detection), 저클러터 | 0.75 | 1.5 px |
  | 탐지, 고클러터 | ~3.0 | 6 px |
  | 인식(recognition), 전차 vs 장갑차 | 4~5 | 8~10 px |
  | 식별(identification) | 6~9 | 12~18 px |

* **코드 값과의 대응**: `pixels_50` EO=8 px(=4 cycles), IR=6 px(=3 cycles).
  즉 EO는 "고클러터 탐지~인식 경계", IR은 "고클러터 탐지 상한"에 해당하는
  **보수적** 설정이다. 저클러터 탐지 기준(1.5~2 px)을 쓰면 탐지확률이 크게
  올라가므로, 본 연구의 설정은 문헌 범위 내에서 비관 쪽 끝을 택한 것으로
  기술하면 된다.

근거 문헌:
* J. Johnson, "Analysis of image forming systems," Proc. Image Intensifier
  Symposium, 1958. (사이클 기준의 원전)
* NVESD, *Night Vision Thermal Imaging Systems Performance Model (NVTherm)
  User's Manual*. <https://zieg.com/links/sensors/NVThermManual5.pdf>
* G. Sjaardema, C. Smith, *History and Evolution of the Johnson Criteria*,
  Sandia National Laboratories. <https://www.osti.gov/servlets/purl/1222446>
* US Patent 7,149,652, *Method for modeling detection of camouflaged targets*
  (ACQUIRE 기반 위장표적 탐지확률 모델링; 중간 클러터 탐지 N50≈1.5).
* R. Vollmerhausen & E. Jacobs의 TTP 계량(Johnson→TTP 전환비 1:2.7):
  "Target detection cycle criteria when using the targeting task performance
  metric," Proc. SPIE 5612 (2004).

## 2. 채널 알고리즘 효율 η — EO 0.90 / IR 0.86

`algorithm_efficiency`는 "픽셀·SNR·거리 조건이 모두 충족돼도 탐지기가 놓치는
몫"을 나타내는 점근 상한이다. 문헌의 두 갈래가 이 값을 괄호친다.

**(a) 현장 관측자 실험의 센서별 탐지 성공률 상한** — Vitoul et al. (ICINCO
2025) 지상 감시 실험, Table 2:

| 관측 수단 | 탐지 성공률 |
|---|---|
| 육안 | 92% |
| MOSKITO (다중센서 관측장비) | 88% |
| 주간 관측경(spotting scope) | 80% |
| 쌍안경 | 78% |
| 열상 조준경 | 70% |
| UAV | 65% |

광학(EO) 계열이 열상(IR)보다 일관되게 높다 — 코드의 η_EO(0.90) > η_IR(0.86)
순서와 일치하고, 두 값 모두 보고 범위(0.65~0.92) 상단에 있다(양호한 운용 조건
가정).

**(b) 딥러닝 ATR(자동표적인식)의 탐지율 상한**:

* 항공 적외선 ATR에서 Faster R-CNN이 난이도 높은 세트에서 FAR 0.01/deg²일 때
  Pd 80% (고전 QCF 기법은 45%) — Advanced ATR with IR Sensors (2021).
* YOLOatr: 열상 ATR 정확도 최대 99.6% (≤5,000 m) — arXiv:2507.11267.
* 위장 인원 데이터셋에서 MC-YOLOv5s recall 86.1%, mAP 94% — Sensors (2022),
  PMC9581598.
* MAH-YOLOv5 위장표적 미탐지율 8.53% (→ Pd≈0.91) — J. Image & Graphics (2024).

즉 현대 ATR의 운용 Pd 상한은 대략 0.80~0.99에 분포하며, η=0.86~0.90은 그
중앙~상단 값이다.

## 3. 위장 계수 (`CamouflageSpec`)

### 3.1 EO 대비 감쇠 (nominal ×0.75, heavy ×0.45)

* Vitoul et al. (2025): 비위장 박격포는 육안으로 **100% 탐지**, 자연 위장
  적용 시 은폐 효과 **약 90%**(탐지율 ~10%); 연구 전반에서 자연 위장이 탐지
  성공률을 **최대 35% 감소**. → "nominal 위장이 EO 신호를 수십 % 깎는다"는
  코드 가정(×0.75)은 보수적(과소) 쪽이며, heavy(×0.45)도 현장 보고 범위
  안이다.
* 다중분광 위장망 평가 능력 연구(A Multispectral Camouflage Net Evaluation
  Capability, 2024): EO 대역별 패턴 대비 측정 절차 — 대비 감쇠를 측정량으로
  다루는 방법론 근거.

### 3.2 IR 대비 감쇠 (nominal ×0.85, heavy ×0.65)

* **실험실 측정**: 다중분광 위장재가 기존 저방사 재질 대비 MWIR 복사 강도
  **53.4% 감소**(×0.47), LWIR **13.0% 감소**(×0.87) — Nature Communications
  12, 1805 (2021). LWIR 값(×0.87)은 코드의 nominal(×0.85)과 거의 일치한다.
* **열상 위장망 물성**: 차열 조건은 열적외 투과율 **< 25%**, 배경과의 복사
  온도차 **±4 K** 이내면 융합 — SPIE 10157 (2016), "Analysis and research on
  thermal infrared properties and adaptability of the camouflage net".
* **상용 체계 주장치**: 다중분광 위장체계의 열 신호 억제 **최대 75%**
  (×0.25 하한) — Talos Umbra MSC 제품 자료(벤더 주장, 참고용).
* **왜 IR 감쇠(0.85/0.65)를 EO(0.75/0.45)보다 약하게 두었나**: 위장망은
  시계·근적외 설계가 우선이고, 망 자체가 주간 가열되며 망눈(porosity)으로
  열복사가 직접 누설된다 — NATO RTO-MP-SCI-145, *Methods for Evaluating
  Thermal Camouflage* (DTIC ADA456649)의 평가 방법론과 패턴 특허들(US
  7,390,123 등)이 이 비대칭을 뒷받침한다. ΔT가 핵심 계량이라는 점도 동 보고서
  근거.

### 3.3 알고리즘 확률 계수 (nominal ×0.92, heavy ×0.75)

위장은 신호 대비만 깎는 게 아니라 탐지 알고리즘·관측자의 판단 자체를
저하시킨다:

* MAH-YOLOv5: 위장표적 미탐지율 8.53% → ×0.91 ≈ 코드 nominal ×0.92.
* MC-YOLOv5s: 위장 인원 recall 86.1%.
* CaReNN(위장 노출 CNN): mAP 70% — heavy ×0.75 수준의 하한 근거.
* Vitoul et al.: 위장·기만 표적은 최고 성능 장비(MOSKITO)로도 "완전하고
  정확한 식별이 일관되게 보장되지 않음".

## 4. 남는 한계 (논문에 그대로 쓸 것)

1. 위장 계수를 **대비 승수 형태로 직접 보고한 현장 실험은 없다**. 문헌값은
   복사 강도 감쇠(실험실), 탐지 성공률 변화(현장), 투과율(물성)로 흩어져
   있고, 본 모델은 이를 대비 승수로 번역한 것이다 — 따라서 계수는 여전히
   가정이며, 문헌은 범위의 타당성만 보증한다.
2. 벤더 주장치(Talos 75% 등)는 독립 검증이 없으므로 보조 근거로만 인용한다.
3. η는 데이터셋·거리·운용 조건에 민감하다(ATR 보고치 0.45~0.996). 민감도
   분석(또는 A/B 시나리오)으로 결론의 강건성을 보이는 것이 안전하다.

## 5. 인용 목록

1. Johnson, J. (1958). Analysis of image forming systems. *Proc. Image
   Intensifier Symposium*, US Army ERDL.
2. NVESD. *NVTherm — Night Vision Thermal Imaging Systems Performance Model,
   User's Manual*. §1.4 (TTPF, E=3.76), §2.8.4–2.8.7 (N50 표).
   <https://zieg.com/links/sensors/NVThermManual5.pdf>
3. Sjaardema, G., Smith, C. *History and Evolution of the Johnson Criteria*.
   Sandia Report. <https://www.osti.gov/servlets/purl/1222446>
4. Hixson, J., Jacobs, E., Vollmerhausen, R. (2004). Target detection cycle
   criteria when using the targeting task performance metric. *Proc. SPIE*
   5612. <https://ui.adsabs.harvard.edu/abs/2004SPIE.5612..275H/abstract>
5. US Patent 7,149,652. *Method for modeling detection of camouflaged
   targets*.
6. Vitoul, V., Ivan, J., Potužák, L., Šustr, M., Hanková, B. (2025).
   Experimental Evaluation of Camouflage Effectiveness Against Ground-Based
   Surveillance. *Proc. ICINCO 2025*, Vol. 1, pp. 425–434.
   DOI: 10.5220/0013712500003982.
   <https://www.scitepress.org/Papers/2025/137125/137125.pdf>
7. Zhu, H. et al. (2021). Multispectral camouflage for infrared, visible,
   lasers and microwave with radiative cooling. *Nature Communications* 12,
   1805. <https://www.nature.com/articles/s41467-021-22051-0>
8. SPIE 10157 (2016). Analysis and research on thermal infrared properties
   and adaptability of the camouflage net.
   <https://ui.adsabs.harvard.edu/abs/2016SPIE10157E..35C/abstract>
9. NATO RTO-MP-SCI-145 (2004). *Methods for Evaluating Thermal Camouflage*.
   DTIC ADA456649. <https://apps.dtic.mil/sti/tr/pdf/ADA456649.pdf>
10. A Multispectral Camouflage Net Evaluation Capability (2024).
    <https://www.researchgate.net/publication/384559285>
11. Advanced Automatic Target Recognition (ATR) with Infrared (IR) Sensors
    (2021). <https://www.researchgate.net/publication/350104472>
12. YOLOatr: Deep Learning Based Automatic Target Detection and Localization
    in Thermal Infrared Imagery. arXiv:2507.11267.
    <https://arxiv.org/pdf/2507.11267>
13. Research on Camouflaged Human Target Detection Based on Deep Learning
    (2022). *Sensors*. <https://pmc.ncbi.nlm.nih.gov/articles/PMC9581598/>
14. MAH-YOLOv5 — Detection of camouflage targets based on attention mechanism
    and multi-detection layer structure (2024). *J. Image and Graphics*.
    <https://www.cjig.cn/en/article/doi/10.11834/jig.221189/>
15. Talos Advanced Systems. Umbra MSC — Multispectral Camouflage System
    (벤더 자료). <https://talosadvancedsystems.com/technologies/umbra-msc>
