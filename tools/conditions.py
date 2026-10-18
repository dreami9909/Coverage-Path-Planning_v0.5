"""도구들이 함께 쓰는 **조건 목록과 항적 규약 이름**.

조건마다 저장된 절단평면 계획 파일이 하나 있다. 이 목록을 도구마다 따로 적어 두면
한쪽만 늘어난 채로 표가 갈린다 (표 1 은 다섯 조건, 2단계는 여덟 조건이던 때가 있었다).

항적 규약은 둘을 나란히 보고한다 (``core/searcher_track``, 2026-10-08 확정).

    centre-sync   본 표. 셀 중심까지 간 뒤 중심에서 회로를 시작한다 (시간 계산은 맞춘다).
                  셀 모형이 가정한 경로 그대로라 인증 판정·학습 비교에 쓴다.
    nearest       참고. 다음 셀의 가장 가까운 차선 끝으로 들어가 거기서 회로를 시작한다.
                  실제 비행에 가까워 PD 가 더 나오지만 출발점에 민감하다.

둘 다 "계획이 정하지 않은 자유도"를 평가기가 채운 것이므로 어느 쪽도 정답이 아니다.
결과 파일은 본 표 규약이 이름 그대로, 참고 규약이 ``-nearest`` 를 붙여 저장된다.
출발점은 두 규약 모두 실제 발진 지점이고, 표적은 셀 모형과 같은 간격으로 전개한다
(``planning/continuous_mc.evaluate`` 의 기본값).
"""
import json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: (케이스, 격자 한 변, 저장된 절단평면 계획). 앞 다섯은 인증이 닫힌 격자다. 12 x 12
#: 부터는 닫히지 않았고, 거기서 "절단평면"은 최적해가 아니라 그 예산(master 300 s x
#: 20 회)에서 얻은 incumbent 다. TEL 14 x 14 는 격자 훑기 때 푼 계획을 그대로 쓴다 —
#: 같은 인스턴스임을 fingerprint 와 재채점 값으로 확인했다.
CONDITIONS = [("Tank", 4, "anchor-Tank-4-20260201.json"),
              ("Tank", 6, "anchor-Tank-6-20260201.json"),
              ("Tank", 8, "occupancy-1.json"),
              ("TEL", 10, "anchor-TEL-10-20260201.json"),
              ("Tank", 10, "scale-Tank-10-300x60.json"),
              ("Tank", 12, "anchor-Tank-12-20260201.json"),
              ("TEL", 12, "anchor-TEL-12-20260201.json"),
              ("Tank", 14, "anchor-Tank-14-20260201.json"),
              ("TEL", 14, "sweep-TEL-14.json"),
              ("TEL", 16, "anchor-TEL-16-20260201.json"),
              ("Tank", 16, "anchor-Tank-16-20260201.json")]


def available_conditions():
    """저장된 계획 파일이 실제로 있는 조건만 (풀이가 아직 안 끝난 격자를 건너뛴다)."""
    return [c for c in CONDITIONS if (ROOT / "results/studies" / c[2]).exists()]

#: 규약 -> (사람이 읽는 이름, 결과 파일 접미어). 앞이 본 표, 뒤가 참고.
CONVENTIONS = {"centre-sync": ("중심 경유", ""), "nearest": ("최근접 진입", "-nearest")}
BASELINE = "centre-sync"
REFERENCE = "nearest"


def convention_label(entry: str) -> str:
    return CONVENTIONS[entry][0]


def convention_suffix(entry: str) -> str:
    """결과 파일 이름에 붙는 접미어. 본 표 규약은 빈 문자열이다."""
    if entry not in CONVENTIONS:
        raise SystemExit(f"항적 규약은 {sorted(CONVENTIONS)} 중 하나다: {entry!r}")
    return CONVENTIONS[entry][1]


def stage2_path(case: str, n: int, entry: str = BASELINE, variant: str = "") -> pathlib.Path:
    """그 규약의 보상으로 학습한 2단계 결과 파일.

    ``variant`` 는 본 실행과 설정이 다른 곁가지 실행의 이름표다 (예: 학습 반복 수를
    늘린 ``it400``). 이름이 달라 ``stage2_results`` 에는 잡히지 않는다.
    """
    extra = f"-{variant}" if variant else ""
    return ROOT / f"results/studies/mappo2{convention_suffix(entry)}{extra}-{case}-{n}.json"


def stage2_results(entry: str = BASELINE, wanted=None):
    """그 규약의 2단계 결과를 조건 순서대로 (태그, 내용) 으로 돌려준다."""
    out = []
    for case, n, _ in CONDITIONS:
        tag = f"{case}-{n}"
        path = stage2_path(case, n, entry)
        if (wanted and tag not in wanted) or not path.exists():
            continue
        saved = json.load(open(path))
        if saved.get("entry") != entry:
            raise SystemExit(f"{path.name}: 항적 규약 {saved.get('entry')!r} 로 만든 파일이다 "
                             f"({entry!r} 가 아니다) — 다시 학습해야 한다")
        out.append((tag, saved))
    return out
