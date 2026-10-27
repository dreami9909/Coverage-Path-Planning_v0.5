"""Input -> Model -> Solver -> Output 파이프라인.

시나리오 파일(TOML) 하나가 실험 하나다. 파일이 입력 전부를 선언하고,
실행 결과는 ``results/runs/<이름>/`` 에 입력 스냅샷·지표·그림으로 남는다.
그대로 다시 돌리면 재현된다.

실험 챕터(``experiments/``)와 같은 구축·채점 함수를 쓴다 — 조건이 조용히
갈리는 것을 막기 위해 이 패키지는 새 모형 코드를 만들지 않는다.
"""

from cpp_search.pipeline.scenario import Scenario, load_scenario

__all__ = ["Scenario", "load_scenario"]
