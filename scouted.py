"""
📌 발굴 추적 — 스크리닝에서 발견한 종목들을 모아 매일 추이를 보는 기능 +
점수 기반 가상 매매 검증의 데이터 원천.

저장 위치: Gist (설정 시) + 로컬 data/scouted.json
형식:
  {
    "005930": {
      "added_at": "2026-05-13",       # 추적 시작일
      "added_score": 4,                # 추적 시작 시점 종합점수
      "added_close": 78500.0,          # 추적 시작 시점 종가 (가상 매수가) — 2026-05 이후 추가
      "added_scores": {                # 항목별 점수 스냅샷 — 2026-05 이후 추가
        "추세": 1, "모멘텀": 1, "거래량": 0, ...
      },
      "universe": "safe"               # 어느 유니버스에서 발굴됐는지
    },
    ...
  }

워치리스트·즐겨찾기와의 차이:
  - 워치리스트: 분석 대상 (수시로 바뀜)
  - 즐겨찾기: 관심 있는 종목 (수동, 점수 추세 모름)
  - 발굴 추적: 스크리닝에서 후보로 발굴된 것만, 점수 변화 추적, 가상 매매 PnL 계산
"""
from datetime import datetime

import cloud_store


FILENAME = "scouted.json"


def load_scouted() -> dict[str, dict]:
    data = cloud_store.load(FILENAME, {})
    return data if isinstance(data, dict) else {}


def save_scouted(items: dict[str, dict]) -> None:
    cloud_store.save(FILENAME, items)


def _compact_scores(scores: list[dict] | None) -> dict[str, int] | None:
    if not scores:
        return None
    return {
        s["name"]: int(s["score"])
        for s in scores
        if isinstance(s, dict) and "name" in s and "score" in s
    }


def add(
    code: str,
    score: int,
    universe: str = "safe",
    close: float | None = None,
    scores: list[dict] | None = None,
) -> bool:
    """추가됐으면 True, 이미 있으면 False.
    close: 발굴 시점 종가 (가상 매수가로 활용 → verifier가 수익률 계산)
    scores: analyzer.AnalysisResult["scores"] 그대로 (항목별 검증용)
    """
    s = load_scouted()
    if code in s:
        return False
    entry: dict = {
        "added_at": datetime.now().strftime("%Y-%m-%d"),
        "added_score": int(score),
        "universe": universe,
    }
    if close is not None:
        entry["added_close"] = float(close)
    compact = _compact_scores(scores)
    if compact:
        entry["added_scores"] = compact
    s[code] = entry
    save_scouted(s)
    return True


def add_from_analysis(result: dict, universe: str = "safe") -> bool:
    """analyzer.analyze() 결과 dict를 그대로 받아 발굴 등록.
    종합점수·종가·항목별 점수까지 한 번에 스냅샷.
    """
    code = result.get("code")
    if not code:
        return False
    return add(
        code=code,
        score=int(result.get("total", 0)),
        universe=universe,
        close=result.get("last_close"),
        scores=result.get("scores"),
    )


def add_many(items: list[tuple[str, int]], universe: str = "safe") -> int:
    """새로 추가된 개수 반환 (이미 있는 건 건너뜀).
    items: [(code, score), ...] — 종가·항목별 점수 없이 간단 등록.
    풍부한 스냅샷이 필요하면 add_from_analysis()를 종목별로 호출하세요.
    """
    s = load_scouted()
    today = datetime.now().strftime("%Y-%m-%d")
    added = 0
    for code, score in items:
        if code not in s:
            s[code] = {
                "added_at": today,
                "added_score": int(score),
                "universe": universe,
            }
            added += 1
    if added > 0:
        save_scouted(s)
    return added


def remove(code: str) -> None:
    s = load_scouted()
    if code in s:
        del s[code]
        save_scouted(s)


def clear_all() -> None:
    save_scouted({})
