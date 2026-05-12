"""
📌 스크리닝 히스토리 — 매일 스크리닝 통과한 종목들을 자동 저장.

저장 위치: Gist (설정 시) + 로컬 data/screening_history.json
형식:
  {
    "2026-05-13": ["005930", "000660", "012450"],
    "2026-05-12": ["005930", "012450"],
    ...
  }

같은 날 여러 번 스크리닝 시 마지막 결과로 덮어씀.
RETENTION_DAYS(기본 90) 초과 자동 제거.

활용:
  - 매일 등장한 종목 자동 집계 → 사이드바에서 한눈에 추이 확인
  - 어제 등장했는데 오늘 안 보이는 종목 = "최근 탈락"으로 자동 감지
  - 별도의 "추적 추가" 버튼 없이 스크리닝만 돌리면 자동 누적
"""
from datetime import datetime, timedelta

import cloud_store


FILENAME = "screening_history.json"
RETENTION_DAYS = 90  # 3개월 — 분기 사이클 + 안정적 후보 패턴 파악


def _load() -> dict[str, list[str]]:
    data = cloud_store.load(FILENAME, {})
    if not isinstance(data, dict):
        return {}
    # 값이 list인 항목만 (안전)
    return {k: v for k, v in data.items() if isinstance(v, list)}


def _save(history: dict[str, list[str]]) -> None:
    cloud_store.save(FILENAME, history)


def record_today(codes: list[str]) -> None:
    """오늘 스크리닝 통과 종목 저장 (덮어쓰기). 90일 초과는 자동 제거."""
    today = datetime.now().strftime("%Y-%m-%d")
    history = _load()
    history[today] = list(codes)

    cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    history = {d: c for d, c in history.items() if d >= cutoff}

    _save(history)


def get_recent(days: int = 90) -> dict[str, dict]:
    """
    최근 N일 등장 종목 집계.

    Returns:
        {
            "005930": {
                "first_seen": "2026-05-10",
                "last_seen": "2026-05-13",
                "count": 4,
                "in_latest": True,  # 가장 최근 스크리닝에 등장했나
            },
            ...
        }
    """
    history = _load()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    filtered = {d: codes for d, codes in history.items() if d >= cutoff}
    if not filtered:
        return {}

    latest_date = max(filtered.keys())
    latest_codes = set(filtered.get(latest_date, []))

    aggregated: dict[str, dict] = {}
    for date in sorted(filtered.keys()):
        for code in filtered[date]:
            if code not in aggregated:
                aggregated[code] = {
                    "first_seen": date,
                    "last_seen": date,
                    "count": 1,
                    "in_latest": False,
                }
            else:
                aggregated[code]["last_seen"] = date
                aggregated[code]["count"] += 1

    # latest 등장 여부 마킹
    for code in aggregated:
        aggregated[code]["in_latest"] = code in latest_codes
        aggregated[code]["latest_date"] = latest_date

    return aggregated


def clear() -> None:
    _save({})
