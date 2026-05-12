"""
점수 히스토리 — 매일 분석 결과 자동 저장 + 30일 추세 계산.

저장 위치: Gist (설정 시) + 로컬 data/score_history.json
형식:
  {
    "005930": [
      {"date": "2026-05-10", "total": 5, "close": 268500, "opinion": "분할 접근 가능"},
      ...
    ]
  }

같은 날 다시 분석하면 덮어씀 (마지막 분석값 유지).
90일 이전 데이터는 자동 제거.
"""
from datetime import datetime, timedelta

import cloud_store


FILENAME = "score_history.json"
RETENTION_DAYS = 90


def _load() -> dict[str, list[dict]]:
    data = cloud_store.load(FILENAME, {})
    return data if isinstance(data, dict) else {}


def _save(history: dict[str, list[dict]]) -> None:
    cloud_store.save(FILENAME, history)


def record_snapshot(code: str, total: int, close: float, opinion: str) -> None:
    """오늘 점수 저장 (같은 날 중복은 덮어쓰기, 90일 초과는 자동 제거)."""
    today = datetime.now().strftime("%Y-%m-%d")
    history = _load()
    entries = [e for e in history.get(code, []) if e.get("date") != today]
    entries.append({
        "date": today,
        "total": int(total),
        "close": float(close),
        "opinion": (opinion or "").split(" — ")[0],
    })

    cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    entries = [e for e in entries if e.get("date", "") >= cutoff]
    entries.sort(key=lambda e: e.get("date", ""))

    history[code] = entries
    _save(history)


def get_history(code: str, days: int = 30) -> list[dict]:
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return [e for e in _load().get(code, []) if e.get("date", "") >= cutoff]


def compute_trend(code: str, days: int = 30) -> dict | None:
    """최근 N일 점수 변화 요약. 데이터 2건 미만이면 None."""
    history = get_history(code, days=days)
    if len(history) < 2:
        return None

    first = history[0]
    last = history[-1]
    delta = last["total"] - first["total"]
    if delta > 0:
        label = f"↗ {first['date']}부터 {delta:+d}점 상승"
    elif delta < 0:
        label = f"↘ {first['date']}부터 {delta:+d}점 하락"
    else:
        label = f"→ {first['date']}부터 변화 없음"
    return {
        "days_recorded": len(history),
        "first_score": first["total"],
        "last_score": last["total"],
        "delta": delta,
        "label": label,
    }
