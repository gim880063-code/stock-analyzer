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

배치 모드: 스크리닝처럼 N종목을 연속 분석할 때, 매번 Gist에 저장하면
API 호출이 폭주해서 느려지고 멈출 위험이 있음. begin_batch()로 시작하고
commit_batch()로 끝내면 그 사이 record_snapshot 호출은 메모리에만 쌓이고
끝에서 한 번만 Gist 저장.
"""
import threading
from datetime import datetime, timedelta

import cloud_store


FILENAME = "score_history.json"
RETENTION_DAYS = 90

_lock = threading.Lock()
_deferred_buffer: dict[str, list[dict]] | None = None


def _load() -> dict[str, list[dict]]:
    data = cloud_store.load(FILENAME, {})
    return data if isinstance(data, dict) else {}


def _save(history: dict[str, list[dict]]) -> None:
    cloud_store.save(FILENAME, history)


def begin_batch() -> None:
    """대량 분석 시작 — 이후 record_snapshot은 메모리 버퍼에만 누적."""
    global _deferred_buffer
    with _lock:
        _deferred_buffer = _load()


def commit_batch() -> None:
    """배치 종료 — 누적된 변경 사항을 한 번에 저장."""
    global _deferred_buffer
    with _lock:
        if _deferred_buffer is not None:
            buf = _deferred_buffer
            _deferred_buffer = None
            _save(buf)


def discard_batch() -> None:
    """배치 취소 — 저장 없이 버퍼만 비움."""
    global _deferred_buffer
    with _lock:
        _deferred_buffer = None


def record_snapshot(code: str, total: int, close: float, opinion: str) -> None:
    """
    오늘 점수 저장 (같은 날 중복은 덮어쓰기, 90일 초과는 자동 제거).

    배치 모드 (begin_batch~commit_batch 사이): Gist 저장 안 함, 버퍼에만 추가.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    with _lock:
        is_deferred = _deferred_buffer is not None
        history = _deferred_buffer if is_deferred else _load()

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
        if not is_deferred:
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
