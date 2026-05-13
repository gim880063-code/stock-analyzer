"""
스크리닝 히스토리 — 매일 스크리닝 통과/탈락 종목과 이유를 자동 저장.

저장 위치: Gist 설정 시 Gist + 로컬 data/screening_history.json

새 저장 형식:
{
  "2026-05-13": {
    "codes": ["005930", "000660"],
    "passed": {
      "005930": {
        "reason": "종합점수 +4점. 긍정: 추세(+1), 재무 건전성(+1)",
        "total": 4,
        "opinion": "분할 접근 가능"
      }
    },
    "dropped": {
      "035420": {
        "reason": "종합점수 +1점이 기준 +3점 미만",
        "total": 1
      }
    },
    "params": {"min_score": 3, "universe": "safe"}
  }
}

이전 저장 형식({"2026-05-13": ["005930", ...]})도 자동으로 읽을 수 있게 유지합니다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import cloud_store

FILENAME = "screening_history.json"
RETENTION_DAYS = 90


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _normalize_day_value(value: Any) -> dict:
    """과거 list 형식과 새 dict 형식을 모두 표준 dict로 변환."""
    if isinstance(value, list):
        codes = [str(c) for c in value if isinstance(c, (str, int))]
        return {
            "codes": codes,
            "passed": {c: {"reason": "스크리닝 통과", "total": None, "opinion": ""} for c in codes},
            "dropped": {},
            "params": {},
        }

    if isinstance(value, dict):
        raw_codes = value.get("codes", [])
        if not isinstance(raw_codes, list):
            raw_codes = []
        codes = [str(c) for c in raw_codes if isinstance(c, (str, int))]

        passed = value.get("passed", {})
        if not isinstance(passed, dict):
            passed = {}
        dropped = value.get("dropped", {})
        if not isinstance(dropped, dict):
            dropped = {}
        params = value.get("params", {})
        if not isinstance(params, dict):
            params = {}

        # codes가 비어 있는데 passed만 있는 경우도 복구
        if not codes and passed:
            codes = [str(c) for c in passed.keys()]

        return {
            "codes": codes,
            "passed": passed,
            "dropped": dropped,
            "params": params,
        }

    return {"codes": [], "passed": {}, "dropped": {}, "params": {}}


def _load() -> dict[str, dict]:
    data = cloud_store.load(FILENAME, {})
    if not isinstance(data, dict):
        return {}
    normalized: dict[str, dict] = {}
    for date, value in data.items():
        if isinstance(date, str):
            normalized[date] = _normalize_day_value(value)
    return normalized


def _save(history: dict[str, dict]) -> None:
    cloud_store.save(FILENAME, history)


def _prune(history: dict[str, dict]) -> dict[str, dict]:
    cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    return {d: v for d, v in history.items() if d >= cutoff}


def record_today(codes: list[str]) -> None:
    """오늘 스크리닝 통과 종목 저장. 기존 app.py와의 호환용."""
    today = _today()
    clean_codes = [str(c) for c in codes]
    history = _load()
    history[today] = {
        "codes": clean_codes,
        "passed": {
            c: {
                "reason": "스크리닝 최종 통과",
                "total": None,
                "opinion": "",
            }
            for c in clean_codes
        },
        "dropped": {},
        "params": {},
    }
    _save(_prune(history))


def record_today_details(
    passed_results: list[dict],
    dropped: list[dict] | None = None,
    min_score: int | None = None,
    universe: str | None = None,
) -> None:
    """오늘 스크리닝 결과를 통과/탈락 이유까지 저장."""
    today = _today()
    dropped = dropped or []

    codes: list[str] = []
    passed_meta: dict[str, dict] = {}
    for r in passed_results:
        code = str(r.get("code", "")).strip()
        if not code:
            continue
        codes.append(code)
        passed_meta[code] = {
            "name": r.get("name", code),
            "reason": r.get("_screen_reason") or r.get("screen_reason") or "스크리닝 최종 통과",
            "total": r.get("total"),
            "opinion": (r.get("opinion") or "").split(" — ")[0],
            "close": r.get("last_close"),
        }

    dropped_meta: dict[str, dict] = {}
    for item in dropped:
        code = str(item.get("code", "")).strip()
        if not code:
            continue
        dropped_meta[code] = {
            "name": item.get("name", code),
            "reason": item.get("reason") or "스크리닝 최종 통과 목록에서 제외",
            "total": item.get("total"),
            "close": item.get("close"),
        }

    history = _load()
    history[today] = {
        "codes": codes,
        "passed": passed_meta,
        "dropped": dropped_meta,
        "params": {
            "min_score": min_score,
            "universe": universe,
        },
    }
    _save(_prune(history))


def get_recent(days: int = 90) -> dict[str, dict]:
    """
    최근 N일 등장 종목 집계.

    Returns:
    {
      "005930": {
        "first_seen": "2026-05-10",
        "last_seen": "2026-05-13",
        "count": 4,
        "in_latest": True,
        "latest_date": "2026-05-13",
        "reason": "최근 통과 이유",
        "drop_reason": "최근 탈락 이유",
      }
    }
    """
    history = _load()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    filtered = {d: v for d, v in history.items() if d >= cutoff}
    if not filtered:
        return {}

    latest_date = max(filtered.keys())
    latest_day = _normalize_day_value(filtered.get(latest_date, {}))
    latest_codes = set(latest_day.get("codes", []))
    latest_dropped = latest_day.get("dropped", {}) or {}
    latest_params = latest_day.get("params", {}) or {}

    aggregated: dict[str, dict] = {}
    for date in sorted(filtered.keys()):
        day = _normalize_day_value(filtered[date])
        codes = day.get("codes", [])
        passed = day.get("passed", {}) or {}
        for code in codes:
            pmeta = passed.get(code, {}) if isinstance(passed, dict) else {}
            if code not in aggregated:
                aggregated[code] = {
                    "first_seen": date,
                    "last_seen": date,
                    "count": 1,
                    "in_latest": False,
                    "latest_date": latest_date,
                    "reason": pmeta.get("reason", ""),
                    "last_reason": pmeta.get("reason", ""),
                    "last_total": pmeta.get("total"),
                    "last_opinion": pmeta.get("opinion", ""),
                    "latest_params": latest_params,
                }
            else:
                aggregated[code]["last_seen"] = date
                aggregated[code]["count"] += 1
                aggregated[code]["last_reason"] = pmeta.get("reason", aggregated[code].get("last_reason", ""))
                aggregated[code]["last_total"] = pmeta.get("total", aggregated[code].get("last_total"))
                aggregated[code]["last_opinion"] = pmeta.get("opinion", aggregated[code].get("last_opinion", ""))

    # latest 등장 여부와 통과/탈락 이유 마킹
    for code, meta in aggregated.items():
        meta["in_latest"] = code in latest_codes
        meta["latest_date"] = latest_date
        meta["latest_params"] = latest_params
        if code in latest_codes:
            latest_passed = latest_day.get("passed", {}) or {}
            pmeta = latest_passed.get(code, {}) if isinstance(latest_passed, dict) else {}
            meta["reason"] = pmeta.get("reason") or meta.get("last_reason") or "최근 스크리닝 통과"
            meta["last_total"] = pmeta.get("total", meta.get("last_total"))
            meta["last_opinion"] = pmeta.get("opinion", meta.get("last_opinion", ""))
        else:
            dmeta = latest_dropped.get(code, {}) if isinstance(latest_dropped, dict) else {}
            meta["drop_reason"] = dmeta.get("reason") or "최근 스크리닝 최종 통과 목록에 없어 탈락으로 표시됨"
            meta["drop_total"] = dmeta.get("total")

    # 최신 날짜에 탈락 기록만 있고 과거 통과 기록이 없는 종목도 보존하고 싶으면 여기서 추가 가능하지만,
    # 현재 화면은 '과거 후보가 오늘 왜 빠졌는지' 추적하는 목적이라 통과 이력이 있는 종목만 반환한다.
    return aggregated


def clear() -> None:
    _save({})
