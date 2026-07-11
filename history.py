"""
점수 히스토리 — 매일 분석 결과 자동 저장 + 추세/검증 계산.

저장 위치: Gist (설정 시) + 로컬 data/score_history.json
형식:
  {
    "005930": [
      {
        "date": "2026-05-10",
        "total": 5,
        "close": 268500,
        "opinion": "분할 접근 가능",
        "scores": {"추세": 1, "모멘텀": 1, "거래량": 0, "수급": 1, ...}
      },
      ...
    ]
  }

`total`은 매매 목적 가중 종합점수.
`scores` 필드는 2026-05 이후 추가됨. 그 이전 엔트리에는 없을 수 있음.
같은 날 다시 분석하면 덮어씀 (마지막 분석값 유지).
365일 이전 데이터는 자동 제거 (검증/백테스트를 위해 1년치 보존).

배치 모드: 스크리닝처럼 N종목을 연속 분석할 때, 매번 Gist에 저장하면
API 호출이 폭주해서 느려지고 멈출 위험이 있음. begin_batch()로 시작하고
commit_batch()로 끝내면 그 사이 record_snapshot 호출은 메모리에만 쌓이고
끝에서 한 번만 Gist 저장.
"""
import threading
from datetime import datetime, timedelta

import cloud_store


FILENAME = "score_history.json"
ARCHIVE_FILENAME = "score_history_archive.json"
# 본 파일에 유지하는 기간. 넘긴 기록은 삭제하지 않고 아카이브 파일로 옮긴다
# (데이터 무손실 — 장기 walk-forward 검증에 계속 활용). 본 파일을 작게 유지하는
# 이유: score_history 가 Gist 전체 용량을 키워 다른 파일이 잘리던 전례.
RETENTION_DAYS = 365

# 과거 항목 이름 → 현재 이름. 점수 항목 이름이 바뀌면(예: "재무"→"재무 건전성")
# 옛 기록이 옛 이름으로 남아, 검증(IC/walk-forward)에서 같은 항목이 둘로 쪼개지고
# 옛 이름 쪽은 표본이 적어 예측력을 못 잰다. 읽는 시점에 현재 이름으로 통일한다.
SCORE_NAME_ALIASES = {"재무": "재무 건전성"}


def normalize_scores(scores: dict) -> dict:
    """scores dict의 옛 항목 이름을 현재 이름으로 통일. 같은 엔트리에 옛/새 이름이
    공존하면 현재 이름(새) 값을 보존한다. (history·발굴 검증이 공용으로 사용)"""
    if not isinstance(scores, dict):
        return scores
    if not any(k in SCORE_NAME_ALIASES for k in scores):
        return scores  # 옛 이름 없으면 그대로 (대다수 경로 — 비용 거의 0)
    out: dict = {}
    for k, v in scores.items():
        canon = SCORE_NAME_ALIASES.get(k, k)
        if canon in out and k != canon:
            continue  # 이미 현재 이름 값이 있으면 옛 이름은 버림
        out[canon] = v
    return out

_lock = threading.Lock()
_deferred_buffer: dict[str, list[dict]] | None = None
# 중첩/동시 batch 호출 시에도 안전하도록 reference count로 관리.
# begin_batch 가 두 번 호출되면 increment, commit_batch 도 두 번이면 0이 돼 저장.
# 동시 호출이라도 한쪽 commit이 다른 쪽 데이터를 잃지 않음.
_batch_depth = 0


def _load() -> dict[str, list[dict]]:
    data = cloud_store.load(FILENAME, {})
    if not isinstance(data, dict):
        return {}
    for entries in data.values():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if isinstance(e, dict) and isinstance(e.get("scores"), dict):
                e["scores"] = normalize_scores(e["scores"])
    return data


def _archive_aged(history: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """RETENTION_DAYS 를 넘긴 기록을 아카이브 파일로 옮기고(삭제 아님) 본체를 돌려준다.

    아카이브는 날짜 중복 없이 병합 — 같은 기록을 두 번 옮겨도 안전.
    옮길 게 없으면(대부분의 저장) 아카이브 파일을 건드리지 않는다.
    """
    cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    aged: dict[str, list[dict]] = {}
    kept: dict[str, list[dict]] = {}
    for code, entries in history.items():
        if not isinstance(entries, list):
            continue
        old = [e for e in entries if e.get("date", "") < cutoff]
        if old:
            aged[code] = old
        kept[code] = [e for e in entries if e.get("date", "") >= cutoff]
    if aged:
        try:
            archive = cloud_store.load(ARCHIVE_FILENAME, {})
            if not isinstance(archive, dict):
                archive = {}
            for code, old in aged.items():
                have = {e.get("date") for e in archive.get(code, []) if isinstance(e, dict)}
                merged = list(archive.get(code, [])) + [e for e in old if e.get("date") not in have]
                merged.sort(key=lambda e: e.get("date", ""))
                archive[code] = merged
            cloud_store.save(ARCHIVE_FILENAME, archive)
        except Exception:
            # 아카이브 실패 시 본체에서도 지우지 않음 — 데이터 유실 방지
            return history
    return kept


def _save(history: dict[str, list[dict]]) -> None:
    cloud_store.save(FILENAME, _archive_aged(history))


def begin_batch() -> None:
    """대량 분석 시작 — 이후 record_snapshot은 메모리 버퍼에만 누적.
    중첩/동시 호출 OK (reference count로 관리)."""
    global _deferred_buffer, _batch_depth
    with _lock:
        if _batch_depth == 0:
            _deferred_buffer = _load()
        _batch_depth += 1


def commit_batch() -> None:
    """배치 종료 — 가장 바깥 commit 시점에만 저장.
    중첩된 begin_batch의 모든 변경이 보존됨."""
    global _deferred_buffer, _batch_depth
    with _lock:
        if _batch_depth > 0:
            _batch_depth -= 1
        if _batch_depth == 0 and _deferred_buffer is not None:
            buf = _deferred_buffer
            _deferred_buffer = None
            _save(buf)


def discard_batch() -> None:
    """배치 취소 — 저장 없이 버퍼만 비움. depth도 0으로 강제 초기화."""
    global _deferred_buffer, _batch_depth
    with _lock:
        _deferred_buffer = None
        _batch_depth = 0


def record_snapshot(
    code: str,
    total: int,
    close: float,
    opinion: str,
    scores: list[dict] | None = None,
) -> None:
    """
    오늘 점수 저장 (같은 날 중복은 덮어쓰기, 365일 초과는 자동 제거).

    scores: analyzer.AnalysisResult["scores"] 그대로 전달. 항목별 검증을 위해
    {name: score} 형태로 압축 저장. None이면 저장 안 함 (구버전 호환).

    배치 모드 (begin_batch~commit_batch 사이): Gist 저장 안 함, 버퍼에만 추가.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    entry = {
        "date": today,
        "total": int(total),
        "close": float(close),
        "opinion": (opinion or "").split(" — ")[0],
    }
    if scores:
        entry["scores"] = {
            s["name"]: int(s["score"])
            for s in scores
            if isinstance(s, dict) and "name" in s and "score" in s
        }

    with _lock:
        is_deferred = _deferred_buffer is not None
        history = _deferred_buffer if is_deferred else _load()

        entries = [e for e in history.get(code, []) if e.get("date") != today]
        entries.append(entry)
        # 오래된 기록은 삭제하지 않는다 — _save 가 아카이브 파일로 옮김(무손실)
        entries.sort(key=lambda e: e.get("date", ""))

        history[code] = entries
        if not is_deferred:
            _save(history)


def get_history(code: str, days: int = 30) -> list[dict]:
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return [e for e in _load().get(code, []) if e.get("date", "") >= cutoff]


def load_all(include_archive: bool = False) -> dict[str, list[dict]]:
    """전체 종목의 히스토리 로드 — 검증/백테스트용.

    include_archive=True 면 아카이브(365일 초과분)까지 병합해 장기 검증 창을 넓힌다.
    """
    data = _load()
    if not include_archive:
        return data
    archive = cloud_store.load(ARCHIVE_FILENAME, {})
    if not isinstance(archive, dict):
        return data
    for code, old in archive.items():
        if not isinstance(old, list):
            continue
        for e in old:
            if isinstance(e, dict) and isinstance(e.get("scores"), dict):
                e["scores"] = normalize_scores(e["scores"])
        cur = data.get(code, [])
        have = {e.get("date") for e in cur if isinstance(e, dict)}
        merged = [e for e in old if isinstance(e, dict) and e.get("date") not in have] + cur
        merged.sort(key=lambda e: e.get("date", ""))
        data[code] = merged
    return data


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
