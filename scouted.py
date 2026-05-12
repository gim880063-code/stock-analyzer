"""
📌 발굴 추적 — 스크리닝에서 발견한 종목들을 모아 매일 추이를 보는 기능.

저장 위치: Gist (설정 시) + 로컬 data/scouted.json
형식:
  {
    "005930": {
      "added_at": "2026-05-13",     # 추적 시작일
      "added_score": 4,              # 추적 시작 시점 점수
      "universe": "safe"             # 어느 유니버스에서 발굴됐는지
    },
    ...
  }

워치리스트·즐겨찾기와의 차이:
  - 워치리스트: 분석 대상 (수시로 바뀜)
  - 즐겨찾기: 관심 있는 종목 (수동, 점수 추세 모름)
  - 발굴 추적: 스크리닝에서 후보로 발굴된 것만, 점수 변화 추적, 유니버스 이탈 감지
"""
from datetime import datetime

import cloud_store


FILENAME = "scouted.json"


def load_scouted() -> dict[str, dict]:
    data = cloud_store.load(FILENAME, {})
    return data if isinstance(data, dict) else {}


def save_scouted(items: dict[str, dict]) -> None:
    cloud_store.save(FILENAME, items)


def add(code: str, score: int, universe: str = "safe") -> bool:
    """추가됐으면 True, 이미 있으면 False."""
    s = load_scouted()
    if code in s:
        return False
    s[code] = {
        "added_at": datetime.now().strftime("%Y-%m-%d"),
        "added_score": int(score),
        "universe": universe,
    }
    save_scouted(s)
    return True


def add_many(items: list[tuple[str, int]], universe: str = "safe") -> int:
    """새로 추가된 개수 반환 (이미 있는 건 건너뜀)."""
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
