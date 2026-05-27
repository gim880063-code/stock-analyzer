"""
GitHub Gist를 영구 저장소로 사용.

Streamlit Cloud의 filesystem은 재배포 시 초기화되므로, 포트폴리오·점수 히스토리·
즐겨찾기·워치리스트 같은 사용자 상태를 Gist에 동기화해서 보존한다.

설정:
  - GITHUB_PAT (Personal Access Token, gist 권한)
  - GIST_ID (사용할 비공개 Gist의 ID)
  → .env 또는 st.secrets에 등록

Gist 미설정 시 로컬 파일로 fallback (개발 환경 등).
"""
import json
import os
import threading
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


ENV_PATH = Path(__file__).parent / ".env"
DATA_DIR = Path(__file__).parent / "data"
GIST_API = "https://api.github.com/gists"

_lock = threading.Lock()
_gist_cache: dict[str, str] | None = None
_sync_log: list[str] = []  # 최근 동기화 결과 (성공·실패 모두) — 진단용


def _log(msg: str) -> None:
    _sync_log.append(f"{datetime_now()} {msg}")
    if len(_sync_log) > 20:
        _sync_log[:] = _sync_log[-20:]


def datetime_now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


def get_sync_log() -> list[str]:
    return list(_sync_log)


def _get_credentials() -> tuple[str, str]:
    """워커 스레드 대응으로 한 번 읽으면 os.environ 에 캐싱 (dart/llm 과 동일 패턴)."""
    # 1) 이미 env에 있으면 그대로 (워커 스레드 fast path)
    cached_pat = os.environ.get("GITHUB_PAT", "").strip()
    cached_gid = os.environ.get("GIST_ID", "").strip()
    if cached_pat and cached_gid:
        return cached_pat, cached_gid
    # 2) Streamlit Cloud secrets (배포 환경, 메인 스레드)
    try:
        import streamlit as st
        pat = str(st.secrets.get("GITHUB_PAT", "")).strip()
        gist_id = str(st.secrets.get("GIST_ID", "")).strip()
        if pat and gist_id:
            os.environ["GITHUB_PAT"] = pat   # 워커 스레드용 캐시
            os.environ["GIST_ID"] = gist_id
            return pat, gist_id
    except Exception:
        pass
    # 3) .env (로컬 환경)
    load_dotenv(ENV_PATH, override=True)
    return (
        os.environ.get("GITHUB_PAT", "").strip(),
        os.environ.get("GIST_ID", "").strip(),
    )


def is_configured() -> bool:
    pat, gist_id = _get_credentials()
    return bool(pat and gist_id)


def _fetch_gist_files() -> dict[str, str]:
    """Gist의 모든 파일 내용을 가져옴. 세션당 1회만 fetch (cache)."""
    global _gist_cache
    with _lock:
        if _gist_cache is not None:
            return _gist_cache
        pat, gist_id = _get_credentials()
        if not (pat and gist_id):
            _gist_cache = {}
            return _gist_cache
        try:
            r = requests.get(
                f"{GIST_API}/{gist_id}",
                headers={
                    "Authorization": f"token {pat}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=10,
            )
            if not r.ok:
                _gist_cache = {}
                return _gist_cache
            files = r.json().get("files", {})
            _gist_cache = {name: f.get("content", "") for name, f in files.items()}
            return _gist_cache
        except Exception:
            _gist_cache = {}
            return _gist_cache


def _invalidate_cache():
    global _gist_cache
    with _lock:
        _gist_cache = None


def _remember_cached_file(filename: str, content: str) -> None:
    """Keep current-session reads consistent when a later Gist PATCH fails."""
    global _gist_cache
    with _lock:
        if _gist_cache is not None:
            _gist_cache = {**_gist_cache, filename: content}


def load(filename: str, default: Any) -> Any:
    """
    JSON 파일 로드. Gist 우선, 실패 시 로컬, 그것도 없으면 default.
    """
    if is_configured():
        files = _fetch_gist_files()
        if filename in files:
            content = (files[filename] or "").strip()
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
    # 로컬 fallback
    local = DATA_DIR / filename
    if local.exists():
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return default


def save(filename: str, data: Any) -> None:
    """
    JSON 저장. 로컬에 항상 저장하고, Gist 설정돼 있으면 그곳에도 동기화.
    Gist 호출 실패해도 로컬은 항상 성공.
    """
    content = json.dumps(data, ensure_ascii=False, indent=2)

    # 1) 로컬 (항상)
    try:
        DATA_DIR.mkdir(exist_ok=True)
        (DATA_DIR / filename).write_text(content, encoding="utf-8")
        _remember_cached_file(filename, content)
    except OSError:
        pass

    # 2) Gist (설정 시)
    pat, gist_id = _get_credentials()
    if not (pat and gist_id):
        _log(f"⏭️ {filename} — Gist 미설정, 로컬만 저장")
        return
    try:
        r = requests.patch(
            f"{GIST_API}/{gist_id}",
            headers={
                "Authorization": f"token {pat}",
                "Accept": "application/vnd.github+json",
            },
            json={"files": {filename: {"content": content}}},
            timeout=10,
        )
        if r.ok:
            _log(f"✅ {filename} → Gist ({len(content)} bytes)")
            _invalidate_cache()
        else:
            _log(f"❌ {filename} HTTP {r.status_code}: {r.text[:150]}")
            _remember_cached_file(filename, content)
    except Exception as e:
        _log(f"❌ {filename} 예외 {type(e).__name__}: {str(e)[:150]}")
        _remember_cached_file(filename, content)
