"""
DART OPEN API 클라이언트
- 종목코드(6자리) → DART 고유번호(corp_code, 8자리) 매핑 캐시
- 최신 사업/분기보고서 기반 재무지표 조회 (ROE, 부채비율, 성장률 등)
- PER/PBR/EPS/BPS 계산 (재무제표 + 발행주식수 기반)
"""
import io
import json
import os
import re
import threading
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv


ENV_PATH = Path(__file__).parent / ".env"
DATA_DIR = Path(__file__).parent / "data"


def _get_api_key() -> str:
    """API 키 조회 — Streamlit Cloud secrets 우선, 로컬 .env 백업.

    워커 스레드 대응: 한 번이라도 st.secrets 에서 읽으면 os.environ 에 캐싱.
    Streamlit 워커 스레드는 st.secrets 접근이 실패할 수 있는데, env var는 프로세스
    전역이라 어느 스레드에서나 읽힘.
    """
    # 1) 이미 env에 있으면 그대로 사용 (워커 스레드 fast path + secrets fallback)
    cached = os.environ.get("DART_API_KEY", "").strip()
    if cached:
        return cached
    # 2) Streamlit secrets (배포 환경, 보통 메인 스레드)
    try:
        import streamlit as st
        if "DART_API_KEY" in st.secrets:
            key = str(st.secrets["DART_API_KEY"]).strip()
            if key:
                os.environ["DART_API_KEY"] = key  # 워커 스레드용 캐시
                return key
    except Exception:
        pass
    # 3) 로컬 .env (개발 환경)
    load_dotenv(ENV_PATH, override=True)
    return os.environ.get("DART_API_KEY", "").strip()
CORP_CODE_CACHE = DATA_DIR / "corp_codes.json"

BASE_URL = "https://opendart.fss.or.kr/api"


class DartError(Exception):
    pass


def is_configured() -> bool:
    return bool(_get_api_key())


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def _download_corp_codes() -> dict[str, str]:
    """DART에서 전체 회사 목록(corpCode.xml ZIP) 다운로드 후 stock_code → corp_code 매핑 추출"""
    key = _get_api_key()
    if not key:
        raise DartError("DART_API_KEY가 설정되지 않았습니다. .env 파일에 키를 입력하세요.")

    resp = requests.get(
        f"{BASE_URL}/corpCode.xml",
        params={"crtfc_key": key},
        timeout=30,
    )
    resp.raise_for_status()

    if resp.headers.get("Content-Type", "").startswith("application/json"):
        msg = resp.json().get("message", resp.text)
        raise DartError(f"DART 응답 오류: {msg}")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("CORPCODE.xml") as f:
            tree = ET.parse(f)

    mapping: dict[str, str] = {}
    for item in tree.getroot().findall("list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        if stock_code and corp_code:
            mapping[stock_code] = corp_code
    return mapping


# corp_code 매핑은 한 번 받으면 거의 안 바뀌어서 프로세스 수명 동안 메모리에 캐시.
# 동시 다운로드 방지용 락 — 스크리닝 첫 실행 시 워커 3개가 동시에 ZIP 받는 것 방지.
_corp_code_lock = threading.Lock()
_corp_code_mapping: dict[str, str] | None = None


def get_corp_code(stock_code: str) -> str:
    """6자리 종목코드를 DART 8자리 corp_code로 변환.
    메모리 캐시 → 파일 캐시 → DART 다운로드 순. 동시 호출 시 다운로드는 1회만."""
    global _corp_code_mapping
    _ensure_data_dir()

    # 1) 메모리 캐시 (lock-free fast path)
    mapping = _corp_code_mapping
    if mapping is not None and stock_code in mapping:
        return mapping[stock_code]

    with _corp_code_lock:
        # 2) double-check — 다른 스레드가 락 대기 중에 채워놨을 수 있음
        if _corp_code_mapping is not None and stock_code in _corp_code_mapping:
            return _corp_code_mapping[stock_code]

        # 3) 메모리 캐시 비어있으면 파일 캐시 먼저
        if _corp_code_mapping is None:
            if CORP_CODE_CACHE.exists():
                try:
                    _corp_code_mapping = json.loads(CORP_CODE_CACHE.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    _corp_code_mapping = {}
            else:
                _corp_code_mapping = {}

        # 4) 파일에도 없으면 다운로드
        if stock_code not in _corp_code_mapping:
            _corp_code_mapping = _download_corp_codes()
            CORP_CODE_CACHE.write_text(
                json.dumps(_corp_code_mapping, ensure_ascii=False), encoding="utf-8"
            )

        if stock_code not in _corp_code_mapping:
            raise DartError(f"DART에 등록된 회사가 아닙니다: {stock_code}")
        return _corp_code_mapping[stock_code]


def _api_get(path: str, params: dict[str, Any], max_retries: int = 2) -> dict:
    key = _get_api_key()
    if not key:
        raise DartError("DART_API_KEY가 설정되지 않았습니다.")
    params = {"crtfc_key": key, **params}
    import time
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(f"{BASE_URL}/{path}", params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status not in ("000", "013"):
                raise DartError(f"DART {path} 오류 [{status}]: {data.get('message')}")
            return data
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(3 + attempt * 2)  # 3s, 5s
                continue
            raise DartError(f"DART {path} 연결 실패 (재시도 {max_retries}회 후)") from e
        except DartError:
            raise
    # unreachable
    if last_exc:
        raise last_exc
    raise DartError(f"DART {path} 실패 (원인 불명)")


def _latest_report_codes() -> list[tuple[int, str]]:
    """
    가장 최근 보고서부터 fallback 순서로 (사업연도, 보고서코드).
    분기보고서가 사업보고서보다 더 최신이면 우선 사용.

    공시 마감일:
      - 사업보고서(11011): 사업연도 종료 후 90일 (대부분 3월 말)
      - 1분기보고서(11013): 5월 15일
      - 반기보고서(11012): 8월 14일
      - 3분기보고서(11014): 11월 14일
    """
    now = datetime.now()
    year = now.year
    candidates: list[tuple[int, str]] = []

    # 1. 당해 분기 보고서 — 최신 분기 우선
    if now.month >= 11:
        candidates.append((year, "11014"))   # 3Q
    if now.month >= 8:
        candidates.append((year, "11012"))   # 반기
    if now.month >= 5:
        candidates.append((year, "11013"))   # 1Q (5/15 마감 — 일부 종목은 좀 더 일찍)

    # 2. 직전 연도 사업보고서 (3월 말 공시 — 가장 안정적인 baseline)
    candidates.append((year - 1, "11011"))

    # 3. 직전 연도 분기들 (그래도 데이터 없으면)
    candidates.append((year - 1, "11014"))
    candidates.append((year - 1, "11012"))
    candidates.append((year - 1, "11013"))

    # 4. 2년 전 사업보고서 (최후)
    candidates.append((year - 2, "11011"))
    return candidates


# 마지막으로 발생한 DART API 오류 — get_financials 가 모든 fallback 후보를 시도했는데
# 빈 결과만 나왔을 때 진짜 원인이 무엇인지 (네트워크/키 무효/rate limit 등) 알기 위해 보존.
# 데이터 없음 (status 013) 은 정상 fallback 흐름이라 기록 안 함.
_last_acnt_error: str | None = None


def _get_last_acnt_error() -> str | None:
    """가장 최근 _fetch_acnt_all 호출에서의 진짜 오류 (013/데이터없음 제외).
    analyzer 가 fin_report_label=None 일 때 진단 메시지로 노출."""
    return _last_acnt_error


def _fetch_acnt_all(corp_code: str, year: int, reprt_code: str) -> list[dict]:
    """단일회사 전체 재무제표"""
    global _last_acnt_error
    try:
        data = _api_get("fnlttSinglAcntAll.json", {
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": reprt_code,
            "fs_div": "CFS",  # 연결재무제표 우선
        })
    except DartError as e:
        # 데이터 없음 (013) 은 _api_get 이 raise 안 함. 여기서 잡히는 건 진짜 오류.
        _last_acnt_error = str(e)
        return []
    if data.get("status") == "013":
        # 데이터 없음 → 별도재무제표로 재시도
        try:
            data = _api_get("fnlttSinglAcntAll.json", {
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": reprt_code,
                "fs_div": "OFS",
            })
        except DartError as e:
            _last_acnt_error = str(e)
            return []
    return data.get("list", [])


def _to_int(v) -> int | None:
    if v is None:
        return None
    s = str(v).replace(",", "").strip()
    if not s or s == "-":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _find_amount(rows: list[dict], account_names: list[str], current: bool = True) -> int | None:
    """
    재무제표 행에서 계정과목의 금액 찾기.
    1단계: 정확 일치 (예: "매출액" == "매출액")
    2단계: 부분 일치 fallback (예: "영업수익" in "기타영업수익")
    이렇게 안 하면 "기타영업수익" 같은 부수 항목이 "매출액"보다 먼저 매칭됨.
    """
    field = "thstrm_amount" if current else "frmtrm_amount"

    for target in account_names:
        for row in rows:
            nm = (row.get("account_nm") or "").strip()
            if nm == target:
                amt = _to_int(row.get(field))
                if amt is not None:
                    return amt

    for target in account_names:
        for row in rows:
            nm = (row.get("account_nm") or "").strip()
            if target in nm:
                amt = _to_int(row.get(field))
                if amt is not None:
                    return amt

    return None


def get_financials(stock_code: str) -> dict:
    """
    종목의 최신 재무지표 반환.
    실패하거나 데이터 없으면 일부 키가 None.
    """
    global _last_acnt_error
    _last_acnt_error = None  # 이 호출 범위에서만 의미 있는 값으로 리셋

    result: dict[str, Any] = {
        "report_label": None,    # 예: "2024 사업보고서"
        "net_income": None,
        "equity": None,
        "debt": None,
        "revenue": None,
        "revenue_prev": None,
        "operating_income": None,
        "operating_income_prev": None,
        "roe": None,             # %
        "debt_ratio": None,      # %
        "revenue_growth": None,  # %
        "op_income_growth": None,  # %
    }

    corp_code = get_corp_code(stock_code)

    rows: list[dict] = []
    chosen: tuple[int, str] | None = None
    for year, reprt in _latest_report_codes():
        rows = _fetch_acnt_all(corp_code, year, reprt)
        if rows:
            chosen = (year, reprt)
            break

    if not rows or not chosen:
        return result

    label_map = {"11011": "사업보고서", "11013": "1분기보고서", "11012": "반기보고서", "11014": "3분기보고서"}
    result["report_label"] = f"{chosen[0]} {label_map.get(chosen[1], chosen[1])}"

    # 재무상태표 (BS) — 자본총계, 부채총계
    bs_rows = [r for r in rows if r.get("sj_div") == "BS"]
    is_rows = [r for r in rows if r.get("sj_div") in ("IS", "CIS")]

    # 회사마다 표기가 달라서 (예: SDI는 "부채 합계", "영업손익", "당기순손익")
    # 후보 여러 개를 정확 일치 우선으로 검색 — 부분 일치 fallback이 잘못 잡는 거
    # ("세후중단영업이익"이 "영업이익" 검색에 매칭되는 등) 방어
    EQUITY_NAMES = ["자본총계", "자본 합계", "자본합계"]
    DEBT_NAMES = ["부채총계", "부채 합계", "부채합계"]
    REVENUE_NAMES = ["매출액", "수익(매출액)", "영업수익"]
    OP_NAMES = ["영업이익", "영업이익(손실)", "영업손익"]
    NI_NAMES = [
        "당기순이익", "당기순이익(손실)", "당기순손익",
        "당기순손실", "반기순이익(손실)", "분기순이익(손실)",
    ]

    result["equity"] = _find_amount(bs_rows, EQUITY_NAMES)
    result["debt"] = _find_amount(bs_rows, DEBT_NAMES)

    # 손익계산서 — 매출, 영업이익, 당기순이익 (당기/전기)
    result["revenue"] = _find_amount(is_rows, REVENUE_NAMES, current=True)
    result["revenue_prev"] = _find_amount(is_rows, REVENUE_NAMES, current=False)
    result["operating_income"] = _find_amount(is_rows, OP_NAMES, current=True)
    result["operating_income_prev"] = _find_amount(is_rows, OP_NAMES, current=False)
    result["net_income"] = _find_amount(is_rows, NI_NAMES, current=True)

    # 비율 계산
    if result["net_income"] is not None and result["equity"]:
        result["roe"] = round(result["net_income"] / result["equity"] * 100, 2)
    if result["debt"] is not None and result["equity"]:
        result["debt_ratio"] = round(result["debt"] / result["equity"] * 100, 2)
    if result["revenue"] is not None and result["revenue_prev"]:
        result["revenue_growth"] = round((result["revenue"] / result["revenue_prev"] - 1) * 100, 2)
    if result["operating_income"] is not None and result["operating_income_prev"]:
        if result["operating_income_prev"] != 0:
            result["op_income_growth"] = round(
                (result["operating_income"] / result["operating_income_prev"] - 1) * 100, 2
            )

    return result


def check_data_freshness(report_label: str | None) -> dict:
    """
    사용 중인 DART 보고서의 신선도 평가.

    Returns:
        {
            "is_stale": bool,                  # True면 더 최신 분기 데이터가 있을 수 있음
            "days_since_coverage": int | None, # 보고서 커버 기간 끝난 후 며칠 지났는지
            "coverage_end": str | None,        # "2025-12-31" 같은 식
            "next_expected": str | None,       # "2026 1분기" 같은 식 — 아직 미제출
        }
    """
    out = {
        "is_stale": False,
        "days_since_coverage": None,
        "coverage_end": None,
        "next_expected": None,
    }
    if not report_label:
        return out

    import re as _re
    m = _re.match(r"(\d{4})\s+(.+)", report_label)
    if not m:
        return out

    rep_year = int(m.group(1))
    rep_type = m.group(2).strip()

    # 보고서가 커버하는 회계기간 종료일 + 다음 보고서 종류
    coverage_map = {
        "1분기보고서": (3, 31, "반기보고서"),
        "반기보고서":   (6, 30, "3분기보고서"),
        "3분기보고서": (9, 30, "사업보고서"),
        "사업보고서":   (12, 31, "1분기보고서"),
    }
    if rep_type not in coverage_map:
        return out

    end_month, end_day, next_type = coverage_map[rep_type]
    coverage_end = datetime(rep_year, end_month, end_day)
    days_old = (datetime.now() - coverage_end).days
    next_year = rep_year + 1 if rep_type == "사업보고서" else rep_year

    out["days_since_coverage"] = days_old
    out["coverage_end"] = coverage_end.strftime("%Y-%m-%d")
    out["next_expected"] = f"{next_year} {next_type.replace('보고서', '')}"
    # 60일 이상 지나면 "이미 다음 분기 회계기간 진행 중" → stale
    out["is_stale"] = days_old > 60
    return out


def get_recent_disclosures(stock_code: str, days: int = 30, max_count: int = 15) -> list[dict]:
    """최근 N일간 DART 공시 목록"""
    try:
        corp_code = get_corp_code(stock_code)
    except DartError:
        return []

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    try:
        data = _api_get("list.json", {
            "corp_code": corp_code,
            "bgn_de": start,
            "end_de": end,
            "page_no": "1",
            "page_count": str(max_count),
        })
    except DartError:
        return []

    items = []
    for d in data.get("list", []):
        date = d.get("rcept_dt", "")
        date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 else date
        rcept_no = d.get("rcept_no", "")
        items.append({
            "date": date_fmt,
            "title": (d.get("report_nm") or "").strip(),
            "submitter": d.get("flr_nm", ""),
            "rcept_no": rcept_no,
            "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}" if rcept_no else "",
        })
    return items


def is_preliminary_disclosure(title: str) -> bool:
    """
    공시 제목이 잠정실적(공정공시) 류인지 판별.
    예시:
      - "(공정공시)연결재무제표기준영업(잠정)실적"
      - "[기재정정](공정공시)연결재무제표기준영업(잠정)실적"
      - "매출액또는손익구조30%이상변동"
      - "(공정공시)영업(잠정)실적(공정공시)"
    """
    if not title:
        return False
    if "잠정실적" in title or "잠정)실적" in title:
        return True
    if "손익구조" in title and "변동" in title:
        return True
    # 공정공시 + 영업/실적 키워드
    if "공정공시" in title and ("영업" in title or "실적" in title):
        return True
    return False


def get_disclosure_content(rcept_no: str, max_chars: int = 6000) -> str:
    """
    DART 공시 본문 텍스트를 가져옴.
    document.xml API → ZIP 다운로드 → 내부 XML/HTML 파일 텍스트 추출.
    """
    key = _get_api_key()
    if not key or not rcept_no:
        return ""

    try:
        resp = requests.get(
            f"{BASE_URL}/document.xml",
            params={"crtfc_key": key, "rcept_no": rcept_no},
            timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return ""

    if resp.headers.get("Content-Type", "").startswith("application/json"):
        return ""  # 오류 응답

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        BeautifulSoup = None  # type: ignore

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            text_parts: list[str] = []
            total = 0
            for name in zf.namelist():
                if not name.lower().endswith((".xml", ".html", ".htm", ".txt")):
                    continue
                try:
                    with zf.open(name) as f:
                        raw = f.read().decode("utf-8", errors="ignore")
                except (OSError, KeyError):
                    continue

                if BeautifulSoup is not None:
                    # BeautifulSoup으로 style/script 블록 제거 후 깨끗한 텍스트 추출
                    soup = BeautifulSoup(raw, "lxml")
                    for tag in soup(["style", "script"]):
                        tag.decompose()
                    text = soup.get_text(separator=" ")
                else:
                    # Fallback: regex
                    text = re.sub(
                        r"<style[^>]*>.*?</style>", " ", raw,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    text = re.sub(
                        r"<script[^>]*>.*?</script>", " ", text,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"&[a-zA-Z]+;", " ", text)

                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    text_parts.append(text)
                    total += len(text)
                    if total >= max_chars * 2:
                        break
        return " ".join(text_parts)[:max_chars]
    except zipfile.BadZipFile:
        return ""


def calc_per_pbr(stock_code: str, current_price: float, shares: int, fin: dict) -> dict:
    """현재가 + 발행주식수 + DART 재무 → PER/PBR 계산"""
    out = {"per": None, "pbr": None, "eps": None, "bps": None}
    if not shares or shares <= 0:
        return out

    if fin.get("net_income") is not None:
        eps = fin["net_income"] / shares
        out["eps"] = round(eps, 2)
        if eps > 0:
            out["per"] = round(current_price / eps, 2)

    if fin.get("equity") is not None:
        bps = fin["equity"] / shares
        out["bps"] = round(bps, 2)
        if bps > 0:
            out["pbr"] = round(current_price / bps, 2)

    return out


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "005930"
    print(f"DART 키 설정: {is_configured()}")
    if not is_configured():
        print("→ .env 파일에 DART_API_KEY를 입력한 후 다시 실행하세요.")
        sys.exit(1)
    fin = get_financials(code)
    print(f"\n[{code}] 재무지표")
    for k, v in fin.items():
        print(f"  {k}: {v}")
