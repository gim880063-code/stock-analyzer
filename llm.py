"""
Gemini API 기반 공시 분석 모듈
- rcept_no 기반 persistent 캐시 → 동일 공시는 1회만 분석
- Flash: 빠른 제목 분류 (모든 공시)
- Pro: 본문 깊이 분석 (중요 공시 top 3~5)
"""
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


ENV_PATH = Path(__file__).parent / ".env"
DATA_DIR = Path(__file__).parent / "data"
CACHE_FILE = DATA_DIR / "llm_disclosure_cache.json"
CACHE_VERSION = "v1"
PLACEHOLDER_MARKERS = (
    "여기에",
    "붙여넣",
    "your_",
    "api_키",
    "api 키",
    "xxxxxxxx",
)

# Gemini -latest aliases: 자동으로 최신 stable Flash 모델로 라우팅.
# 2.5 Flash/Pro/Flash-Lite는 무료 한도가 일 20~0회로 매우 빡빡함 (확인됨).
# -latest 풀이 별도라 더 넉넉. 정형화된 분류·추출엔 Lite로도 충분.
FLASH_MODEL = "gemini-flash-lite-latest"
PRO_MODEL = "gemini-flash-latest"  # 깊이 분석엔 latest stable Flash


def _get_api_key() -> str:
    """API 키 조회 — Streamlit Cloud secrets 우선, 로컬 .env 백업.
    워커 스레드 대응으로 한 번 읽으면 os.environ 에 캐싱 (dart._get_api_key 와 동일 패턴)."""
    cached = os.environ.get("GEMINI_API_KEY", "").strip()
    if _is_real_api_key(cached):
        return cached
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            key = str(st.secrets["GEMINI_API_KEY"]).strip()
            if _is_real_api_key(key):
                os.environ["GEMINI_API_KEY"] = key
                return key
    except Exception:
        pass
    load_dotenv(ENV_PATH, override=True)
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key if _is_real_api_key(key) else ""


def _is_real_api_key(key: str) -> bool:
    if not key:
        return False
    low = key.lower()
    return not any(marker in low for marker in PLACEHOLDER_MARKERS)


def is_configured() -> bool:
    return bool(_get_api_key())


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = _get_api_key()
        if not key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        # timeout(ms) — 응답이 멈춘 호출이 워커를 무한 점유해 스크리닝이
        # 몇 시간씩 끝나지 않던 문제 방지. Flash 호출은 보통 1~5초.
        _client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=30_000),
        )
    return _client


# ─────────── persistent 캐시 ───────────

class DisclosureCache:
    def __init__(self, path: Path = CACHE_FILE):
        self.path = path
        # 여러 워커 스레드가 동시에 set()/_save() 호출 시 dict 변경 중 직렬화 보호
        self._lock = threading.Lock()
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if d.get("_version") == CACHE_VERSION:
                    return d
            except (json.JSONDecodeError, OSError):
                pass
        return {"_version": CACHE_VERSION, "items": {}}

    def get(self, rcept_no: str) -> dict | None:
        return self.data.get("items", {}).get(rcept_no)

    def has_pro_analysis(self, rcept_no: str) -> bool:
        # 'rationale' 필드 존재 여부로 깊이 분석됐는지 판별 (Pro/Flash-deep 모두 포함)
        item = self.get(rcept_no)
        return bool(item and item.get("rationale"))

    def set(self, rcept_no: str, analysis: dict) -> None:
        with self._lock:
            self.data.setdefault("items", {})[rcept_no] = analysis
            self._save_locked()

    def _save_locked(self) -> None:
        """_lock 이미 보유한 상태에서만 호출. 외부 호출 금지."""
        self.path.parent.mkdir(exist_ok=True)
        # json.dumps 도중 다른 스레드가 dict 변경하면 RuntimeError 가능 → 스냅샷 후 직렬화
        snapshot = {
            "_version": self.data.get("_version", CACHE_VERSION),
            "items": dict(self.data.get("items", {})),
        }
        self.path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


_cache: DisclosureCache | None = None
_cache_init_lock = threading.Lock()


def get_cache() -> DisclosureCache:
    global _cache
    if _cache is None:
        with _cache_init_lock:
            if _cache is None:
                _cache = DisclosureCache()
    return _cache


# ─────────── Gemini 호출 ───────────

CLASSIFY_PROMPT = """당신은 한국 주식 DART 공시 분석가입니다.
다음 공시를 분류하고 요약해주세요.

공시 제목: {title}
제출자: {submitter}
공시 날짜: {date}

분류 기준:
- "critical" (-2): 회사 자체에 중대한 문제 — 횡령·배임혐의발생, 상장폐지사유발생, 회생절차개시, 파산, 감자결정, 거래정지
- "negative" (-1): 주식가치 희석 또는 사업 위험 — 유상증자결정, 전환사채/신주인수권부사채 발행, 소송 등의 제기·신청, 영업정지
- "positive" (+1): 본업·주가에 호재 — 단일판매·공급계약체결(수주), 자기주식취득(자사주매입), 신규시설투자, 무상증자, 흑자전환
- "neutral" (0): 정보용 — 정기보고서(사업/분기/반기), 임원·주요주주 변동, 대량보유보고서, 특수관계인 거래, routine한 신고

JSON 형식으로만 응답:
{{
  "category": "critical|negative|positive|neutral",
  "score_impact": -2 ~ +1 정수,
  "summary": "핵심 요약 (한국어, 50자 이내, 명사형 종결)"
}}"""

DEEP_ANALYSIS_PROMPT = """당신은 한국 주식 공시 전문 분석가입니다.
다음 공시 본문을 깊이 분석해서 호재/악재 영향을 평가해주세요.

공시 제목: {title}
제출자: {submitter}
공시 날짜: {date}

[공시 본문 발췌]
{content}

분석 시 고려사항:
- 단일판매·공급계약은 계약 금액이 매출 대비 얼마나 큰지가 중요
- 유상증자는 자금 사용 목적(신사업 vs 운영자금)에 따라 호재/악재 갈림
- 소송은 우리 회사가 받은 것인지, 건 것인지 확인 필요

JSON 형식으로만 응답:
{{
  "category": "critical|negative|positive|neutral",
  "score_impact": -2 ~ +1 정수,
  "summary": "핵심 요약 (한국어, 80자 이내)",
  "key_points": ["구체적 사실 1 (금액·비율·당사자 포함)", "사실 2", "사실 3"],
  "rationale": "이 점수를 매긴 이유 (한국어, 100자 이내)"
}}"""


# ─── rate-limit 서킷 브레이커 ──────────────────────────────────────────
# 무료 일일 한도가 소진되면 모든 호출이 429 → 매번 최대 ~90초 재시도를 반복해
# 유니버스 전체(safe 는 100종목+)가 몇 시간씩 걸린다. 연속 rate-limit 이 임계를
# 넘으면 한동안 LLM 호출을 즉시 포기(룰 기반 분류로 graceful fallback)해 런이 빨리 끝나게 한다.
_rl_lock = threading.Lock()
_rl_consecutive = 0
_rl_tripped_until = 0.0
_RL_TRIP_THRESHOLD = 6      # 연속 rate-limit 최종 실패 횟수
_RL_COOLDOWN = 600.0        # 트립 후 LLM 호출을 건너뛸 시간(초)


class RateLimitExhausted(RuntimeError):
    """일일 한도 소진 추정 — 이번 런에서는 LLM 호출을 건너뜀."""


def _rl_should_skip() -> bool:
    with _rl_lock:
        return time.time() < _rl_tripped_until


def _rl_record(success: bool) -> None:
    global _rl_consecutive, _rl_tripped_until
    with _rl_lock:
        if success:
            _rl_consecutive = 0
        else:
            _rl_consecutive += 1
            if _rl_consecutive >= _RL_TRIP_THRESHOLD:
                _rl_tripped_until = time.time() + _RL_COOLDOWN


def _safe_call(model: str, prompt: str, temperature: float = 0.2, max_retries: int = 3) -> dict:
    """
    LLM 호출 + 429 (rate limit) 자동 retry.
    Gemini가 "retry in Xs" 힌트를 주면 그만큼 기다리고, 없으면 지수 백오프.
    한도 소진이 연속되면 서킷 브레이커가 트립돼 즉시 RateLimitExhausted 를 던진다.
    """
    if _rl_should_skip():
        raise RateLimitExhausted("Gemini 일일 한도 소진 추정 — LLM 호출 건너뜀")

    client = _get_client()
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=temperature,
                ),
            )
            result = json.loads(response.text)
            _rl_record(success=True)
            return result
        except Exception as e:
            last_error = e
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            # rate limit 이 아닌 오류(타임아웃·파싱 등)는 즉시 포기 — 워커 점유 방지
            if not is_rate_limit:
                raise
            if attempt >= max_retries - 1:
                _rl_record(success=False)   # 한도 소진 신호 누적 → 임계 넘으면 트립
                raise
            # 권장 retry 시간 파싱 (예: "retry in 29.1s")
            m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
            wait = float(m.group(1)) + 1 if m else (2 ** (attempt + 1))
            wait = min(wait, 30.0)
            time.sleep(wait)
    if last_error:
        raise last_error
    raise RuntimeError("LLM call failed without exception (impossible)")


BATCH_CLASSIFY_PROMPT = """다음 한국 주식 DART 공시 목록을 각각 분류하세요.

분류 기준:
- "critical" (-2): 회사 자체에 중대한 문제 — 횡령·배임혐의발생, 상장폐지사유발생, 회생절차개시, 파산, 감자결정, 거래정지
- "negative" (-1): 주식가치 희석 또는 사업 위험 — 유상증자결정, 전환사채/신주인수권부사채 발행, 소송 등의 제기·신청, 영업정지
- "positive" (+1): 본업·주가에 호재 — 단일판매·공급계약체결(수주), 자기주식취득(자사주매입), 신규시설투자, 무상증자, 흑자전환
- "neutral" (0): 정보용 — 정기보고서(사업/분기/반기), 임원·주요주주 변동, 대량보유보고서, 특수관계인 거래, routine한 신고

공시 목록:
{disclosures_text}

JSON 배열로만 응답 (위 순서대로, 항목 누락 금지):
[
  {{"index": 0, "category": "critical|negative|positive|neutral", "score_impact": -2 ~ +1, "summary": "핵심 요약 50자 이내"}},
  ...
]"""


def classify_titles_batch(disclosures: list[dict]) -> list[dict]:
    """
    공시 목록 전체를 한 번의 LLM 호출로 분류. 캐시 적중 항목은 LLM 호출에서 제외.
    classify_title()을 N번 호출하는 것보다 훨씬 빠름 (15회 → 1회).

    Returns: disclosures와 동일 길이의 분석 결과 리스트.
    """
    if not disclosures:
        return []

    cache = get_cache()
    results: list[dict | None] = [None] * len(disclosures)
    pending: list[tuple[int, dict]] = []  # (원본 인덱스, disclosure)

    for i, d in enumerate(disclosures):
        rcept_no = d.get("rcept_no", "")
        cached = cache.get(rcept_no) if rcept_no else None
        if cached and "category" in cached:
            results[i] = cached
        else:
            pending.append((i, d))

    # 모두 캐시 적중이면 LLM 호출 생략
    if not pending:
        return results  # type: ignore

    if is_configured():
        try:
            text_lines = []
            for j, (_, d) in enumerate(pending):
                text_lines.append(
                    f"{j}. 제목: {d.get('title', '')}  |  "
                    f"제출자: {d.get('submitter', '')}  |  "
                    f"날짜: {d.get('date', '')}"
                )
            prompt = BATCH_CLASSIFY_PROMPT.format(disclosures_text="\n".join(text_lines))
            batch_response = _safe_call(FLASH_MODEL, prompt, temperature=0.2)

            # 응답 파싱 + 개별 캐시 저장
            for item in batch_response:
                idx_in_batch = item.get("index", -1)
                if not (0 <= idx_in_batch < len(pending)):
                    continue
                full_idx, d = pending[idx_in_batch]
                out = {
                    "category": item.get("category", "neutral"),
                    "score_impact": int(item.get("score_impact", 0)),
                    "summary": (item.get("summary") or "")[:80],
                    "title": d.get("title", ""),
                    "rcept_no": d.get("rcept_no", ""),
                    "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "model": FLASH_MODEL,
                }
                results[full_idx] = out
                if d.get("rcept_no"):
                    cache.set(d["rcept_no"], out)
        except Exception:
            pass  # 실패 시 fallback으로 룰 기반 채워넣음

    # 누락된 것은 룰 기반으로 채움
    for i, d in enumerate(disclosures):
        if results[i] is None:
            results[i] = classify_by_rule(d.get("title", ""))

    return results  # type: ignore


def classify_title(rcept_no: str, title: str, submitter: str, date: str) -> dict:
    """공시 제목 기반 빠른 분류 (Flash). rcept_no로 캐시."""
    cache = get_cache()
    cached = cache.get(rcept_no)
    if cached and "category" in cached:
        return cached

    prompt = CLASSIFY_PROMPT.format(title=title, submitter=submitter, date=date)
    try:
        result = _safe_call(FLASH_MODEL, prompt)
    except Exception as e:
        # 실패 시 룰 기반 fallback
        return _rule_based_classify(title)

    out = {
        "category": result.get("category", "neutral"),
        "score_impact": int(result.get("score_impact", 0)),
        "summary": result.get("summary", title)[:80],
        "title": title,
        "rcept_no": rcept_no,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": FLASH_MODEL,
    }
    cache.set(rcept_no, out)
    return out


EXTRACT_PRELIMINARY_PROMPT = """당신은 한국 상장사 잠정실적 공시 분석가입니다.
다음 공시 본문에서 분기 또는 반기 매출·영업이익·당기순이익을 추출하세요.

[공시 제목]
{title}

[공시 본문 발췌]
{content}

【매우 중요: 단위 확인】
공시 본문에 "단위: 억원" 또는 "단위: 백만원" 같은 표시가 있습니다. 반드시 이를 보고 변환하세요.

  - "단위: 억원" 표기 시:
      예) "67,717" → 67,717 × 100,000,000 = 6,771,700,000,000원 (6.77조)
  - "단위: 백만원" 표기 시:
      예) "6,771,700" → 6,771,700 × 1,000,000 = 6,771,700,000,000원 (6.77조)
  - "단위: 조원" 표기 시:
      예) "6.77" → 6.77 × 1,000,000,000,000 = 6,770,000,000,000원 (6.77조)
  - 단위 표시 없으면 본문 맥락으로 추론 (대형주의 분기 매출은 보통 수천억~수십조 범위)

【추출 규칙】
- 적자(손실, △, ▽, "-", 마이너스) → 음수로 변환
- "전년동기" / "전기" / "전년동기실적" 항목을 *_yoy로 사용
- 자동차 판매대수 공시·생산판매 공시 등 매출/영업이익이 명시 안 된 공시는 모두 null로 응답
- 못 찾는 값은 null

JSON으로만 응답:
{{
  "period_label": "2026 1Q | 2025 반기 | 2025 사업연도",
  "is_consolidated": true,
  "revenue": 12345000000000,
  "revenue_yoy": 11000000000000,
  "operating_income": 100000000000,
  "operating_income_yoy": -50000000000,
  "net_income": 80000000000,
  "net_income_yoy": null
}}"""


def extract_preliminary_results(rcept_no: str, title: str, content: str) -> dict | None:
    """
    잠정실적 공시 본문에서 매출·영업이익·당기순이익을 추출.
    Flash 사용 — 정형화된 표 파싱이라 Pro 불필요, 무료 한도 넉넉.
    rcept_no 기반 영구 캐시.
    """
    if not rcept_no or not content:
        return None

    cache = get_cache()
    existing = cache.get(rcept_no) or {}
    if "preliminary_extract" in existing:
        return existing["preliminary_extract"]

    if not is_configured():
        return None

    prompt = EXTRACT_PRELIMINARY_PROMPT.format(
        title=title or "", content=(content or "")[:6000],
    )
    try:
        result = _safe_call(FLASH_MODEL, prompt, temperature=0.1)
    except Exception:
        return None

    # Sanity checks
    rev = result.get("revenue")
    rev_yoy = result.get("revenue_yoy")

    # 1) 매출이 비현실적으로 크면 단위 변환 오류 (KOSPI 분기 매출은 70조 이하)
    if rev is not None and rev > 200_000_000_000_000:  # 200조원 초과
        return None
    # 2) 매출은 음수일 수 없음
    if rev is not None and rev < 0:
        return None
    # 3) 전년동기 매출도 음수일 수 없음 (정정공시에서 일부 항목만 있는 경우 흔히 발생)
    if rev_yoy is not None and rev_yoy < 0:
        return None
    # 4) 매출 + 영업이익 둘 다 None이면 분기 실적 공시가 아님 (자동차 월간 판매 등)
    if rev is None and result.get("operating_income") is None:
        return None
    # 5) 정정공시인데 revenue/yoy가 비어있으면 정보 부족으로 신뢰 안 함
    is_correction = "정정" in (title or "")
    if is_correction and (rev is None or rev_yoy is None):
        return None

    # 기존 캐시 항목에 병합 저장
    existing["preliminary_extract"] = result
    existing.setdefault("title", title)
    existing.setdefault("rcept_no", rcept_no)
    existing["preliminary_extracted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    cache.set(rcept_no, existing)
    return result


def deep_analyze(rcept_no: str, title: str, submitter: str, date: str, content: str) -> dict:
    """
    공시 본문 깊이 분석 (Flash 사용).

    이전엔 Pro로 시도했으나 무료 티어에서 Pro 한도가 0이라 실패함.
    Flash 2.5도 분류·요약·근거 작성은 충분히 정확.
    """
    cache = get_cache()
    cached = cache.get(rcept_no)
    if cached and cached.get("rationale"):
        return cached  # 이미 깊이 분석된 결과

    content = (content or "").strip()[:6000]
    prompt = DEEP_ANALYSIS_PROMPT.format(
        title=title, submitter=submitter, date=date, content=content,
    )
    try:
        result = _safe_call(FLASH_MODEL, prompt)
    except Exception:
        if cached:
            return cached
        return _rule_based_classify(title)

    out = {
        "category": result.get("category", "neutral"),
        "score_impact": int(result.get("score_impact", 0)),
        "summary": result.get("summary", title)[:120],
        "key_points": result.get("key_points", []),
        "rationale": result.get("rationale", ""),
        "title": title,
        "rcept_no": rcept_no,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": FLASH_MODEL + "-deep",
    }
    cache.set(rcept_no, out)
    return out


# ─────────── 룰 기반 fallback ───────────

CRITICAL_KW = ["횡령", "배임", "상장폐지사유", "회생절차개시", "파산", "감자결정", "거래정지"]
NEGATIVE_KW = ["유상증자결정", "전환사채", "신주인수권부사채", "소송 등의 제기", "영업정지"]
POSITIVE_KW = ["단일판매", "공급계약체결", "자기주식취득", "자사주매입", "신규시설투자", "무상증자"]


def classify_by_rule(title: str) -> dict:
    """API 키 없거나 LLM 실패 시 룰 기반 fallback 분류."""
    if any(kw in title for kw in CRITICAL_KW):
        return {"category": "critical", "score_impact": -2, "summary": title, "model": "rule"}
    if any(kw in title for kw in NEGATIVE_KW):
        return {"category": "negative", "score_impact": -1, "summary": title, "model": "rule"}
    if any(kw in title for kw in POSITIVE_KW):
        return {"category": "positive", "score_impact": 1, "summary": title, "model": "rule"}
    return {"category": "neutral", "score_impact": 0, "summary": title, "model": "rule"}


_rule_based_classify = classify_by_rule  # 하위 호환


if __name__ == "__main__":
    print("Gemini 키 설정:", is_configured())
    if is_configured():
        r = classify_title(
            "TEST_NO", "단일판매·공급계약체결", "삼성전자", "2026-05-08"
        )
        print(json.dumps(r, ensure_ascii=False, indent=2))
