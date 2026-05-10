"""
네이버 금융 외국인/기관 매매 동향 스크래퍼
- https://finance.naver.com/item/frgn.naver?code=005930

⚠️ 주의: 네이버 금융 페이지 구조가 바뀌면 깨질 수 있습니다 (개인 분석용 한정).
        배포/상업화 단계에서는 한투 OpenAPI 등 공식 소스로 교체 권장.
"""
from io import StringIO
import re

import pandas as pd
import requests


HEADERS = {"User-Agent": "Mozilla/5.0"}
URL_TMPL = "https://finance.naver.com/item/frgn.naver?code={code}&page={page}"
NEWS_URL_TMPL = "https://finance.naver.com/item/news_news.naver?code={code}&page={page}&clusterId="


def _parse_int(v) -> float | None:
    if pd.isna(v):
        return None
    s = re.sub(r"[^\d\-]", "", str(v))
    if not s or s in ("-",):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def fetch_supply_demand(code: str, pages: int = 2) -> pd.DataFrame:
    """
    네이버 금융에서 종목별 외국인/기관 일별 순매매 데이터 가져오기.
    Returns DataFrame columns: date, close, volume, inst_net, foreign_net, foreign_ratio
    """
    rows = []
    for page in range(1, pages + 1):
        url = URL_TMPL.format(code=code, page=page)
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "euc-kr"
        if resp.status_code != 200:
            continue

        try:
            tables = pd.read_html(StringIO(resp.text))
        except ValueError:
            continue

        for t in tables:
            if t.shape[1] != 9 or t.shape[0] < 3:
                continue
            cols = [str(c) for c in t.columns]
            joined = " ".join(cols)
            if "외국인" in joined and "기관" in joined and "순매매량" in joined:
                t.columns = ["date", "close", "diff", "rate", "volume",
                             "inst_net", "foreign_net", "foreign_holding", "foreign_ratio"]
                rows.append(t)
                break

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["date", "close"])
    df = df[df["date"].astype(str).str.contains(r"\d{4}\.\d{2}\.\d{2}", regex=True)]

    df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d", errors="coerce")
    df["close"] = df["close"].apply(_parse_int)
    df["volume"] = df["volume"].apply(_parse_int)
    df["inst_net"] = df["inst_net"].apply(_parse_int)
    df["foreign_net"] = df["foreign_net"].apply(_parse_int)
    df["foreign_ratio"] = (
        df["foreign_ratio"].astype(str).str.replace("%", "").str.strip()
        .apply(lambda s: float(s) if re.match(r"^[\d\.\-]+$", s) else None)
    )

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def get_flow_summary(code: str) -> dict:
    """
    수급 요약 — 5일/20일 외국인/기관 누적 순매수, 외국인 보유율 변화
    """
    out = {
        "foreign_net_5d": None,    # 5일 누적 외국인 순매수 (주)
        "inst_net_5d": None,
        "foreign_net_20d": None,
        "inst_net_20d": None,
        "foreign_buy_streak": 0,   # 외국인 연속 순매수 일수 (음수면 연속 순매도)
        "inst_buy_streak": 0,
        "foreign_ratio": None,     # 최신 외국인 보유율
        "ratio_change_20d": None,  # 20일 전 대비 보유율 증감(%p)
        "last_date": None,         # 데이터 마지막 일자
    }

    df = fetch_supply_demand(code, pages=2)
    if df.empty:
        return out

    df = df.tail(40).reset_index(drop=True)
    if not df.empty:
        out["last_date"] = df["date"].iloc[-1].strftime("%Y-%m-%d")

    if len(df) >= 5:
        out["foreign_net_5d"] = int(df["foreign_net"].tail(5).sum())
        out["inst_net_5d"] = int(df["inst_net"].tail(5).sum())
    if len(df) >= 20:
        out["foreign_net_20d"] = int(df["foreign_net"].tail(20).sum())
        out["inst_net_20d"] = int(df["inst_net"].tail(20).sum())

    # 연속 순매수/순매도 카운트
    def streak(series: pd.Series) -> int:
        s = series.dropna().tolist()
        if not s:
            return 0
        sign = 1 if s[-1] > 0 else (-1 if s[-1] < 0 else 0)
        if sign == 0:
            return 0
        cnt = 0
        for v in reversed(s):
            if (sign > 0 and v > 0) or (sign < 0 and v < 0):
                cnt += 1
            else:
                break
        return cnt * sign

    out["foreign_buy_streak"] = streak(df["foreign_net"])
    out["inst_buy_streak"] = streak(df["inst_net"])

    if not df["foreign_ratio"].dropna().empty:
        out["foreign_ratio"] = float(df["foreign_ratio"].dropna().iloc[-1])
        if len(df["foreign_ratio"].dropna()) >= 20:
            out["ratio_change_20d"] = round(
                df["foreign_ratio"].dropna().iloc[-1] - df["foreign_ratio"].dropna().iloc[-20],
                2,
            )

    return out


CONGLOMERATE_PREFIXES = [
    "삼성", "현대", "LG", "SK", "롯데", "한화", "포스코", "POSCO",
    "두산", "GS", "CJ", "신세계", "KT", "DL", "효성", "코오롱",
]


# 종목코드 → 뉴스 제목 매칭 키워드 (수동 큐레이션).
# 원칙: 다른 회사와 절대 헷갈리지 않는 식별성 높은 별칭만.
# 의도적으로 제외:
#   - "엔솔" 단독 (다른 -엔솔 종목 또는 비종목 단어)
#   - "포스코" 단독 (POSCO인터, 퓨처엠 등 그룹사 섞임)
#   - "신한" 단독 (신한카드·은행 등)
#   - "전자"·"전기" 같은 일반어
STOCK_ALIASES: dict[str, list[str]] = {
    "005930": ["삼성전자", "삼전"],
    "000660": ["SK하이닉스", "하이닉스", "하닉"],
    "035420": ["NAVER", "네이버"],
    "035720": ["카카오"],
    "005380": ["현대차", "현대자동차"],
    "000270": ["기아", "기아차", "기아자동차"],
    "051910": ["LG화학"],
    "207940": ["삼성바이오로직스", "삼바"],
    "005490": ["POSCO홀딩스", "포스코홀딩스"],
    "068270": ["셀트리온"],
    "012330": ["현대모비스", "모비스"],
    "105560": ["KB금융"],
    "055550": ["신한지주", "신한금융지주"],
    "066570": ["LG전자"],
    "017670": ["SK텔레콤", "SKT"],
    "030200": ["KT"],
    "373220": ["LG에너지솔루션", "LG엔솔", "에너지솔루션"],
    "006400": ["삼성SDI", "SDI"],
    "086520": ["에코프로"],
    "247540": ["에코프로비엠"],
    "352820": ["하이브"],
    "028260": ["삼성물산"],
    "000810": ["삼성화재"],
    "032830": ["삼성생명"],
    "003670": ["포스코퓨처엠"],
    "041510": ["에스엠", "SM엔터테인먼트"],
    "035900": ["JYP엔터", "JYP Ent"],
    "086790": ["하나금융지주", "하나금융"],
    "316140": ["우리금융지주", "우리금융"],
    "138930": ["BNK금융지주", "BNK금융"],
}


def _stock_name_keywords(code: str = "", name: str = "") -> list[str]:
    """
    뉴스 제목 매칭에 쓸 키워드 후보 생성.

    1순위: 큐레이션된 STOCK_ALIASES (코드 기준) — 가장 정확
    2순위: 종목명 + 재벌 접두 제외 단축형(3자 이상)
    3순위: 종목명만
    """
    if code and code in STOCK_ALIASES:
        return list(STOCK_ALIASES[code])

    if not name:
        return []

    keywords = {name}
    for prefix in CONGLOMERATE_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix) + 1:
            tail = name[len(prefix):].strip()
            if len(tail) >= 3:
                keywords.add(tail)
            break
    return list(keywords)


def _is_stock_relevant(title: str, keywords: list[str]) -> bool:
    """제목에 종목 키워드 중 하나라도 들어있는지."""
    if not keywords:
        return True
    return any(kw in title for kw in keywords)


def get_recent_news(
    code: str,
    name: str = "",
    max_count: int = 12,
    pages: int = 2,
    strict_filter: bool = True,
) -> list[dict]:
    """
    네이버 금융 종목 뉴스 페이지에서 최근 뉴스 목록 가져오기.
    Returns: [{date, title, source, url}, ...] (제목 중복 제거)

    name이 주어지면 해당 종목과 관련 없는 뉴스(예: 삼성전자 페이지에 섞인 롯데하이마트 기사)
    를 제거합니다.

    ⚠️ 약관상 회색지대. 본인 분석용 한정.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    rows: list[dict] = []
    date_pat = re.compile(r"^\d{4}\.\d{2}\.\d{2}")

    for page in range(1, pages + 1):
        url = NEWS_URL_TMPL.format(code=code, page=page)
        try:
            resp = requests.get(url, headers={**HEADERS, "Referer": "https://finance.naver.com/"}, timeout=10)
            resp.encoding = "euc-kr"
            if resp.status_code != 200:
                continue
        except requests.RequestException:
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        for table in soup.find_all("table", class_="type5"):
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 3:
                    continue

                # 위치 무관: 링크 td / 날짜 td / 출처 td 패턴으로 식별
                title_td = None
                for td in tds:
                    a = td.find("a")
                    if a and "news_read" in (a.get("href") or ""):
                        title_td = td
                        break
                if not title_td:
                    continue

                a = title_td.find("a")
                title = a.get_text(strip=True)
                if not title or len(title) < 5:
                    continue
                href = a.get("href", "")
                full_url = "https://finance.naver.com" + href if href.startswith("/") else href

                date = ""
                source_candidates: list[str] = []
                for td in tds:
                    if td is title_td:
                        continue
                    text = td.get_text(strip=True)
                    if date_pat.match(text):
                        date = text
                    elif text and text != title:
                        source_candidates.append(text)
                # 출처는 보통 짧고 단일 — 후보 중 가장 짧은 것 선택
                source = min(source_candidates, key=len) if source_candidates else ""

                rows.append({
                    "title": title,
                    "url": full_url,
                    "source": source,
                    "date": date,
                })

    # 제목 중복 제거 (보존 순서)
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        if r["title"] in seen:
            continue
        seen.add(r["title"])
        deduped.append(r)

    # 종목 관련성 필터: 제목에 종목명(또는 단축형) 포함된 것만
    # (이전엔 0~2건이면 필터 해제했는데, 그러면 산업 일반 뉴스만 있는 종목은
    #  결국 노이즈가 다 나옴. 차라리 정직하게 0건이면 0건으로.)
    if strict_filter and (code or name):
        keywords = _stock_name_keywords(code, name)
        if keywords:
            deduped = [r for r in deduped if _is_stock_relevant(r["title"], keywords)]

    return deduped[:max_count]


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "005930"
    print(f"[{code}] 수급 요약")
    summary = get_flow_summary(code)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n[{code}] 뉴스 (최근 12건)")
    for n in get_recent_news(code, max_count=12):
        print(f"  {n['date']}  [{n['source']}]  {n['title'][:50]}")
