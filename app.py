"""
주식 분석 리포트 - Streamlit 웹 앱
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

from analyzer import (
    KOREAN_NAMES, UNIVERSE_LABELS, all_korean_stocks, analyze,
    get_universe_codes,
)
import dart
import llm
import portfolio as port
import history as hist_module
import cloud_store
import screening_history


DEFAULT_WATCHLIST = ["005930", "000660", "035420"]


def _load_str_list(filename: str) -> list[str]:
    data = cloud_store.load(filename, [])
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    return []


def _save_str_list(filename: str, items: list[str]) -> None:
    cloud_store.save(filename, items)


def load_watchlist() -> list[str]:
    items = _load_str_list("watchlist.json")
    return items if items else DEFAULT_WATCHLIST.copy()


def save_watchlist(codes: list[str]) -> None:
    _save_str_list("watchlist.json", codes)


def load_favorites() -> list[str]:
    return _load_str_list("favorites.json")


def save_favorites(codes: list[str]) -> None:
    _save_str_list("favorites.json", codes)


def toggle_favorite(code: str) -> None:
    favs = load_favorites()
    if code in favs:
        favs.remove(code)
    else:
        favs.append(code)
    save_favorites(favs)


@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_dict() -> dict[str, str]:
    """전체 KRX 종목 목록 (코드 → 이름). 24시간 캐시."""
    try:
        return all_korean_stocks()
    except Exception:
        return KOREAN_NAMES.copy()


st.set_page_config(
    page_title="주식 분석 리포트",
    # 이모지 대신 PNG 이미지 URL — iOS Safari가 home screen 아이콘으로 사용하는 경우가 있음
    # (Streamlit의 기본 PWA 매니페스트는 그대로라 Android에선 효과 제한적)
    page_icon="https://abs.twimg.com/emoji/v2/72x72/1f4c8.png",
    layout="wide",
)


def _check_password() -> bool:
    """
    비밀번호 보호 — st.secrets에 APP_PASSWORD가 설정된 경우만 활성화.
    로컬 개발(.env 사용)에선 자동으로 통과 (secrets 없음).
    """
    try:
        configured = st.secrets.get("APP_PASSWORD", "")
    except (FileNotFoundError, AttributeError, Exception):
        return True
    if not configured:
        return True
    if st.session_state.get("_authenticated"):
        return True

    st.markdown("## 🔒 주식 분석 리포트")
    st.caption("비밀번호를 입력하세요")
    pw = st.text_input("비밀번호", type="password", label_visibility="collapsed")
    if pw == configured:
        st.session_state["_authenticated"] = True
        st.rerun()
    elif pw:
        st.error("비밀번호가 틀렸습니다.")
    return False


if not _check_password():
    st.stop()

st.title("📈 주식 분석 리포트")
st.caption("관심 종목을 등록하면 추세·모멘텀·거래량·가격 리스크를 자동으로 점수화합니다.")


# 사이드바: 관심 종목 관리
with st.sidebar:
    st.header("관심 종목")

    stock_dict = get_stock_dict()
    options = [f"{code} {name}" for code, name in sorted(stock_dict.items())]

    # 즐겨찾기 클릭으로 추가 요청된 코드를 multiselect 렌더 전에 주입
    pending_to_add = st.session_state.pop("_pending_add_to_watchlist", [])
    if pending_to_add:
        current = st.session_state.get("watchlist_select")
        if current is None:
            saved = load_watchlist()
            current = [f"{c} {stock_dict.get(c, '?')}" for c in saved if c in stock_dict]
        for c in pending_to_add:
            opt = f"{c} {stock_dict.get(c, '?')}"
            if opt in options and opt not in current:
                current = current + [opt]
        st.session_state["watchlist_select"] = current

    saved_codes = load_watchlist()
    default_options = [f"{c} {stock_dict.get(c, '?')}" for c in saved_codes if c in stock_dict]

    multiselect_kwargs = {
        "options": options,
        "help": "입력창에 '삼성', '하이닉스', '005930' 등을 타이핑하면 검색됩니다",
        "placeholder": "예: 삼성전자",
        "key": "watchlist_select",
    }
    if "watchlist_select" not in st.session_state:
        multiselect_kwargs["default"] = default_options

    selected = st.multiselect("종목 검색 (이름·코드 모두 가능)", **multiselect_kwargs)
    selected_codes = [s.split(maxsplit=1)[0] for s in selected]

    custom_code = st.text_input(
        "여기 없는 종목코드 직접 추가 (6자리)",
        placeholder="예: 005935",
        max_chars=6,
    )
    if custom_code and custom_code.strip().isdigit() and len(custom_code.strip()) == 6:
        if custom_code.strip() not in selected_codes:
            selected_codes.append(custom_code.strip())

    # 변경될 때마다 자동 저장
    save_watchlist(selected_codes)

    st.divider()
    run_analysis = st.button("🔍 분석하기", type="primary", use_container_width=True)

    st.divider()
    if dart.is_configured():
        st.success("✅ DART API 키 연결됨\n\n가치/재무/성장성/공시 포함")
    else:
        st.warning("⚠️ DART API 키 미설정\n\n`.env` 파일에 `DART_API_KEY` 입력 후 재시작하세요")

    if llm.is_configured():
        st.success("✅ Gemini API 키 연결됨\n\n공시 자동 분류·요약 활성화")
    else:
        st.info("ℹ️ Gemini API 키 미설정 — 공시는 룰 기반 분류만 동작 (`GEMINI_API_KEY`)")

    if cloud_store.is_configured():
        st.success(
            "✅ Gist 영구 저장소 연결됨\n\n"
            "포트폴리오·점수 히스토리·즐겨찾기·워치리스트가 클라우드 재배포 후에도 유지됩니다."
        )
    else:
        st.info(
            "ℹ️ Gist 미설정 — 로컬 파일에만 저장 "
            "(Streamlit Cloud 재배포 시 데이터 초기화됨). "
            "secrets에 `GITHUB_PAT` + `GIST_ID` 추가 시 영구 저장."
        )

    st.divider()
    st.subheader("⭐ 즐겨찾기")
    st.caption("종목명 클릭 시 자동으로 분석됩니다")
    favorites = load_favorites()

    if not favorites:
        st.caption("종목 카드의 ☆ 버튼으로 추가하세요")
    else:
        results_map = {r["code"]: r for r in st.session_state.get("results", [])}
        for code in favorites:
            name = stock_dict.get(code, "?")
            r = results_map.get(code)
            with st.container(border=True):
                col_main, col_x = st.columns([5, 1])
                with col_main:
                    btn_label = f"⭐  {name}  ({code})"
                    if st.button(
                        btn_label,
                        key=f"fav_btn_{code}",
                        use_container_width=True,
                        help=f"{name}만 단독 분석 (워치리스트 유지)",
                    ):
                        st.session_state["_focus_code"] = code
                        st.rerun()

                    if r and not r.get("error"):
                        score = r["total"]
                        opinion = r["opinion"].split(" — ")[0]
                        color = "#1f7a3a" if score > 0 else ("#a3201a" if score < 0 else "#666")
                        st.markdown(
                            f"<small>{r['last_close']:,.0f}원 ({r['change_pct']:+.2f}%) · "
                            f"<span style='color:{color};font-weight:600'>"
                            f"{score:+d}점 · {opinion}</span></small>",
                            unsafe_allow_html=True,
                        )

                with col_x:
                    if st.button("✖", key=f"unfav_{code}", help="즐겨찾기 해제"):
                        toggle_favorite(code)
                        st.rerun()

        if st.button("⭐ 즐겨찾기만 분석", use_container_width=True, key="analyze_favs"):
            st.session_state["_analyze_favs_only"] = True
            st.rerun()
        st.caption("워치리스트와 별도로 즐겨찾기만 분석합니다")

    # ─────────── 포트폴리오 ───────────
    st.divider()
    st.subheader("💼 내 포트폴리오")

    portfolio = port.load_portfolio()
    results_map = {r["code"]: r for r in st.session_state.get("results", []) if not r.get("error")}

    if portfolio:
        # 분석된 종목 한정으로 손익 합계 계산
        total_cost = 0.0
        total_value = 0.0
        priced = 0
        for code, h in portfolio.items():
            r = results_map.get(code)
            if r:
                pnl = port.compute_pnl(h, r["last_close"])
                total_cost += pnl["cost"]
                total_value += pnl["market_value"]
                priced += 1

        if priced > 0:
            total_profit = total_value - total_cost
            total_pct = (total_value / total_cost - 1) * 100 if total_cost > 0 else 0
            color = "#1f7a3a" if total_profit >= 0 else "#a3201a"
            st.markdown(
                f"<small>총 평가액 <b>{total_value:,.0f}원</b> · "
                f"<span style='color:{color}'>손익 {total_profit:+,.0f}원 "
                f"({total_pct:+.2f}%)</span> · {priced}/{len(portfolio)}개 가격 반영</small>",
                unsafe_allow_html=True,
            )

        for code, h in portfolio.items():
            name = stock_dict.get(code, "?")
            r = results_map.get(code)
            with st.container(border=True):
                col_main, col_x = st.columns([5, 1])
                with col_main:
                    if st.button(
                        f"💼 {name} ({code})",
                        key=f"port_btn_{code}",
                        use_container_width=True,
                        help=f"{name}만 단독 분석",
                    ):
                        st.session_state["_focus_code"] = code
                        st.rerun()
                    cost = h["quantity"] * h["avg_price"]
                    info = f"<small>{h['quantity']:,}주 @ {h['avg_price']:,.0f}원 · 원금 {cost:,.0f}원</small>"
                    if r:
                        pnl = port.compute_pnl(h, r["last_close"])
                        color = "#1f7a3a" if pnl["profit"] >= 0 else "#a3201a"
                        info += (
                            f"<br><small>현재 {r['last_close']:,.0f}원 · "
                            f"<span style='color:{color};font-weight:600'>"
                            f"{pnl['profit']:+,.0f}원 ({pnl['profit_pct']:+.2f}%)</span></small>"
                        )
                    st.markdown(info, unsafe_allow_html=True)
                with col_x:
                    if st.button("✖", key=f"del_port_{code}", help="포트폴리오에서 제거"):
                        port.remove_holding(code)
                        st.rerun()
    else:
        st.caption("아래에서 보유 종목을 등록하세요")

    with st.expander("➕ 보유 종목 추가/수정"):
        # 한글 이름이 먼저 → 가나다 순 정렬 → 검색창에 "삼성" 같은 한글 입력으로 빠르게 필터
        port_options = [f"{n} {c}" for c, n in sorted(stock_dict.items(), key=lambda kv: kv[1])]
        port_sel = st.selectbox(
            "종목 검색 (한글 이름 또는 코드)",
            options=port_options,
            key="port_add_select",
            index=0,
        )
        port_qty = st.number_input(
            "수량 (주)", min_value=1, step=1, value=10, key="port_add_qty",
        )
        port_price = st.number_input(
            "평균 매수가 (원)", min_value=1, step=100, value=10000, key="port_add_price",
        )
        if st.button("저장", key="port_add_save", use_container_width=True):
            # "삼성전자 005930" 형식 → 마지막 토큰이 6자리 코드
            code = port_sel.rsplit(maxsplit=1)[-1]
            port.add_holding(code, port_qty, port_price)
            st.success(f"{stock_dict.get(code, code)} 저장됨")
            st.rerun()

    # ─────────── 스크리닝 후보 진입점 (가벼움) ───────────
    st.divider()
    recent_picks_count = screening_history.get_recent(days=90)
    if recent_picks_count:
        # 간단 통계
        try:
            safe_codes_quick = set(get_universe_codes("safe"))
        except Exception:
            safe_codes_quick = set()
        n_total = len(recent_picks_count)
        n_dropped_screen = sum(1 for m in recent_picks_count.values() if not m["in_latest"])
        n_dropped_uni = sum(1 for c in recent_picks_count if c not in safe_codes_quick)
        n_active = n_total - n_dropped_screen

        btn_label = f"📌 최근 스크리닝 후보 ({n_active}/{n_total})"
        if st.button(btn_label, use_container_width=True, key="open_screen_hist"):
            st.session_state["_view_mode"] = "screening_history"
            st.rerun()

        sub_parts = [f"현역 {n_active}개"]
        if n_dropped_screen > 0:
            sub_parts.append(f"⚠️탈락 {n_dropped_screen}")
        if n_dropped_uni > 0:
            sub_parts.append(f"⚠️유니버스이탈 {n_dropped_uni}")
        st.caption(" · ".join(sub_parts) + " · 90일 누적")
    else:
        st.caption(
            "📌 스크리닝 돌리면 통과 종목이 여기 자동으로 누적됩니다 (90일 보관)"
        )

    st.divider()
    st.subheader("🔍 종목 발굴 (스크리닝)")
    st.caption("종합점수 높은 종목을 자동 검색")

    universe = st.selectbox(
        "유니버스",
        options=list(UNIVERSE_LABELS.keys()),
        format_func=lambda k: UNIVERSE_LABELS[k],
        key="screen_universe_select",
        help="🛡️ 안전 유니버스 = 시총 5조+ / 거래대금 500억+ / 관리종목 등 제외 — 작전주 위험 낮은 후보들",
    )

    min_score = st.slider("최소 종합점수", -10, 10, 0, 1, key="screen_min_score")

    # 시간 추정 — 1차 스크리닝(공시 분류·잠정실적·뉴스, 깊이 분석 X) + 2차 깊이 분석
    est_codes = len(get_universe_codes(universe))
    est_pass1_sec = est_codes * 8 / 3
    est_min = max(1, round(est_pass1_sec / 60))
    st.caption(
        f"⏱️ 1차 스크리닝 약 {est_min}분 예상 "
        f"({est_codes}개 ÷ 3병렬). 통과 종목엔 추가 깊이 분석 자동 실행."
    )

    if st.button("🔍 발굴 시작", use_container_width=True, key="run_screen"):
        st.session_state["_screen"] = {
            "universe": universe,
            "min_score": min_score,
        }
        st.rerun()

    st.divider()
    st.subheader("📚 용어 사전")
    st.caption("처음이라면 펼쳐서 읽어보세요")

    with st.expander("📈 추세 / 60일선"):
        st.markdown("""
**추세**: 주가가 큰 방향으로 어디로 가고 있는가.

**60일 이동평균선 (60일선)**: 최근 60일 종가의 평균을 이은 선. **중기 추세**를 보는 가장 보편적인 기준입니다.

- 현재가 > 60일선 → 중기 상승 흐름 ✅
- 현재가 < 60일선 → 중기 하락 흐름 ⚠️

📌 **예시:** 삼성전자 현재가 268,500원, 60일선 200,025원 → 60일선 위 → 중기 흐름 양호.

**왜 보나?** 하루이틀 흔들림에 휘둘리지 않고 큰 방향을 잡기 위해.
""")

    with st.expander("⚡ 모멘텀 / RSI"):
        st.markdown("""
**모멘텀**: 주가의 상승/하락 "기운"이 얼마나 강한가.

**RSI (14일)**: 최근 14일간 오른 날과 내린 날의 비율을 0~100으로 표시.

| RSI | 의미 |
|---|---|
| 70 이상 | 🔴 단기 **과열** — 조정 가능성 |
| 50~70 | 🟢 상승 모멘텀 유효 |
| 30~50 | ⚪ 모멘텀 약화 |
| 30 이하 | 🟢 단기 **과매도** — 반등 가능성 |

📌 **예시:** SK하이닉스 RSI 93 → 극단적 과열 → 단기 조정 주의.

**왜 보나?** 좋은 종목도 너무 빨리 오르면 잠시 쉬어갑니다. 비싸게 사지 않기 위해.
""")

    with st.expander("📊 거래량"):
        st.markdown("""
**거래량**: 그날 사고팔린 주식 수. 가격 움직임의 "무게"를 알려줍니다.

20일 평균 대비 비율 + 최근 5일 가격 방향을 함께 봅니다:

| 조합 | 의미 | 점수 |
|---|---|---|
| 평균 ≥1.5배 + 가격 상승 | ✅ 매수세 유입 | +1 |
| 평균 ≥1.5배 + 가격 하락 | ⚠️ 매도 압력 | -1 |
| **평균 <0.6배 + 단기 상승** | ⚠️ **거래량 부족 — 상승 신뢰도 약함** | **-1** |
| 평균 <0.6배 + 단기 하락 | 매도 압력도 약함 | 0 |
| 그 외 | 평소 수준 | 0 |

📌 **예시:** 어떤 종목이 5일 +5% 상승했는데 거래량은 평균의 0.49배 → "사람들이 의심하면서 살짝 오른 것" → 신뢰도 낮음.

**왜 보나?** 거래량 없는 가격 변동은 신뢰도가 낮습니다.
""")

    with st.expander("🚨 가격 리스크"):
        st.markdown("""
**단기 변동성이 너무 커진 상태인가** 를 봅니다. 여러 신호가 누적되면 점수가 깊어집니다.

| 트리거 | 점수 |
|---|---|
| 5일 +15% 이상 급등 | -2 |
| 5일 +8~15% 상승 | -1 |
| **RSI 80 이상 (극단 과열)** | -1 |
| **52주 고점 97% 이상 근접** | -1 |
| 20일 고점 대비 -10% 이상 조정 | -1 |

(최대 -2점까지 누적)

📌 **예시:** SK하이닉스 5일 +30% 급등 + RSI 93 → 두 신호 다 발동 → -2점 (cap).

**왜 보나?** 좋은 종목이라도 비싼 시점에 진입하면 단기 손실 위험이 커집니다. 단일 신호가 아닌 **여러 신호의 누적**으로 봐야 정확합니다.
""")

    with st.expander("💰 수급 (외국인/기관)"):
        st.markdown("""
**큰 손**들이 사고 있는가, 팔고 있는가.

- **외국인**: 해외 자금. 보통 장기적·기관적 시각.
- **기관**: 연기금/자산운용사 등 국내 전문 투자자.

| 신호 | 의미 |
|---|---|
| 외국인 + 기관 동반 순매수 | ✅ 강한 수급 |
| 동반 순매도 | ⚠️ 수급 이탈 |
| 외국인 5일 연속 순매수 | ✅ 추세적 매집 |
| 외국인 보유율 상승 | ✅ 장기 신뢰 |

📌 **예시:** NAVER는 외국인+기관 동반 순매도 + 외국인 보유율 -0.9%p → 수급 약화.

**왜 보나?** 큰 자금의 방향은 가격을 움직이는 가장 강한 힘 중 하나입니다.
""")

    with st.expander("💎 가치 / PER · PBR"):
        st.markdown("""
**가치**: 지금 가격이 비싼가, 싼가 (실적·자산 대비).

### PER (주가수익비율)
> 현재 주가 ÷ 1주당 순이익

"이 회사 **1년 이익의 몇 배** 가격에 거래되는가"

| PER | 의미 |
|---|---|
| 10 이하 | 🟢 저평가 가능성 |
| 10~20 | ⚪ 보통 |
| 20 이상 | 🔴 고평가 (성장 기대 반영) |

### PBR (주가순자산비율)
> 현재 주가 ÷ 1주당 순자산

"회사가 **청산되면 받을 수 있는 돈의 몇 배**에 거래되는가"

| PBR | 의미 |
|---|---|
| 1 미만 | 🟢 자본 대비 저평가 |
| 1~2 | ⚪ 보통 |
| 2 이상 | 🔴 자본 대비 고평가 |

📌 **예시:** 삼성전자 PER 34.7 / PBR 3.60 → 둘 다 고평가 영역.

**주의:** 업종마다 적정 PER이 다릅니다 (제조업 ↓ / IT·바이오 ↑). 같은 업종 내 비교가 더 의미 있어요.
""")

    with st.expander("🏥 재무 건전성 / ROE · 부채비율"):
        st.markdown("""
**회사의 체력**을 봅니다.

### ROE (자기자본이익률)
> 당기순이익 ÷ 자본총계 × 100

"가진 자본으로 **얼마나 효율적으로 돈을 벌었는가**"

| ROE | 의미 |
|---|---|
| 15% 이상 | 🟢 우수 |
| 8~15% | ⚪ 양호 |
| 8% 미만 | 🔴 낮음 |
| 마이너스 | ❌ 적자 |

### 부채비율
> 부채총계 ÷ 자본총계 × 100

"자본 1원당 빚이 얼마인가"

| 부채비율 | 의미 |
|---|---|
| 100% 이하 | 🟢 안정적 |
| 100~200% | ⚪ 양호 |
| 200% 이상 | 🔴 부채 부담 |

📌 **예시:** SK하이닉스 ROE 35.6% (우수) + 부채비율 46% (안정) → 매우 건강.

**주의:** 금융업·조선업·건설업은 구조상 부채비율이 높습니다.
""")

    with st.expander("🛡️ 안전 유니버스 (작전주 회피)"):
        st.markdown("""
**작전주(주가 조작 위험 종목)** 를 자동으로 걸러낸 후보 종목 묶음입니다.

### 4단계 필터

| 조건 | 의도 |
|---|---|
| **시가총액 ≥ 5조원** | 시총이 큰 회사는 시장 조작에 큰 자금 필요 → 작전주 어려움 |
| **하루 거래대금 ≥ 500억원** | 유동성 충분 → 큰 매수·매도 흡수 가능, 가격 왜곡 어려움 |
| **유통주식 충분 (간접)** | 시총·거래대금 통과 시 자연스럽게 충족 |
| **공시 분류 정상** | 관리종목·투자주의환기·SPAC·외국기업 제외 → 공시 투명·정상 |

### 단계별 효과 (오늘 KRX 기준)

```
전체:                        2,878개
시총 5조 이상:                  135개
+ 거래대금 500억 이상:           94개
+ 정상 분류 (관리종목 등 제외):   83개
```

→ 약 **3% (83개)** 만 통과. 모두 KOSPI 대형주.

### 기준값

| 항목 | 값 | 통과 종목 |
|---|---|---|
| 시가총액 | 5조원 이상 | 약 90개 |
| 일일 거래대금 | 500억원 이상 | (위 + 거래대금 필터) |
| 분류 | 관리종목·SPAC·외국기업 제외 | (정상 분류만) |

### 한계

- 작전주 위험이 **0**은 아님 — 대형주도 단기 조정 가능
- "좋은 종목"이 아니라 "위험 분류는 아닌 종목" — 진입 안전성 ≠ 수익성
- 종합점수가 함께 좋아야 매수 후보 (스크리닝의 본질)

**권장 사용:** 안전 유니버스 → 종합점수 ≥ 0 필터 → 후보 5~10개 발견 → 즐겨찾기 → 풀 분석으로 깊이 검증
""")

    with st.expander("📰 종목 뉴스"):
        st.markdown("""
**참고용**으로만 표시되는 최근 종목 관련 뉴스 헤드라인입니다.

- 출처: 네이버 금융 종목뉴스 (HTML 스크래핑)
- **점수에 영향 없음** — 정보만 제공
- 광고성·추측성 기사가 섞일 수 있으니 본인 판단 필요

### 왜 점수에 안 넣었나?

| 함정 | 영향 |
|---|---|
| "[속보] OO 호재 폭발" 류 광고성 기사 | LLM이 호재로 분류 → 점수 왜곡 |
| 같은 사건을 10곳이 다른 톤으로 보도 | 중복 제거 어려움 |
| 작전주 리딩방 글 | 잘못된 신호 |

**DART 공시는 법적 책임이 있는 사실**이지만 뉴스는 추측·해설이 섞여 있어서 점수화하지 않습니다.

### 뉴스 vs DART 공시 역할

- **DART 공시** = 사실의 1차 자료 (점수에 반영)
- **뉴스** = 산업·정책·시장 분위기 (참고용)

뉴스로 "어 무슨 일 있나?" 확인 → DART에서 사실 검증 순서로 활용하세요.
""")

    with st.expander("📋 공시 (DART)"):
        st.markdown("""
**공시**: 회사가 의무적으로 알려야 하는 중대 사건 — 수주, 증자, 소송, 지분 변동, 정기 실적 보고 등.

DART OPEN API에서 최근 30일 공시 목록을 받아서 **Gemini로 분류**합니다.

| 카테고리 | 점수 | 예시 |
|---|---|---|
| 🚨 **중대** | -2 | 횡령·배임, 상장폐지사유, 회생/파산, 감자, 거래정지 |
| ⚠️ **부정** | -1 | 유상증자, 전환사채, 소송 제기, 영업정지 |
| ✅ **긍정** | +1 | 단일판매·공급계약(수주), 자사주 매입, 신규시설투자 |
| ⚪ **중립** | 0 | 정기보고서, 임원·주요주주 변동, 대량보유보고서 |

**점수 산출:**
- 중대 1건이라도 → -2
- 부정 ≥1건 → -1 (중대 없을 때)
- 긍정 ≥1건 → +1 (부정/중대 없을 때)

**처리 흐름:**
1. **모든 공시** → Gemini Flash로 제목 분류 (빠름, 캐시됨)
2. **중요 공시 상위 3개** → Gemini Pro로 본문까지 깊이 분석 (캐시됨)
3. 같은 공시(`rcept_no`)는 **한 번만 LLM 호출** — 새 공시 올라오기 전까지 재사용

각 공시 옆 표시:
- 🧠 = Pro로 본문 분석 완료
- ⚡ = Flash 제목 분류만
- (없음) = 룰 기반 분류 (LLM 키 없을 때)

📌 **예시:** 단일판매·공급계약체결 → ✅ 긍정 → "5,000억원 규모 ASML 계약, 매출의 X%" 같은 요약.

**한계:** 정기보고서(사업보고서)는 너무 길어서 본문 분석 안 함. 핵심 공시(수주·증자·소송)만 깊이 분석.
""")

    with st.expander("🚀 성장성"):
        st.markdown("""
**작년 대비 회사가 얼마나 컸는가** — 매출 성장 + **영업이익의 흑자/적자 상태와 마진 변화**를 함께 봅니다.

### 매출 성장률
| 매출 성장률 | 점수 |
|---|---|
| 15% 이상 | +1 |
| 5~15% | 0 |
| 0~5% | 0 |
| 마이너스 | -1 |

### 영업이익 (흑자/적자 상태가 핵심)
| 상태 변화 | 점수 |
|---|---|
| 흑자 → 흑자, **마진 +1%p 이상 개선** | +1 |
| 흑자 → 흑자, 마진 비슷 | 0 |
| 흑자 → 흑자, **마진 -2%p 이상 악화** | -1 |
| **적자 → 흑자 (흑자전환)** | **+2** |
| **흑자 → 적자 (적자전환)** | **-2** |
| 적자 → 적자, 손실 폭 축소 | 0 |
| 적자 → 적자, 손실 확대 | -1 |

⚠️ **단순 +426% 같은 % 숫자는 함정**입니다. 작년 영업이익이 -100억이었다가 -50억이 되면 성장률 표면 숫자는 좋아 보이지만 여전히 적자입니다. 그래서 이 앱은 **영업이익률(margin)** 변화를 직접 봅니다.

📌 **예시:**
- SK하이닉스: 흑자 → 흑자, 마진 크게 개선 → +1
- 어떤 적자 회사가 손실을 줄임 → 0 (좋아진 것이지만 아직 흑자가 아님)
- 흑자 회사가 적자전환 → -2 (가장 강한 부정 신호)
""")

    with st.expander("🎯 종합점수와 의견"):
        st.markdown("""
모든 항목 점수를 더해 종합점수를 냅니다. **가능한 양의 점수 합 대비 비율**로 의견을 결정합니다:

| 의견 | 기준 (비율) | 뜻 |
|---|---|---|
| ✨ **관심 유지** | ≥ 50% | 다수 지표가 우호적 |
| 🟡 **분할 접근 가능** | ≥ 20% | 일부 위험 있어 나눠서 매수 권장 |
| ⚪ **중립** | ≥ 0% | 긍정·부정 신호 비슷, **추가 확인 필요** |
| 👀 **관망** | ≥ -30% | 위험 요인 많음, 진입 미루는 게 안전 |
| ⚠️ **리스크 확대** | < -30% | 다수 부정적, 신중 |

**0점은 더 이상 "분할 접근"이 아닙니다.** 0~약간 양수는 "중립"으로 분류돼 추가 확인을 권장합니다.

**중요:** 이 앱은 **매수·매도 추천이 아닙니다**. 여러 지표를 정리해서 보여줄 뿐, 최종 판단은 본인의 몫입니다.
""")


# 분석 결과 캐시
@st.cache_data(ttl=600, show_spinner=False)
def cached_analyze(code: str, lite: bool = False, deep_top: int = 3) -> dict:
    return analyze(code, lite=lite, deep_top=deep_top)


# 분석 로직이 바뀔 때마다 이 버전을 올려서 기존 캐시를 무효화
ANALYZER_VERSION = "v24-2026-05-10-gist-storage"
if st.session_state.get("_analyzer_cache_version") != ANALYZER_VERSION:
    cached_analyze.clear()
    st.session_state["_analyzer_cache_version"] = ANALYZER_VERSION


def score_color(score: int) -> str:
    if score > 0:
        return "🟢"
    if score < 0:
        return "🔴"
    return "⚪"


def opinion_emoji(total: int) -> str:
    if total >= 2:
        return "✨"
    if total >= 0:
        return "🟡"
    if total >= -2:
        return "👀"
    return "⚠️"


SCORE_COLUMNS = ["추세", "모멘텀", "거래량", "가격 리스크", "수급", "공시", "가치", "재무 건전성", "성장성"]


def _cell_color(val) -> str:
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 2:
        return "background-color: rgba(46, 160, 67, 0.45); color: #0d3a1a; font-weight: 700"
    if v == 1:
        return "background-color: rgba(46, 160, 67, 0.22); color: #1f7a3a; font-weight: 600"
    if v == 0:
        return ""
    if v == -1:
        return "background-color: rgba(248, 81, 73, 0.22); color: #a3201a; font-weight: 600"
    if v <= -2:
        return "background-color: rgba(248, 81, 73, 0.45); color: #5a0e0a; font-weight: 700"
    return ""


def _fmt_signed(val) -> str:
    if pd.isna(val):
        return "-"
    try:
        return f"{int(val):+d}"
    except (TypeError, ValueError):
        return str(val)


def render_summary_table(results: list[dict]) -> None:
    rows = []
    for r in results:
        if r["error"]:
            row = {
                "종목": f"{r['name']} ({r['code']})",
                "현재가": "-",
                "등락률": "-",
                **{c: None for c in SCORE_COLUMNS},
                "종합점수": None,
                "의견": r["error"],
            }
            rows.append(row)
            continue

        score_map = {s["name"]: s["score"] for s in r["scores"]}
        fresh = (r.get("sources") or {}).get("fin_freshness") or {}
        stale_mark = " 📅" if fresh.get("is_stale") else ""
        row = {
            "종목": f"{r['name']} ({r['code']}){stale_mark}",
            "현재가": f"{r['last_close']:,.0f}원",
            "등락률": f"{r['change_pct']:+.2f}%",
            **{c: score_map.get(c) for c in SCORE_COLUMNS},
            "종합점수": r["total"],
            "의견": r["opinion"].split(" — ")[0],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    score_cols = [c for c in SCORE_COLUMNS if c in df.columns] + ["종합점수"]

    styled = (
        df.style
        .map(_cell_color, subset=score_cols)
        .format({c: _fmt_signed for c in score_cols}, na_rep="-")
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)


CATEGORY_META = {
    "critical": {"emoji": "🚨", "label": "중대 공시", "render": "error"},
    "negative": {"emoji": "⚠️", "label": "부정 공시", "render": "warning"},
    "positive": {"emoji": "✅", "label": "긍정 공시", "render": "success"},
    "neutral":  {"emoji": "📋", "label": "중립/정보 공시", "render": "neutral"},
}


def _render_one_disclosure(d: dict, brief: bool = False) -> None:
    title = (d.get("title") or "(제목 없음)").strip()
    url = d.get("url") or ""
    title_md = f"[{title}]({url})" if url else title
    submitter = d.get("submitter", "")
    summary = (d.get("summary") or "").strip()
    key_points = d.get("key_points") or []
    rationale = (d.get("rationale") or "").strip()
    model = d.get("model", "")

    if brief:
        st.markdown(
            f"- `{d['date']}` {title_md}  <small>· _{submitter}_</small>",
            unsafe_allow_html=True,
        )
        return

    badge = ""
    if model.startswith("gemini-2.5-pro"):
        badge = " 🧠"  # 본문 깊이 분석됨
    elif model.startswith("gemini-2.5-flash") or model.startswith("gemini-2.0-flash"):
        badge = " ⚡"  # 제목 분류만

    st.markdown(
        f"**`{d['date']}` {title_md}{badge}**  <small>_{submitter}_</small>",
        unsafe_allow_html=True,
    )
    if summary and summary != title:
        st.markdown(f"&nbsp;&nbsp;&nbsp;💡 {summary}", unsafe_allow_html=True)
    for pt in key_points:
        st.markdown(f"&nbsp;&nbsp;&nbsp;• {pt}", unsafe_allow_html=True)
    if rationale:
        st.markdown(
            f"&nbsp;&nbsp;&nbsp;<small>📌 _{rationale}_</small>",
            unsafe_allow_html=True,
        )


def _render_disclosures(disclosures: list[dict]) -> None:
    by_cat: dict[str, list[dict]] = {"critical": [], "negative": [], "positive": [], "neutral": []}
    for d in disclosures:
        by_cat.setdefault(d.get("category", "neutral"), by_cat["neutral"]).append(d)

    n_total = len(disclosures)
    n_important = len(by_cat["critical"]) + len(by_cat["negative"]) + len(by_cat["positive"])

    if n_important == 0:
        with st.expander(f"📋 최근 30일 DART 공시 — routine만 ({n_total}건)"):
            for d in disclosures:
                _render_one_disclosure(d, brief=True)
        return

    st.markdown("**📋 최근 30일 DART 공시**")

    for cat in ("critical", "negative", "positive"):
        items = by_cat[cat]
        if not items:
            continue
        meta = CATEGORY_META[cat]
        header = f"{meta['emoji']} {meta['label']} {len(items)}건"
        if meta["render"] == "error":
            st.error(header)
        elif meta["render"] == "warning":
            st.warning(header)
        else:
            st.success(header)
        with st.container():
            for d in items:
                _render_one_disclosure(d)

    if by_cat["neutral"]:
        with st.expander(f"📋 중립/정보 공시 {len(by_cat['neutral'])}건"):
            for d in by_cat["neutral"]:
                _render_one_disclosure(d, brief=True)


def render_stock_card(r: dict, favorites: list[str]) -> None:
    if r["error"]:
        st.error(f"**{r['name']} ({r['code']})** — {r['error']}")
        return

    is_fav = r["code"] in favorites
    src = r.get("sources") or {}
    fresh = src.get("fin_freshness") or {}
    is_stale = fresh.get("is_stale", False)

    with st.container(border=True):
        header_col, star_col, date_col = st.columns([5, 1, 2])
        with header_col:
            st.subheader(f"{r['name']}  `{r['code']}`")
        with star_col:
            star_label = "⭐" if is_fav else "☆"
            tooltip = "즐겨찾기 해제" if is_fav else "즐겨찾기 추가"
            if st.button(star_label, key=f"star_{r['code']}", help=tooltip):
                toggle_favorite(r["code"])
                st.rerun()
        with date_col:
            st.caption(f"{r['last_date']} 기준")

        if is_stale and src.get("fin_report_label"):
            days_old = fresh.get("days_since_coverage", 0)
            next_exp = fresh.get("next_expected", "")
            if src.get("preliminary_used_for_growth"):
                st.success(
                    f"📢 **잠정실적공시 반영됨** — 성장성 점수가 잠정실적 데이터로 갱신됐습니다. "
                    f"(정식 보고서: `{src['fin_report_label']}`, {days_old}일 경과)"
                )
            else:
                st.warning(
                    f"📅 **재무 데이터 갱신 가능성** — `{src['fin_report_label']}` "
                    f"기준 (회계기간 종료 후 {days_old}일 경과). "
                    f"**{next_exp}보고서**가 아직 DART에 미제출이라 가치/재무/성장성 점수는 "
                    f"이 시점 이후 변화를 못 잡습니다."
                )

        prelim = r.get("preliminary")
        if prelim and prelim.get("revenue") is not None:
            with st.expander(
                f"📢 잠정실적 — {prelim.get('period_label', '?')} "
                f"({prelim.get('disclosure_date', '')} 공시)"
            ):
                cols = st.columns(3)
                rev = prelim.get("revenue")
                rev_yoy = prelim.get("revenue_yoy")
                op = prelim.get("operating_income")
                op_yoy = prelim.get("operating_income_yoy")
                ni = prelim.get("net_income")
                ni_yoy = prelim.get("net_income_yoy")

                def _fmt_money(v):
                    if v is None:
                        return "—"
                    sign = "▼ " if v < 0 else ""
                    if abs(v) >= 1_000_000_000_000:
                        return f"{sign}{v / 1_000_000_000_000:.2f}조원"
                    if abs(v) >= 100_000_000:
                        return f"{sign}{v / 100_000_000:.0f}억원"
                    return f"{sign}{v:,}원"

                def _fmt_growth(curr, prev):
                    if curr is None or prev in (None, 0):
                        return ""
                    pct = (curr / prev - 1) * 100 if prev != 0 else 0
                    if prev < 0 and curr >= 0:
                        return " (흑자전환)"
                    if prev >= 0 and curr < 0:
                        return " (적자전환)"
                    if prev < 0 and curr < 0 and curr > prev:
                        return " (적자축소)"
                    return f" ({pct:+.1f}%)"

                cols[0].metric("매출", _fmt_money(rev), delta=_fmt_growth(rev, rev_yoy) or None)
                cols[1].metric("영업이익", _fmt_money(op), delta=_fmt_growth(op, op_yoy) or None)
                cols[2].metric("당기순이익", _fmt_money(ni), delta=_fmt_growth(ni, ni_yoy) or None)

                st.caption(
                    f"_{prelim.get('disclosure_title', '')}_  \n"
                    "⚠️ 잠정실적 = 회사 자체 발표, 외부 감사 전. 정식 보고서와 차이 가능."
                )

        m1, m2, m3 = st.columns(3)
        m1.metric("현재가", f"{r['last_close']:,.0f}원", f"{r['change_pct']:+.2f}%")
        m2.metric("종합점수", f"{r['total']:+d}점")
        m3.metric("의견", f"{opinion_emoji(r['total'])} {r['opinion'].split(' — ')[0]}")

        # 보유 종목이면 손익 badge
        portfolio_local = port.load_portfolio()
        if r["code"] in portfolio_local:
            h = portfolio_local[r["code"]]
            pnl = port.compute_pnl(h, r["last_close"])
            color = "#1f7a3a" if pnl["profit"] >= 0 else "#a3201a"
            st.markdown(
                f"<div style='padding:8px 12px;border-radius:6px;"
                f"background:rgba(46,160,67,0.08);margin-bottom:8px'>"
                f"💼 <b>보유 중</b>: {h['quantity']:,}주 @ {h['avg_price']:,.0f}원 "
                f"→ 평가액 <b>{pnl['market_value']:,.0f}원</b> · "
                f"<span style='color:{color};font-weight:600'>"
                f"손익 {pnl['profit']:+,.0f}원 ({pnl['profit_pct']:+.2f}%)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if r.get("history"):
            hist = pd.DataFrame(r["history"]).set_index("Date")
            st.markdown("**최근 120일 종가 / 20일선 / 60일선**")
            st.line_chart(hist, height=240)

        # 점수 히스토리 (30일 추세) + 해당 분석일 주가
        score_hist = hist_module.get_history(r["code"], days=30)
        st.markdown("**📊 최근 30일 종합점수 / 주가 추세**")
        if len(score_hist) >= 2:
            sh_df = pd.DataFrame(score_hist).copy()

            sh_df["date"] = pd.to_datetime(sh_df["date"], errors="coerce")
            sh_df["종합점수"] = pd.to_numeric(sh_df["total"], errors="coerce")
            if "close" in sh_df.columns:
                sh_df["주가"] = pd.to_numeric(sh_df["close"], errors="coerce")
            else:
                sh_df["주가"] = pd.NA

            sh_df = sh_df.dropna(subset=["date", "종합점수"]).sort_values("date")

            score_line = (
                alt.Chart(sh_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="분석일"),
                    y=alt.Y(
                        "종합점수:Q",
                        title="종합점수",
                        axis=alt.Axis(title="종합점수", orient="left"),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"),
                        alt.Tooltip("종합점수:Q", title="종합점수", format="+.0f"),
                    ],
                )
            )

            price_df = sh_df.dropna(subset=["주가"])
            if not price_df.empty:
                price_line = (
                    alt.Chart(price_df)
                    .mark_line(point=True, strokeDash=[5, 4])
                    .encode(
                        x=alt.X("date:T", title="분석일"),
                        y=alt.Y(
                            "주가:Q",
                            title="주가(원)",
                            axis=alt.Axis(title="주가(원)", orient="right", format=","),
                            scale=alt.Scale(zero=False),
                        ),
                        tooltip=[
                            alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"),
                            alt.Tooltip("주가:Q", title="주가", format=",.0f"),
                        ],
                    )
                )

                chart = (
                    alt.layer(score_line, price_line)
                    .resolve_scale(y="independent")
                    .properties(height=220)
                )
                st.altair_chart(chart, use_container_width=True)
                st.caption("실선은 종합점수, 점선은 해당 분석일 종가입니다. 왼쪽 축은 점수, 오른쪽 축은 주가입니다.")
            else:
                st.line_chart(sh_df.set_index("date")[["종합점수"]], height=160)
                st.caption("기존 기록에 주가가 없어 종합점수만 표시합니다. 다음 분석부터 주가가 함께 표시됩니다.")

            trend = hist_module.compute_trend(r["code"], days=30)
            if trend:
                delta = trend["delta"]
                if delta > 0:
                    st.caption(
                        f"📈 {trend['first_score']:+d} → {trend['last_score']:+d} "
                        f"({delta:+d}점 상승 · {trend['days_recorded']}회 분석 기록)"
                    )
                elif delta < 0:
                    st.caption(
                        f"📉 {trend['first_score']:+d} → {trend['last_score']:+d} "
                        f"({delta:+d}점 하락 · {trend['days_recorded']}회 분석 기록)"
                    )
                else:
                    st.caption(f"→ 변화 없음 · {trend['days_recorded']}회 분석 기록")
        elif len(score_hist) == 1:
            close_text = ""
            if score_hist[0].get("close") is not None:
                close_text = f", 주가 {score_hist[0]['close']:,.0f}원"
            st.info(
                f"📌 오늘 분석 1회 기록됨 (`{score_hist[0]['date']}`, "
                f"{score_hist[0]['total']:+d}점{close_text}). **내일 한 번 더 분석하면 추세 차트가 표시됩니다.**"
            )
        else:
            st.caption(
                "분석할 때마다 자동으로 종합점수와 주가가 기록됩니다. "
                "2일 이상 누적되면 추세 차트가 표시됩니다."
            )

        st.markdown("**항목별 분석**")
        score_rows = []
        for s in r["scores"]:
            score_rows.append({
                "": score_color(s["score"]),
                "항목": s["name"],
                "분석": s["msg"],
                "점수": s["score"],
            })
        score_df = pd.DataFrame(score_rows)
        st.dataframe(
            score_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "점수": st.column_config.NumberColumn(format="%+d", width="small"),
                "": st.column_config.TextColumn(width="small"),
            },
        )

        st.info(f"💡 {r['opinion']}")

        disclosures = r.get("disclosures") or []
        if disclosures:
            _render_disclosures(disclosures)

        news = r.get("news") or []
        has_dart_or_naver = (r.get("sources") or {}).get("price_last_date")
        if news:
            with st.expander(f"📰 최근 종목 뉴스 ({len(news)}건) — 참고용"):
                st.caption(
                    "⚠️ 뉴스는 **점수에 영향 없습니다**. 종목명이 제목에 직접 포함된 기사만 표시됩니다."
                )
                for n in news:
                    title = (n.get("title") or "").strip() or "(제목 없음)"
                    url = n.get("url") or ""
                    source = n.get("source") or ""
                    date = n.get("date") or ""
                    title_md = f"[{title}]({url})" if url else title
                    st.markdown(
                        f"- `{date}` {title_md}  <small>· _{source}_</small>",
                        unsafe_allow_html=True,
                    )
        elif has_dart_or_naver:
            with st.expander("📰 최근 종목 뉴스 — 0건"):
                st.caption(
                    f"제목에 **{r['name']}**(또는 단축형)이 직접 들어간 최근 기사를 찾지 못했습니다.  \n"
                    "산업·시장 일반 뉴스는 노이즈가 많아 자동 제외됩니다. "
                    "네이버 금융 종목뉴스 페이지에서 직접 확인하시려면 종목코드 검색을 활용하세요."
                )

        src = r.get("sources") or {}
        if src:
            with st.expander("ℹ️ 데이터 출처와 기준 시점"):
                lines = []
                if src.get("price_last_date"):
                    lines.append(
                        f"- **추세 / 모멘텀 / 거래량 / 가격 리스크** — "
                        f"FinanceDataReader (KRX 일봉) · 마지막 거래일 `{src['price_last_date']}`"
                    )
                if src.get("flow_last_date"):
                    lines.append(
                        f"- **외국인·기관 수급** — 네이버 금융 (HTML 스크래핑) · "
                        f"마지막 거래일 `{src['flow_last_date']}`"
                    )
                elif "수급" in [s["name"] for s in r["scores"]]:
                    lines.append("- **외국인·기관 수급** — 네이버 금융 (조회 실패 또는 데이터 없음)")
                if src.get("fin_report_label"):
                    fresh_note = ""
                    if is_stale:
                        days_old = fresh.get("days_since_coverage", 0)
                        next_exp = fresh.get("next_expected", "")
                        fresh_note = (
                            f"  \n  ⚠️ 회계기간 종료 후 **{days_old}일 경과** — "
                            f"**{next_exp}보고서** 미제출 상태"
                        )
                    lines.append(
                        f"- **가치 (PER/PBR) / 재무 건전성 / 성장성** — "
                        f"DART OPEN API · `{src['fin_report_label']}` 기준"
                        f"{fresh_note}"
                    )
                elif src.get("has_dart"):
                    lines.append("- **재무 데이터** — DART (보고서 조회 실패)")
                if src.get("has_dart") and disclosures:
                    lines.append("- **공시 목록·분류** — DART OPEN API · 최근 30일 접수분 (Gemini 분류)")
                if src.get("news_count"):
                    lines.append(
                        f"- **종목 뉴스** — 네이버 금융 종목뉴스 (HTML 스크래핑) · "
                        f"최근 {src['news_count']}건 · 점수 영향 없음 (참고용)"
                    )
                if src.get("analyzed_at"):
                    lines.append(f"- **이 분석을 실행한 시각** — `{src['analyzed_at']}`")
                st.markdown("\n".join(lines))
                st.caption(
                    "⚠️ 네이버 금융 스크래핑은 약관상 회색지대입니다. "
                    "본인 분석용으로만 사용하시고, 배포·상업화 단계에선 한투 OpenAPI 등 "
                    "공식 데이터로 교체를 권장합니다."
                )


# 메인 영역

# 1) 스크리닝 후보 전용 뷰 모드 (사이드바 '📌 최근 스크리닝 후보' 버튼 클릭 시)
if st.session_state.get("_view_mode") == "screening_history":
    st.subheader("📌 최근 스크리닝 후보 (90일 누적)")

    nav_back, nav_refresh = st.columns([1, 1])
    with nav_back:
        if st.button("← 분석 화면으로 돌아가기", key="back_to_analysis", use_container_width=True):
            st.session_state.pop("_view_mode", None)
            st.rerun()

    recent_picks = screening_history.get_recent(days=90)
    if recent_picks:
        with nav_refresh:
            if st.button(
                f"🔄 모두 다시 분석 ({len(recent_picks)}개 점수 업데이트)",
                key="refresh_picks_btn",
                use_container_width=True,
                type="primary",
                help="오늘 시세·재무·공시로 추적 중인 종목 모두 풀 분석. ~분 단위 소요.",
            ):
                st.session_state["_run_refresh_picks"] = True
                st.rerun()

    # 추적 종목 다시 분석 (사용자가 위 버튼 누른 경우)
    if st.session_state.pop("_run_refresh_picks", False) and recent_picks:
        codes_to_refresh = list(recent_picks.keys())
        n_codes = len(codes_to_refresh)

        # 캐시 초기화 — 진짜 fresh 데이터 받게
        cached_analyze.clear()

        from analyzer import (
            enrich_with_deep_analysis,
            recompute_score_after_deep,
        )

        st.caption(f"1단계: {n_codes}개 종목 재분석")
        p1 = st.progress(0)
        s1 = st.empty()
        refreshed: list[dict] = []
        errs = 0

        # 히스토리 배치 — 한 번에 저장
        hist_module.begin_batch()
        try:
            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = {ex.submit(cached_analyze, c, False, 0): c for c in codes_to_refresh}
                done = 0
                for f in as_completed(futs):
                    c = futs[f]
                    try:
                        r = f.result()
                        if not r.get("error"):
                            refreshed.append(r)
                        else:
                            errs += 1
                    except Exception:
                        errs += 1
                    done += 1
                    p1.progress(done / n_codes)
                    s1.caption(f"{done}/{n_codes}: {stock_dict.get(c, c)} ({c})")

            # 2단계: 깊이 분석 + 점수 재계산
            if refreshed:
                s1.caption(f"2단계: {len(refreshed)}개 정밀 분석 중")
                p2 = st.progress(0)

                def _enrich(r):
                    enrich_with_deep_analysis(r, top_n=3)
                    recompute_score_after_deep(r)

                with ThreadPoolExecutor(max_workers=3) as dex:
                    dfuts = {dex.submit(_enrich, r): r for r in refreshed}
                    d_done = 0
                    for f in as_completed(dfuts):
                        try:
                            f.result()
                        except Exception:
                            pass
                        d_done += 1
                        p2.progress(d_done / len(refreshed))
                p2.empty()
        finally:
            try:
                hist_module.commit_batch()
            except Exception:
                pass

        p1.empty()
        s1.empty()
        st.success(
            f"✅ {len(refreshed)}개 점수 업데이트 완료 "
            f"(에러 {errs}건). 표가 새 점수로 갱신됐습니다."
        )
        # 표 그리기 위해 recent_picks 다시 로드 (record_snapshot이 score_history 갱신했음)
        recent_picks = screening_history.get_recent(days=90)

    if not recent_picks:
        st.info("아직 누적된 스크리닝 결과가 없습니다. 사이드바 '🔍 발굴 시작' 눌러주세요.")
    else:
        # 현재 안전 유니버스 + 점수 히스토리
        try:
            safe_codes = set(get_universe_codes("safe"))
        except Exception:
            safe_codes = set()
        all_history = hist_module._load()
        latest_date = next(iter(recent_picks.values())).get("latest_date", "")

        # 요약 통계
        n_total = len(recent_picks)
        n_active = sum(1 for m in recent_picks.values() if m["in_latest"])
        n_dropped = n_total - n_active
        n_uni_out = sum(1 for c in recent_picks if c not in safe_codes)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("누적 후보", f"{n_total}개")
        c2.metric("최근 등장", f"{n_active}개", help=f"{latest_date} 스크리닝에 통과")
        c3.metric("탈락", f"{n_dropped}개", delta=None if n_dropped == 0 else f"-{n_dropped}",
                  delta_color="inverse")
        c4.metric("유니버스 이탈", f"{n_uni_out}개", delta=None if n_uni_out == 0 else "⚠️",
                  delta_color="inverse")

        # 필터
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            status_filter = st.radio(
                "상태",
                options=["전체", "현역만", "탈락만"],
                horizontal=True,
                key="sh_status_filter",
            )
        with f_col2:
            sort_key = st.radio(
                "정렬",
                options=["최근 등장순", "등장 횟수순", "점수순", "전일대비%순", "추적기간%순"],
                horizontal=True,
                key="sh_sort_key",
            )

        # 행 데이터 빌드
        rows = []
        code_order = []
        for code, meta in recent_picks.items():
            in_universe = code in safe_codes
            in_latest = meta["in_latest"]

            if status_filter == "현역만" and not in_latest:
                continue
            if status_filter == "탈락만" and in_latest:
                continue

            entries = all_history.get(code, [])
            latest_score = entries[-1]["total"] if entries else None
            latest_close = entries[-1].get("close") if entries else None
            prev_score = entries[-2]["total"] if len(entries) >= 2 else None
            prev_close = entries[-2].get("close") if len(entries) >= 2 else None
            score_delta = (
                latest_score - prev_score
                if (latest_score is not None and prev_score is not None)
                else None
            )
            price_delta_pct = (
                (latest_close / prev_close - 1) * 100
                if latest_close and prev_close
                else None
            )

            # 추적 시작 시점 주가 찾기 (첫 등장일 또는 가장 가까운 entry)
            first_seen = meta.get("first_seen", "")
            first_close = None
            for e in entries:
                if e.get("date") == first_seen:
                    first_close = e.get("close")
                    break
            # 첫 등장일에 종가 데이터 없으면 가장 오래된 entry 사용
            if first_close is None and entries:
                first_close = entries[0].get("close")
            since_first_pct = (
                (latest_close / first_close - 1) * 100
                if latest_close and first_close
                else None
            )

            name = stock_dict.get(code, code)
            mark = ""
            if not in_latest:
                mark = "⚠️탈락"
            if not in_universe:
                mark = (mark + " " if mark else "") + "⚠️유니버스이탈"

            rows.append({
                "종목": f"{name} ({code}){' ' + mark if mark else ''}",
                "점수": latest_score if latest_score is not None else "-",
                "점수변화": score_delta if score_delta is not None else "-",
                "주가(원)": latest_close if latest_close is not None else "-",
                "전일대비%": (
                    round(price_delta_pct, 2) if price_delta_pct is not None else "-"
                ),
                "추적기간%": (
                    round(since_first_pct, 2) if since_first_pct is not None else "-"
                ),
                "유니버스": "✅" if in_universe else "⚠️ 이탈",
                "등장": meta["count"],
                "첫등장": meta["first_seen"],
                "마지막": meta["last_seen"],
                "상태": "현역" if in_latest else "탈락",
            })
            code_order.append(code)

        # 정렬
        def _sort_key(idx_row):
            i, r = idx_row
            v = r.get({
                "최근 등장순": "마지막",
                "등장 횟수순": "등장",
                "점수순": "점수",
                "전일대비%순": "전일대비%",
                "추적기간%순": "추적기간%",
            }[sort_key])
            if v in ("-", None):
                return (1, 0)
            return (0, -v if isinstance(v, (int, float)) else v)

        indexed = list(enumerate(rows))
        indexed.sort(key=_sort_key)
        rows = [r for _, r in indexed]
        code_order = [code_order[i] for i, _ in indexed]

        if not rows:
            st.info("필터 조건에 맞는 종목이 없습니다.")
        else:
            df = pd.DataFrame(rows)
            event = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "점수": st.column_config.NumberColumn(format="%+d"),
                    "점수변화": st.column_config.NumberColumn(format="%+d"),
                    "주가(원)": st.column_config.NumberColumn(format="%,.0f"),
                    "전일대비%": st.column_config.NumberColumn(
                        format="%+.2f%%",
                        help="가장 최근 분석과 그 직전 분석 사이의 종가 변화",
                    ),
                    "추적기간%": st.column_config.NumberColumn(
                        format="%+.2f%%",
                        help="추적 시작(첫 등장)일 종가 대비 현재 종가 변화",
                    ),
                },
            )
            st.caption("종목 행을 클릭하면 해당 종목 단독 분석으로 이동합니다.")

            # 행 클릭 → 분석 화면
            if event and event.selection.rows:
                sel_idx = event.selection.rows[0]
                if sel_idx < len(code_order):
                    st.session_state["_focus_code"] = code_order[sel_idx]
                    st.session_state.pop("_view_mode", None)
                    st.rerun()

    st.stop()  # 이 뷰만 보여주고 아래 일반 분석 흐름은 실행 안 함


focus_code = st.session_state.pop("_focus_code", None)
analyze_favs_only = st.session_state.pop("_analyze_favs_only", False)
screen_req = st.session_state.pop("_screen", None)
auto_run = st.session_state.pop("_auto_run_analyze", False)

results = None
if screen_req:
    universe = screen_req["universe"]
    min_score = screen_req["min_score"]
    use_lite = False  # 항상 풀 모드 (lite 모드 제거됨)
    codes = get_universe_codes(universe)
    label = UNIVERSE_LABELS.get(universe, universe)

    st.subheader(f"🔍 발굴 결과 — {label}")
    if not codes:
        st.error("유니버스 종목 목록을 가져오지 못했습니다.")
    else:
        st.caption("1단계: 전체 분석 (공시 분류 + 잠정실적 + 뉴스)")
        progress = st.progress(0)
        status = st.empty()
        screened: list[dict] = []
        errors = 0
        max_workers = 3
        completed = 0

        # 스크리닝 동안 점수 히스토리를 배치 모드로 — N개 분석마다 N번 Gist 저장하는
        # 대신 끝에 1번만 저장. 동시 호출로 인한 멈춤·리셋 방지.
        hist_module.begin_batch()
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 스크리닝은 깊이 분석 스킵 (deep_top=0) — 종목당 10초 이상 절감
                futures = {
                    executor.submit(cached_analyze, c, use_lite, 0): c
                    for c in codes
                }
                for future in as_completed(futures):
                    c = futures[future]
                    try:
                        r = future.result()
                        if not r.get("error"):
                            screened.append(r)
                        else:
                            errors += 1
                    except Exception:
                        errors += 1
                    completed += 1
                    progress.progress(completed / len(codes))
                    nm = stock_dict.get(c, c)
                    status.caption(f"분석 중 ({completed}/{len(codes)}): {nm} ({c})")
        finally:
            # 정상 종료든 예외든 항상 누적된 히스토리를 한 번에 저장
            try:
                hist_module.commit_batch()
            except Exception:
                pass

        progress.empty()
        status.empty()

        # 1) 재무 데이터 신선도 필터 — stale + 잠정실적 없음 = 제외
        excluded_stale = 0
        fresh_results = []
        for r in screened:
            src = r.get("sources") or {}
            is_stale = (src.get("fin_freshness") or {}).get("is_stale", False)
            has_preliminary = bool(r.get("preliminary"))
            if is_stale and not has_preliminary:
                excluded_stale += 1
                continue
            fresh_results.append(r)

        # 2) 종합점수 내림차순 + 최소 점수 필터
        fresh_results.sort(key=lambda r: r.get("total", -999), reverse=True)
        results = [r for r in fresh_results if r.get("total", -999) >= min_score]
        excluded_score = len(fresh_results) - len(results)

        # 2단계: 통과 종목들에 깊이 분석 추가 + 점수 재계산
        pre_deep_count = len(results)
        if results:
            from analyzer import enrich_with_deep_analysis, recompute_score_after_deep
            st.caption(f"2단계: 통과 {len(results)}개 종목 정밀 분석 중...")
            deep_progress = st.progress(0)

            def _enrich_and_recompute(r: dict) -> None:
                enrich_with_deep_analysis(r, top_n=3)
                recompute_score_after_deep(r)

            with ThreadPoolExecutor(max_workers=3) as deep_exec:
                deep_futures = {
                    deep_exec.submit(_enrich_and_recompute, r): r for r in results
                }
                deep_done = 0
                for future in as_completed(deep_futures):
                    try:
                        future.result()
                    except Exception:
                        pass
                    deep_done += 1
                    deep_progress.progress(deep_done / len(results))
            deep_progress.empty()

            # 깊이 분석 결과로 점수가 떨어진 종목 제거 (재필터)
            results = [r for r in results if r.get("total", -999) >= min_score]
            results.sort(key=lambda r: r.get("total", -999), reverse=True)
            dropped_after_deep = pre_deep_count - len(results)

            # 스크리닝 히스토리 자동 저장 — 깊이 분석 후 통과 종목만
            try:
                screening_history.record_today([r["code"] for r in results])
            except Exception:
                pass
        else:
            dropped_after_deep = 0

        if results:
            # 정상: 통과 종목 있음
            msg = (
                f"✅ {len(codes)}개 1차 분석 · **{len(results)}개 최종 통과** "
                f"(점수 ≥ {min_score:+d}, 에러 {errors}건"
            )
            if excluded_stale > 0:
                msg += f", stale 제외 {excluded_stale}"
            if dropped_after_deep > 0:
                msg += f", 정밀 분석 후 탈락 {dropped_after_deep}"
            msg += ")."
            st.success(msg)
            st.info(
                f"📌 최종 통과 {len(results)}개가 사이드바 **'최근 스크리닝 후보'** 에 "
                "자동 추가됐습니다. 매일 돌리면 점수 변화·주가 추이·이탈이 추적돼요."
            )

        else:
            # 빈 결과 — 명확히 안내 + 원인 설명 + 다음 액션 제안
            top_score = max(
                (r.get("total", -999) for r in fresh_results), default=None,
            )
            st.warning(
                f"🚫 **조건을 통과한 종목이 없습니다** "
                f"({len(codes)}개 분석)"
            )
            with st.container(border=True):
                st.markdown("#### 📊 발굴 내역")
                lines = [
                    f"- 분석 시도: **{len(codes)}개**",
                    f"- 에러: {errors}개",
                ]
                if excluded_stale > 0:
                    lines.append(
                        f"- 재무 stale + 잠정실적 없음 자동 제외: **{excluded_stale}개**"
                    )
                lines.append(
                    f"- 점수 {min_score:+d}점 미만으로 탈락: **{excluded_score}개**"
                )
                if dropped_after_deep > 0:
                    lines.append(
                        f"- 정밀 분석 후 점수 미달로 추가 탈락: **{dropped_after_deep}개**"
                    )
                if top_score is not None:
                    lines.append(f"- 통과 후보 중 최고 점수: **{top_score:+d}점**")
                st.markdown("\n".join(lines))

                st.markdown("#### 💡 다음 시도")
                st.markdown(
                    f"- 사이드바에서 **최소 종합점수를 낮춰**보세요 "
                    f"(현재 `{min_score:+d}`, 추천 `{max(top_score, 0) if top_score is not None else 0:+d}` 정도부터)\n"
                    "- 더 넓은 유니버스(KOSPI TOP 50 등) 선택\n"
                    "- 5/15 이후 1Q 보고서가 등록되면 stale 제외 종목들도 후보에 다시 들어옵니다"
                )
        st.session_state.results = results

elif focus_code:
    name = stock_dict.get(focus_code, focus_code)
    with st.spinner(f"{name} 분석 중..."):
        results = [cached_analyze(focus_code)]
        st.session_state.results = results
    st.info(f"⭐ **{name}** 단독 분석 결과입니다. 워치리스트 전체를 보려면 **🔍 분석하기**를 누르세요.")
elif analyze_favs_only:
    favs = load_favorites()
    if not favs:
        st.warning("⭐ 즐겨찾기가 비어있습니다. 종목 카드의 ☆ 버튼으로 추가하세요.")
    else:
        with st.spinner(f"즐겨찾기 {len(favs)}개 분석 중..."):
            results = [cached_analyze(c) for c in favs]
            st.session_state.results = results
        st.info(
            f"⭐ 즐겨찾기 **{len(favs)}개**만 분석한 결과입니다. "
            "워치리스트는 그대로이며, 워치리스트 전체를 보려면 **🔍 분석하기**를 누르세요."
        )
elif not selected_codes:
    st.info("👈 사이드바에서 분석할 종목을 선택하세요.")
elif run_analysis or auto_run:
    with st.spinner("종목 데이터 수집 및 점수 계산 중..."):
        results = [cached_analyze(c) for c in selected_codes]
        st.session_state.results = results
elif "results" in st.session_state:
    results = st.session_state.results
else:
    st.info(f"선택된 종목: {len(selected_codes)}개. **분석하기** 버튼을 눌러 시작하세요.")

if results:
    if len(results) > 1:
        st.subheader("📊 비교표")
        render_summary_table(results)
        st.divider()

    st.subheader("📋 종목별 상세 분석")
    favorites = load_favorites()
    for r in results:
        render_stock_card(r, favorites)
