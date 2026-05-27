"""
주식 분석 리포트 - Streamlit 웹 앱
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

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
import scouted


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

    # 실제로 변경됐을 때만 저장 — 매 rerun 호출되면 GitHub API rate limit 소진됨
    # (st_autorefresh 폴링이 2초마다 rerun 유발하므로 무조건 호출하면 분당 30회 PATCH)
    if selected_codes != saved_codes:
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

    # ─────────── 점수 시뮬레이션 진입점 ───────────
    if st.button(
        "📊 점수 시뮬레이션 — 발굴 종목 가상 수익률",
        key="open_verifier",
        use_container_width=True,
        help="스크리닝에서 발굴된 종목을 그 시점에 가상 매수했다고 가정했을 때의 수익률",
    ):
        st.session_state["_view_mode"] = "verifier"
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

    # 시간 추정 — KRX 종목 목록 서버가 불안정해도 앱 전체가 멈추지 않도록 보호
    try:
        est_codes = len(get_universe_codes(universe))
    except Exception:
        est_codes = 0

    if est_codes > 0:
        est_pass1_sec = est_codes * 8 / 3
        est_min = max(1, round(est_pass1_sec / 60))
        st.caption(
            f"⏱️ 1차 스크리닝 약 {est_min}분 예상 "
            f"({est_codes}개 ÷ 3병렬). 통과 종목엔 추가 깊이 분석 자동 실행."
        )
    else:
        st.caption(
            "⏱️ 현재 KRX 종목 목록을 불러오지 못해 예상 시간을 계산하지 못했습니다. "
            "관심 종목 분석은 계속 사용할 수 있고, 종목 발굴은 잠시 후 다시 시도하세요."
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


def _pct(done: int, total: int) -> int:
    if total <= 0:
        return 100
    return max(0, min(100, round(done / total * 100)))


CUTE_LOADING_ARTS = [
    {
        "name": "곰돌이",
        "grid": [
            ["▫️", "▫️", "🟫", "▫️", "▫️", "🟫", "▫️", "▫️"],
            ["▫️", "🟫", "🟫", "🟫", "🟫", "🟫", "🟫", "▫️"],
            ["🟫", "⬛", "🟫", "🟫", "🟫", "🟫", "⬛", "🟫"],
            ["🟫", "🟫", "🟫", "⬛", "⬛", "🟫", "🟫", "🟫"],
            ["🟫", "🩷", "🟫", "🟫", "🟫", "🟫", "🩷", "🟫"],
            ["🟫", "🟫", "🟫", "🟨", "🟨", "🟫", "🟫", "🟫"],
            ["▫️", "🟫", "🟫", "🟫", "🟫", "🟫", "🟫", "▫️"],
            ["▫️", "▫️", "🟫", "▫️", "▫️", "🟫", "▫️", "▫️"],
        ],
    },
    {
        "name": "병아리",
        "grid": [
            ["▫️", "▫️", "🟨", "🟨", "🟨", "🟨", "▫️", "▫️"],
            ["▫️", "🟨", "🟨", "🟨", "🟨", "🟨", "🟨", "▫️"],
            ["🟨", "⬛", "🟨", "🟨", "🟨", "🟨", "⬛", "🟨"],
            ["🟨", "🟨", "🟨", "🟧", "🟧", "🟨", "🟨", "🟨"],
            ["🟨", "🩷", "🟨", "🟨", "🟨", "🟨", "🩷", "🟨"],
            ["▫️", "🟨", "🟨", "🟨", "🟨", "🟨", "🟨", "▫️"],
            ["▫️", "▫️", "🟨", "🟨", "🟨", "🟨", "▫️", "▫️"],
            ["▫️", "▫️", "🟧", "▫️", "▫️", "🟧", "▫️", "▫️"],
        ],
    },
    {
        "name": "토끼",
        "grid": [
            ["▫️", "▫️", "🐰", "▫️", "▫️", "🐰", "▫️", "▫️"],
            ["▫️", "▫️", "🐰", "▫️", "▫️", "🐰", "▫️", "▫️"],
            ["▫️", "🐰", "🐰", "🐰", "🐰", "🐰", "🐰", "▫️"],
            ["🐰", "⬛", "🐰", "🐰", "🐰", "🐰", "⬛", "🐰"],
            ["🐰", "🐰", "🐰", "⬛", "⬛", "🐰", "🐰", "🐰"],
            ["🐰", "🩷", "🐰", "🐰", "🐰", "🐰", "🩷", "🐰"],
            ["▫️", "🐰", "🐰", "🟨", "🟨", "🐰", "🐰", "▫️"],
            ["▫️", "▫️", "🐰", "🐰", "🐰", "🐰", "▫️", "▫️"],
        ],
    },
]


def _pick_loading_art(seed: str):
    seed = seed or "loading"
    idx = sum(ord(ch) for ch in seed) % len(CUTE_LOADING_ARTS)
    return CUTE_LOADING_ARTS[idx]


def _render_loading_art(art_box, percent: int, label: str, detail: str = "", art_seed: str = "loading") -> None:
    if art_box is None:
        return

    art = _pick_loading_art(art_seed)
    grid = art["grid"]
    total_cells = sum(len(row) for row in grid)
    reveal_count = round(total_cells * max(0, min(100, percent)) / 100)

    non_bg = []
    bg = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "▫️":
                bg.append((r, c))
            else:
                non_bg.append((r, c))
    reveal_order = non_bg + bg
    visible = set(reveal_order[:reveal_count])
    covered = "🟪"

    rendered_rows = []
    for r, row in enumerate(grid):
        rendered_rows.append(" ".join(cell if (r, c) in visible else covered for c, cell in enumerate(row)))

    sub = f"{percent}% 완성"
    if detail:
        sub += f" · {detail}"
    title = f"🎨 로딩중... {art['name']}가 나타나는 중"
    html = f"""
    <div style="padding:0.85rem 1rem; margin:0.35rem 0 0.2rem 0; border:1px solid #f0e6ff; border-radius:16px; background:linear-gradient(135deg,#fff8fd,#f7f9ff);">
        <div style="font-weight:700; font-size:0.98rem; margin-bottom:0.35rem;">{title}</div>
        <div style="font-size:1.38rem; line-height:1.1; letter-spacing:0.02rem;">{'<br>'.join(rendered_rows)}</div>
        <div style="margin-top:0.45rem; color:#666; font-size:0.86rem;">{label} · {sub}</div>
    </div>
    """
    art_box.markdown(html, unsafe_allow_html=True)


def _progress_update(
    progress_bar,
    status_box,
    done: int,
    total: int,
    label: str,
    detail: str = "",
    art_box=None,
    art_seed: str = "loading",
) -> None:
    percent = _pct(done, total)
    progress_bar.progress(done / total if total else 1.0)
    msg = f"{label} {percent}% ({done}/{total})"
    if detail:
        msg += f" · {detail}"
    status_box.caption(msg)
    _render_loading_art(art_box, percent, label, detail, art_seed)


# 분석 로직이 바뀔 때마다 이 버전을 올려서 기존 캐시를 무효화
ANALYZER_VERSION = "v28-2026-05-27-secrets-worker-thread-cache"
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


SCORE_COLUMNS = [
    "추세", "모멘텀", "거래량", "가격 리스크",
    "시장 상대강도", "수급", "공시",
    "가치", "재무 건전성", "성장성",
]


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
                "단기": None,
                "중기": None,
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
            "단기": r.get("short_term_score"),
            "중기": r.get("mid_term_score"),
            "종합점수": r["total"],
            "의견": r["opinion"].split(" — ")[0],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    score_cols = [c for c in SCORE_COLUMNS if c in df.columns] + ["단기", "중기", "종합점수"]

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


def _compact_score_reasons(r: dict, positive: bool = True, limit: int = 3) -> str:
    """분석 결과의 항목별 점수에서 신규/탈락 이유로 보여줄 핵심 문장 생성."""
    scores = r.get("scores") or []
    if positive:
        picked = [x for x in scores if isinstance(x, dict) and x.get("score", 0) > 0]
        picked.sort(key=lambda x: x.get("score", 0), reverse=True)
    else:
        picked = [x for x in scores if isinstance(x, dict) and x.get("score", 0) < 0]
        picked.sort(key=lambda x: x.get("score", 0))

    parts = []
    for item in picked[:limit]:
        name = item.get("name", "항목")
        score = item.get("score", 0)
        msg = str(item.get("msg", "")).strip()
        if len(msg) > 34:
            msg = msg[:34].rstrip() + "…"
        parts.append(f"{name}({score:+d}: {msg})" if msg else f"{name}({score:+d})")
    return ", ".join(parts)


def _build_pass_reason(r: dict, min_score: int | None = None) -> str:
    total = r.get("total")
    threshold = f"기준 {min_score:+d}점 이상" if min_score is not None else "기준 통과"
    positive = _compact_score_reasons(r, positive=True, limit=3)
    negative = _compact_score_reasons(r, positive=False, limit=2)
    base = f"종합점수 {total:+d}점으로 {threshold}" if isinstance(total, int) else threshold
    if positive:
        base += f". 긍정 요인: {positive}"
    if negative:
        base += f". 주의 요인: {negative}"
    return base


def _build_drop_reason(r: dict, min_score: int | None = None, kind: str = "score") -> str:
    total = r.get("total")
    negative = _compact_score_reasons(r, positive=False, limit=3)
    positive = _compact_score_reasons(r, positive=True, limit=2)

    if kind == "stale":
        base = "재무 데이터가 오래됐고 잠정실적공시도 없어 스크리닝에서 제외"
    elif kind == "surge":
        surge_info = r.get("recent_surge") or {}
        triggers = surge_info.get("triggers") or []
        is_fundamental = surge_info.get("fundamental_backed_out", False)
        if triggers:
            if is_fundamental:
                base = (
                    f"펀더멘털 없는 급등 감지({', '.join(triggers)}) — "
                    "실적 뒷받침 없는 단기 상승으로 자동 제외"
                )
            else:
                base = (
                    f"발굴 시점 직전 급등 감지({', '.join(triggers)}) — "
                    "고점 추격 회피용 자동 제외"
                )
        else:
            base = "발굴 시점 직전 단기 급등 — 평균 회귀 위험으로 자동 제외"
    elif kind == "deep":
        if isinstance(total, int) and min_score is not None:
            base = f"정밀 분석 후 종합점수 {total:+d}점으로 기준 {min_score:+d}점 미만"
        else:
            base = "정밀 분석 후 기준 미달"
    else:
        if isinstance(total, int) and min_score is not None:
            base = f"종합점수 {total:+d}점으로 기준 {min_score:+d}점 미만"
        else:
            base = "종합점수 기준 미달"

    # surge 사유는 점수 요인을 덧붙이지 않음 — 사유가 점수 외적이라 혼선 방지
    if kind != "surge":
        if negative:
            base += f". 부담 요인: {negative}"
        elif positive:
            base += f". 긍정 요인은 있으나 기준점 미달: {positive}"
    return base


def _history_status_label(meta: dict) -> str:
    latest_date = meta.get("latest_date")
    if meta.get("in_latest"):
        return "🆕 신규" if meta.get("first_seen") == latest_date else "✅ 유지"
    return "⚠️ 탈락"


def _history_reason_text(meta: dict, in_universe: bool) -> str:
    if meta.get("in_latest"):
        reason = meta.get("reason") or meta.get("last_reason") or "최근 스크리닝 최종 통과"
        prefix = "신규 등장" if meta.get("first_seen") == meta.get("latest_date") else "유지"
    else:
        reason = meta.get("drop_reason") or "최근 스크리닝 최종 통과 목록에 없어 탈락으로 표시됨"
        prefix = "탈락"

    if not in_universe:
        reason += " / 안전 유니버스 기준에서도 이탈"
    return f"{prefix}: {reason}"


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
        st_score = r.get("short_term_score", 0)
        st_max = r.get("short_term_max", 0)
        mt_score = r.get("mid_term_score", 0)
        mt_max = r.get("mid_term_max", 0)
        m2.caption(
            f"단기 **{st_score:+d}**/{st_max} · 중기 **{mt_score:+d}**/{mt_max}  \n"
            f":gray[1~4주 매매엔 단기 / 분기 보유엔 중기 점수 참고]"
        )
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

            # 화면에는 날짜만 보이도록 하고, 같은 날짜에 여러 번 분석한 경우 마지막 기록만 사용
            sh_df["분석일"] = sh_df["date"].dt.strftime("%Y-%m-%d")
            chart_df = (
                sh_df.sort_values("date")
                .groupby("분석일", as_index=False)
                .last()
                .sort_values("분석일")
            )

            # 오른쪽 축 숫자가 4.64e+5처럼 보이지 않도록 만원 단위로 변환
            chart_df["주가_만원"] = chart_df["주가"] / 10000

            if chart_df["주가"].notna().any():
                chart_df = chart_df.dropna(subset=["주가"]).copy()

                chart_spec = {
                    "height": 240,
                    "layer": [
                        {
                            "mark": {"type": "line", "point": True},
                            "encoding": {
                                "x": {
                                    "field": "분석일",
                                    "type": "ordinal",
                                    "title": "분석일",
                                    "axis": {"labelAngle": -45},
                                },
                                "y": {
                                    "field": "종합점수",
                                    "type": "quantitative",
                                    "title": "종합점수",
                                    "scale": {"domain": [-4, 6]},
                                    "axis": {
                                        "title": "종합점수",
                                        "orient": "left",
                                        "values": [-4, -2, 0, 2, 4, 6],
                                    },
                                },
                                "tooltip": [
                                    {"field": "분석일", "type": "nominal", "title": "날짜"},
                                    {"field": "종합점수", "type": "quantitative", "title": "종합점수", "format": "+d"},
                                    {"field": "주가", "type": "quantitative", "title": "주가(원)", "format": ",.0f"},
                                ],
                            },
                        },
                        {
                            "mark": {"type": "line", "point": True, "strokeDash": [5, 4]},
                            "encoding": {
                                "x": {
                                    "field": "분석일",
                                    "type": "ordinal",
                                    "title": "분석일",
                                    "axis": {"labelAngle": -45},
                                },
                                "y": {
                                    "field": "주가_만원",
                                    "type": "quantitative",
                                    "title": "주가(만원)",
                                    "axis": {
                                        "title": "주가(만원)",
                                        "orient": "right",
                                        "format": ",.0f",
                                    },
                                    "scale": {"zero": False},
                                },
                                "tooltip": [
                                    {"field": "분석일", "type": "nominal", "title": "날짜"},
                                    {"field": "종합점수", "type": "quantitative", "title": "종합점수", "format": "+d"},
                                    {"field": "주가", "type": "quantitative", "title": "주가(원)", "format": ",.0f"},
                                ],
                            },
                        },
                    ],
                    "resolve": {"scale": {"y": "independent"}},
                }
                st.vega_lite_chart(chart_df, chart_spec, use_container_width=True)
                st.caption(
                    "실선은 종합점수, 점선은 해당 분석일 종가입니다. "
                    "왼쪽 축은 -4~+6점, 오른쪽 축은 주가(만원)입니다."
                )
            else:
                score_chart_spec = {
                    "height": 200,
                    "mark": {"type": "line", "point": True},
                    "encoding": {
                        "x": {
                            "field": "분석일",
                            "type": "ordinal",
                            "title": "분석일",
                            "axis": {"labelAngle": -45},
                        },
                        "y": {
                            "field": "종합점수",
                            "type": "quantitative",
                            "title": "종합점수",
                            "scale": {"domain": [-4, 6]},
                            "axis": {"values": [-4, -2, 0, 2, 4, 6]},
                        },
                        "tooltip": [
                            {"field": "분석일", "type": "nominal", "title": "날짜"},
                            {"field": "종합점수", "type": "quantitative", "title": "종합점수", "format": "+d"},
                        ],
                    },
                }
                st.vega_lite_chart(chart_df, score_chart_spec, use_container_width=True)
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
                elif src.get("has_dart") and src.get("dart_error"):
                    # 진짜 오류(네트워크·키·HTTP)일 때만 표시.
                    # ETF·SPAC·신규상장 등 단순 미등록은 dart_error=None이라 조용히 패스.
                    lines.append(
                        f"- **재무 데이터** — DART 조회 실패 (`{src['dart_error']}`)"
                    )
                elif not src.get("has_dart"):
                    # DART 미연동 상태 — 가치/재무/성장/공시 점수 모두 안 나옴.
                    # 사용자가 원인을 모르고 헤매지 않게 명시적으로 노출.
                    lines.append(
                        "- **DART 미연동** — 가치/재무 건전성/성장성/공시 점수가 빠집니다. "
                        "사이드바의 DART 상태를 확인하거나, 워커 스레드 secrets 접근 "
                        "이슈일 수 있습니다 (메인 스레드는 정상이어도 워커가 못 읽는 경우)."
                    )
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


# ─────────────────────────────────────────────────────────────────────────────
# 백그라운드 러너 레지스트리 — st.cache_resource 싱글톤
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit은 매 rerun마다 스크립트 본문을 위에서 아래로 재실행한다. 따라서
# 모듈 레벨에서 `D: dict = {}` 라고 쓰면 rerun마다 D가 새 dict로 교체돼서,
# 백그라운드 스레드가 채워놓은 데이터가 다음 rerun에선 사라진다.
# st.cache_resource로 감싸면 함수 본문이 단 한 번만 실행되고 그 반환 dict가
# 캐시돼 — 모든 rerun에서 같은 dict 객체를 받게 된다.
@st.cache_resource(show_spinner=False)
def _runner_registry() -> dict:
    return {
        "screen": None,
        "analyze": None,
        "refresh": None,
        "lock": threading.Lock(),
    }


def _get_screen_runner() -> dict | None:
    reg = _runner_registry()
    with reg["lock"]:
        return reg["screen"]


def _clear_screen_runner() -> None:
    reg = _runner_registry()
    with reg["lock"]:
        reg["screen"] = None


def _render_screen_diagnostics(runner: dict) -> None:
    """스크리닝 결과 저장 여부 진단 — 항상 표시. save 블록 도달 여부, Gist 응답을 노출."""
    save_ok = runner.get("save_ok")
    save_err = runner.get("save_error")
    stage = runner.get("stage")
    captured_log = runner.get("sync_log") or []

    try:
        import cloud_store as _cs
        live_log = _cs.get_sync_log()[-15:]
        gist_configured = _cs.is_configured()
    except Exception:
        live_log = []
        gist_configured = None

    scouted_added = runner.get("scouted_added")
    scouted_skipped = runner.get("scouted_skipped")
    scouted_err = runner.get("scouted_error")

    if save_err:
        st.error(f"⚠️ 스크리닝 결과 저장 실패: {save_err}")
    elif save_ok is None:
        st.warning(
            f"⚠️ save 블록 미도달 — 마지막 stage: `{stage}`. "
            "worker가 save 단계 전에 종료됨."
        )
    elif save_ok is True and not gist_configured:
        st.warning(
            "⚠️ Gist 미설정 상태로 저장 — 로컬에만 저장됐고 Streamlit Cloud "
            "재배포 시 사라집니다. (백그라운드 스레드에서 `st.secrets` 접근 실패 의심)"
        )

    if scouted_err:
        st.error(f"⚠️ 점수 시뮬레이션(scouted) 저장 실패: {scouted_err}")
    elif scouted_added == 0 and scouted_skipped and scouted_skipped > 0:
        st.info(
            f"ℹ️ 점수 시뮬레이션: 통과 {scouted_skipped}개 모두 이미 추적 중 — "
            "새로 추가된 종목 없음 (이미 발굴 시점 점수·종가가 기록돼있음)."
        )
    elif scouted_added and scouted_added > 0:
        st.success(
            f"✅ 점수 시뮬레이션에 **{scouted_added}개** 새로 추가 "
            f"(이미 추적 중 {scouted_skipped or 0}개는 건너뜀)."
        )

    log_to_show = captured_log or live_log
    with st.expander("🔍 진단: 저장 상태 / Gist 동기화 로그", expanded=bool(save_err) or save_ok is None or bool(scouted_err)):
        st.markdown(
            f"- `save_ok` (screening_history) = `{save_ok}`\n"
            f"- `save_error` = `{save_err}`\n"
            f"- `scouted_added` = `{scouted_added}`, `scouted_skipped` = `{scouted_skipped}`\n"
            f"- `scouted_error` = `{scouted_err}`\n"
            f"- 마지막 `stage` = `{stage}`\n"
            f"- Gist `is_configured()` (메인 스레드 기준) = `{gist_configured}`\n"
            f"- 캡처된 sync_log 항목 수 = `{len(captured_log)}`\n"
            f"- 현재 sync_log 항목 수 = `{len(live_log)}`"
        )
        if log_to_show:
            st.markdown("**최근 sync_log:**")
            for line in log_to_show:
                st.code(line, language=None)
        else:
            st.caption("(sync_log 비어있음 — `cloud_store.save()` 가 한 번도 호출 안 됨)")


def _screen_worker(runner: dict, stock_dict_local: dict[str, str]) -> None:
    universe = runner["universe"]
    min_score = runner["min_score"]

    runner["stage"] = "fetch_universe"
    runner["stage_label"] = "유니버스 종목 목록 로딩"
    try:
        codes = get_universe_codes(universe)
    except Exception as e:
        runner["universe_error"] = str(e)
        runner["status"] = "error"
        runner["error_msg"] = f"KRX 종목 목록을 가져오지 못했습니다: {e}"
        return

    if not codes:
        runner["status"] = "error"
        runner["error_msg"] = "유니버스 종목 목록을 가져오지 못했습니다."
        return

    runner["universe_size"] = len(codes)
    runner["total"] = len(codes)
    runner["stage"] = "stage1"
    runner["stage_label"] = "1단계 스크리닝"

    screened: list[dict] = []
    completed = 0
    errors = 0
    hist_module.begin_batch()
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(cached_analyze, c, False, 0): c for c in codes}
            for future in as_completed(futures):
                if runner.get("_cancel"):
                    break
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
                runner["completed"] = completed
                runner["errors"] = errors
                runner["current"] = stock_dict_local.get(c, c)
    finally:
        try:
            hist_module.commit_batch()
        except Exception:
            pass

    if runner.get("_cancel"):
        runner["status"] = "cancelled"
        return

    dropped_details: list[dict] = []
    excluded_stale = 0
    excluded_surge = 0
    fresh_results = []
    for r in screened:
        src = r.get("sources") or {}
        is_stale = (src.get("fin_freshness") or {}).get("is_stale", False)
        has_preliminary = bool(r.get("preliminary"))
        if is_stale and not has_preliminary:
            excluded_stale += 1
            dropped_details.append({
                "code": r.get("code"),
                "name": r.get("name", r.get("code")),
                "total": r.get("total"),
                "close": r.get("last_close"),
                "reason": _build_drop_reason(r, min_score, kind="stale"),
            })
            continue
        # 최근 급등 자동 제외 — 발굴 시점이 단기 고점이라 평균 회귀 위험.
        # 점수 시뮬레이션에서 단기 ≥3 그룹이 전부 손실 (-3~-8%) 보인 패턴 대응.
        surge_info = r.get("recent_surge") or {}
        if surge_info.get("is_surge"):
            excluded_surge += 1
            dropped_details.append({
                "code": r.get("code"),
                "name": r.get("name", r.get("code")),
                "total": r.get("total"),
                "close": r.get("last_close"),
                "reason": _build_drop_reason(r, min_score, kind="surge"),
            })
            continue
        fresh_results.append(r)

    fresh_results.sort(key=lambda r: r.get("total", -999), reverse=True)
    results = [r for r in fresh_results if r.get("total", -999) >= min_score]
    score_dropped = [r for r in fresh_results if r.get("total", -999) < min_score]
    excluded_score = len(score_dropped)
    for r in score_dropped:
        dropped_details.append({
            "code": r.get("code"),
            "name": r.get("name", r.get("code")),
            "total": r.get("total"),
            "close": r.get("last_close"),
            "reason": _build_drop_reason(r, min_score, kind="score"),
        })

    runner["fresh_count"] = len(fresh_results)
    runner["fresh_top_score"] = max(
        (r.get("total", -999) for r in fresh_results), default=None,
    )
    runner["excluded_stale"] = excluded_stale
    runner["excluded_surge"] = excluded_surge
    runner["excluded_score"] = excluded_score
    runner["pre_deep_count"] = len(results)

    if results:
        from analyzer import enrich_with_deep_analysis, recompute_score_after_deep
        runner["stage"] = "stage2"
        runner["stage_label"] = "2단계 정밀 분석"
        runner["total"] = len(results)
        runner["completed"] = 0
        runner["current"] = "준비 중"

        def _enrich(r: dict) -> None:
            enrich_with_deep_analysis(r, top_n=3)
            recompute_score_after_deep(r)

        with ThreadPoolExecutor(max_workers=3) as deep_exec:
            deep_futures = {deep_exec.submit(_enrich, r): r for r in results}
            deep_done = 0
            for future in as_completed(deep_futures):
                if runner.get("_cancel"):
                    break
                try:
                    future.result()
                except Exception:
                    pass
                deep_done += 1
                rr = deep_futures[future]
                runner["completed"] = deep_done
                runner["current"] = rr.get("name", rr.get("code", ""))

        if runner.get("_cancel"):
            runner["status"] = "cancelled"
            return

        pre_filter_results = list(results)
        deep_dropped = [r for r in pre_filter_results if r.get("total", -999) < min_score]
        for r in deep_dropped:
            dropped_details.append({
                "code": r.get("code"),
                "name": r.get("name", r.get("code")),
                "total": r.get("total"),
                "close": r.get("last_close"),
                "reason": _build_drop_reason(r, min_score, kind="deep"),
            })
        results = [r for r in pre_filter_results if r.get("total", -999) >= min_score]
        results.sort(key=lambda r: r.get("total", -999), reverse=True)
        runner["dropped_after_deep"] = len(deep_dropped)

    for r in results:
        r["_screen_reason"] = _build_pass_reason(r, min_score)

    runner["stage"] = "save"
    runner["stage_label"] = "결과 저장 중"
    runner["save_ok"] = False
    runner["save_error"] = None
    try:
        if hasattr(screening_history, "record_today_details"):
            screening_history.record_today_details(
                results, dropped_details,
                min_score=min_score, universe=universe,
            )
        else:
            screening_history.record_today([r["code"] for r in results])
        runner["save_ok"] = True
    except Exception as e:
        runner["save_error"] = f"{type(e).__name__}: {e}"
    try:
        import cloud_store as _cs
        runner["sync_log"] = _cs.get_sync_log()[-10:]
    except Exception:
        runner["sync_log"] = []

    runner["scouted_added"] = None
    runner["scouted_skipped"] = None
    runner["scouted_error"] = None
    try:
        added, skipped = scouted.add_many_from_analysis(results, universe=universe)
        runner["scouted_added"] = added
        runner["scouted_skipped"] = skipped
    except Exception as e:
        runner["scouted_error"] = f"{type(e).__name__}: {e}"
    try:
        import cloud_store as _cs
        runner["sync_log"] = _cs.get_sync_log()[-15:]
    except Exception:
        pass

    runner["results"] = results
    runner["dropped_details"] = dropped_details
    runner["status"] = "done"


def _start_screen_runner(
    universe: str, min_score: int, stock_dict_local: dict[str, str],
) -> dict:
    runner: dict = {
        "status": "running",
        "stage": "init",
        "stage_label": "준비 중",
        "completed": 0,
        "total": 0,
        "universe_size": 0,
        "current": "",
        "errors": 0,
        "universe": universe,
        "min_score": min_score,
        "results": None,
        "dropped_details": None,
        "excluded_stale": 0,
        "excluded_surge": 0,
        "excluded_score": 0,
        "dropped_after_deep": 0,
        "fresh_top_score": None,
        "fresh_count": 0,
        "pre_deep_count": 0,
        "universe_error": None,
        "error_msg": None,
        "started_at": time.time(),
        "finished_at": None,
        "_cancel": False,
    }
    reg = _runner_registry()
    with reg["lock"]:
        reg["screen"] = runner

    def _work():
        try:
            _screen_worker(runner, stock_dict_local)
        except Exception as e:
            runner["status"] = "error"
            runner["error_msg"] = f"{type(e).__name__}: {e}"
        finally:
            runner["finished_at"] = time.time()
            if runner["status"] == "running":
                runner["status"] = "done"

    t = threading.Thread(target=_work, daemon=True, name="screen-worker")
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        add_script_run_ctx(t, get_script_run_ctx())
    except Exception:
        pass
    t.start()
    return runner


# ─────────────────────────────────────────────────────────────────────────────
# 백그라운드 일반 분석 러너 — 관심종목/즐겨찾기/단독/선택 종목 분석용
# ─────────────────────────────────────────────────────────────────────────────
def _get_analyze_runner() -> dict | None:
    reg = _runner_registry()
    with reg["lock"]:
        return reg["analyze"]


def _clear_analyze_runner() -> None:
    reg = _runner_registry()
    with reg["lock"]:
        reg["analyze"] = None


def _analyze_worker(runner: dict, stock_dict_local: dict[str, str]) -> None:
    codes = runner["codes"]
    lite = runner.get("lite", False)
    deep_top = runner.get("deep_top", 3)
    label = runner.get("label", "분석")
    total = len(codes)

    runner["total"] = total
    runner["stage"] = "analyzing"
    runner["stage_label"] = label

    if runner.get("clear_cache"):
        try:
            cached_analyze.clear()
        except Exception:
            pass

    analyzed: list[dict] = []
    errors = 0
    hist_module.begin_batch()
    try:
        # 병렬화 — 종목 수가 적어도 IO bound라 3병렬이면 충분히 빠름
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(cached_analyze, c, lite, deep_top): c for c in codes}
            done = 0
            for f in as_completed(futs):
                if runner.get("_cancel"):
                    break
                c = futs[f]
                try:
                    r = f.result()
                    if not r.get("error"):
                        analyzed.append(r)
                    else:
                        errors += 1
                except Exception:
                    errors += 1
                done += 1
                runner["completed"] = done
                runner["errors"] = errors
                runner["current"] = stock_dict_local.get(c, c)
    finally:
        try:
            hist_module.commit_batch()
        except Exception:
            pass

    if runner.get("_cancel"):
        runner["status"] = "cancelled"
        runner["results"] = analyzed  # 부분 결과 보존
        return

    # 입력 순서를 유지하려면 코드 순으로 정렬
    code_order = {c: i for i, c in enumerate(codes)}
    analyzed.sort(key=lambda r: code_order.get(r.get("code"), 999_999))

    runner["results"] = analyzed
    runner["status"] = "done"


def _start_analyze_runner(
    kind: str,
    codes: list[str],
    label: str,
    *,
    lite: bool = False,
    deep_top: int = 3,
    clear_cache: bool = False,
    subheader: str | None = None,
    info_msg: str | None = None,
    stock_dict_local: dict[str, str] | None = None,
) -> dict:
    clean_codes = [str(c).zfill(6) for c in codes if str(c).strip()]
    runner: dict = {
        "kind": kind,
        "status": "running",
        "stage": "init",
        "stage_label": label,
        "label": label,
        "codes": clean_codes,
        "total": len(clean_codes),
        "completed": 0,
        "current": "",
        "errors": 0,
        "lite": lite,
        "deep_top": deep_top,
        "clear_cache": clear_cache,
        "subheader": subheader,
        "info_msg": info_msg,
        "results": None,
        "error_msg": None,
        "started_at": time.time(),
        "finished_at": None,
        "_cancel": False,
    }
    reg = _runner_registry()
    with reg["lock"]:
        reg["analyze"] = runner

    sd = stock_dict_local if stock_dict_local is not None else {}

    def _work():
        try:
            _analyze_worker(runner, sd)
        except Exception as e:
            runner["status"] = "error"
            runner["error_msg"] = f"{type(e).__name__}: {e}"
        finally:
            runner["finished_at"] = time.time()
            if runner["status"] == "running":
                runner["status"] = "done"

    t = threading.Thread(target=_work, daemon=True, name="analyze-worker")
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        add_script_run_ctx(t, get_script_run_ctx())
    except Exception:
        pass
    t.start()
    return runner


# ─────────────────────────────────────────────────────────────────────────────
# 백그라운드 추적 종목 재분석 러너 — '최근 스크리닝 후보' 뷰의 '모두 다시 분석'
# ─────────────────────────────────────────────────────────────────────────────
def _get_refresh_runner() -> dict | None:
    reg = _runner_registry()
    with reg["lock"]:
        return reg["refresh"]


def _clear_refresh_runner() -> None:
    reg = _runner_registry()
    with reg["lock"]:
        reg["refresh"] = None


def _refresh_worker(runner: dict, stock_dict_local: dict[str, str]) -> None:
    codes = runner["codes"]
    n_codes = len(codes)
    runner["total"] = n_codes
    runner["stage"] = "stage1"
    runner["stage_label"] = "1단계 재분석"

    try:
        cached_analyze.clear()
    except Exception:
        pass

    from analyzer import enrich_with_deep_analysis, recompute_score_after_deep

    refreshed: list[dict] = []
    errs = 0
    hist_module.begin_batch()
    try:
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(cached_analyze, c, False, 0): c for c in codes}
            done = 0
            for f in as_completed(futs):
                if runner.get("_cancel"):
                    break
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
                runner["completed"] = done
                runner["errors"] = errs
                runner["current"] = stock_dict_local.get(c, c)

        if runner.get("_cancel"):
            runner["status"] = "cancelled"
            return

        if refreshed:
            runner["stage"] = "stage2"
            runner["stage_label"] = "2단계 정밀 분석"
            runner["total"] = len(refreshed)
            runner["completed"] = 0
            runner["current"] = "준비 중"

            def _enrich(r):
                enrich_with_deep_analysis(r, top_n=3)
                recompute_score_after_deep(r)

            with ThreadPoolExecutor(max_workers=3) as dex:
                dfuts = {dex.submit(_enrich, r): r for r in refreshed}
                d_done = 0
                for f in as_completed(dfuts):
                    if runner.get("_cancel"):
                        break
                    try:
                        f.result()
                    except Exception:
                        pass
                    d_done += 1
                    rr = dfuts[f]
                    runner["completed"] = d_done
                    runner["current"] = rr.get("name", rr.get("code", ""))

            if runner.get("_cancel"):
                runner["status"] = "cancelled"
                return
    finally:
        try:
            hist_module.commit_batch()
        except Exception:
            pass

    runner["refreshed_count"] = len(refreshed)
    runner["errors"] = errs
    runner["status"] = "done"


def _start_refresh_runner(
    codes: list[str], stock_dict_local: dict[str, str],
) -> dict:
    clean_codes = [str(c).zfill(6) for c in codes if str(c).strip()]
    runner: dict = {
        "status": "running",
        "stage": "init",
        "stage_label": "준비 중",
        "codes": clean_codes,
        "total": len(clean_codes),
        "completed": 0,
        "current": "",
        "errors": 0,
        "refreshed_count": 0,
        "error_msg": None,
        "started_at": time.time(),
        "finished_at": None,
        "_cancel": False,
    }
    reg = _runner_registry()
    with reg["lock"]:
        reg["refresh"] = runner

    def _work():
        try:
            _refresh_worker(runner, stock_dict_local)
        except Exception as e:
            runner["status"] = "error"
            runner["error_msg"] = f"{type(e).__name__}: {e}"
        finally:
            runner["finished_at"] = time.time()
            if runner["status"] == "running":
                runner["status"] = "done"

    t = threading.Thread(target=_work, daemon=True, name="refresh-worker")
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        add_script_run_ctx(t, get_script_run_ctx())
    except Exception:
        pass
    t.start()
    return runner


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

    # 추적 종목 다시 분석 — 백그라운드 러너로 위임 (탭 백그라운드/새로고침에도 살아남음)
    _refresh_runner = _get_refresh_runner()
    _refresh_can_start = (
        _refresh_runner is None
        or _refresh_runner.get("status") in ("done", "error", "cancelled")
    )
    if st.session_state.pop("_run_refresh_picks", False) and recent_picks and _refresh_can_start:
        if _refresh_runner is not None:
            _clear_refresh_runner()
        _refresh_runner = _start_refresh_runner(
            list(recent_picks.keys()), stock_dict,
        )

    if _refresh_runner is not None and _refresh_runner.get("status") == "running":
        st.info(
            "💡 백그라운드에서 진행됩니다. **다른 화면에 갔다 와도, 새로고침해도 "
            "재분석은 계속 돕니다.** 이 페이지로 돌아오면 진행 상황을 다시 볼 수 있어요."
        )
        _rr_stage = _refresh_runner.get("stage")
        _rr_label = _refresh_runner.get("stage_label", "진행 중")
        _rr_completed = _refresh_runner.get("completed", 0)
        _rr_total = _refresh_runner.get("total", 0) or 0
        _rr_current = _refresh_runner.get("current", "") or ""

        if _rr_stage == "stage1":
            st.caption(f"1단계: {_rr_total}개 종목 재분석")
        elif _rr_stage == "stage2":
            st.caption(f"2단계: {_rr_total}개 종목 정밀 분석")

        _rr_art = st.empty()
        _rr_pb = st.progress(0)
        _rr_sb = st.empty()
        _progress_update(
            _rr_pb, _rr_sb, _rr_completed, _rr_total,
            _rr_label, f"완료: {_rr_current}",
            _rr_art, f"refresh-{_rr_stage}",
        )

        _rc_col, _ = st.columns([1, 5])
        if _rc_col.button("취소", key="cancel_refresh_btn"):
            _refresh_runner["_cancel"] = True

        # JS 기반 자동 새로고침 — Streamlit Cloud에서 time.sleep+st.rerun보다 안정적
        st_autorefresh(interval=2000, key="refresh_runner_poll")
    elif _refresh_runner is not None and _refresh_runner.get("status") == "error":
        st.error(_refresh_runner.get("error_msg") or "재분석 중 오류 발생")
        if st.button("닫기", key="close_refresh_err"):
            _clear_refresh_runner()
            st.rerun()
    elif _refresh_runner is not None and _refresh_runner.get("status") == "cancelled":
        st.warning("🛑 재분석이 취소됐습니다.")
        if st.button("닫기", key="close_refresh_cancel"):
            _clear_refresh_runner()
            st.rerun()
    elif _refresh_runner is not None and _refresh_runner.get("status") == "done":
        st.success(
            f"✅ {_refresh_runner.get('refreshed_count', 0)}개 점수 업데이트 완료 "
            f"(에러 {_refresh_runner.get('errors', 0)}건). 표가 새 점수로 갱신됐습니다."
        )
        # 표 그리기 위해 recent_picks 다시 로드 (worker가 score_history 갱신했음)
        recent_picks = screening_history.get_recent(days=90)
        _clear_refresh_runner()

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
                "구분": _history_status_label(meta),
                "이유": _history_reason_text(meta, in_universe),
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

        # 정렬 — 모든 옵션이 "큰 값/최근 날짜 먼저" (DESC) 의도라 통일 처리.
        # 날짜 컬럼은 문자열이라 단순히 -v가 안 먹어서 YYYYMMDD 정수로 변환해 부호 반전.
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
            if isinstance(v, (int, float)):
                return (0, -v)
            if isinstance(v, str):
                try:
                    return (0, -int(v.replace("-", "")))
                except ValueError:
                    return (0, v)
            return (0, v)

        indexed = list(enumerate(rows))
        indexed.sort(key=_sort_key)
        rows = [r for _, r in indexed]
        code_order = [code_order[i] for i, _ in indexed]

        if not rows:
            st.info("필터 조건에 맞는 종목이 없습니다.")
        else:
            df = pd.DataFrame(rows)
            df.insert(0, "선택", False)

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                key="screening_pick_editor",
                disabled=[c for c in df.columns if c != "선택"],
                column_config={
                    "선택": st.column_config.CheckboxColumn(
                        "선택",
                        help="분석하고 싶은 종목만 체크하세요",
                        width="small",
                    ),
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
                    "구분": st.column_config.TextColumn(
                        width="small",
                        help="이번 최신 스크리닝 기준 신규·유지·탈락 여부",
                    ),
                    "이유": st.column_config.TextColumn(
                        width="large",
                        help="신규 등장 또는 탈락으로 표시된 핵심 이유",
                    ),
                },
            )

            selected_screen_codes: list[str] = []
            if isinstance(edited_df, pd.DataFrame) and "선택" in edited_df.columns:
                for idx, checked in enumerate(edited_df["선택"].fillna(False).tolist()):
                    if bool(checked) and idx < len(code_order):
                        selected_screen_codes.append(code_order[idx])

            a_col1, a_col2 = st.columns([1, 2])
            with a_col1:
                if st.button(
                    f"🔍 선택한 종목만 분석 ({len(selected_screen_codes)}개)",
                    key="analyze_selected_screening_rows",
                    use_container_width=True,
                    type="primary",
                    disabled=len(selected_screen_codes) == 0,
                    help="체크한 종목만 현재 시세·재무·공시 기준으로 다시 분석합니다.",
                ):
                    st.session_state["_selected_screen_codes"] = selected_screen_codes
                    st.session_state.pop("_view_mode", None)
                    st.rerun()
            with a_col2:
                st.caption(
                    "왼쪽 체크박스에서 원하는 종목만 고른 뒤 분석하세요. "
                    "1개만 체크하면 단독 분석처럼 볼 수 있고, 여러 개를 체크하면 비교표와 상세 분석이 함께 나옵니다."
                )

            latest_new = [r for r in rows if str(r.get("구분", "")).startswith("🆕")]
            latest_dropped = [r for r in rows if str(r.get("구분", "")).startswith("⚠️")]
            if latest_new or latest_dropped:
                with st.expander("🆕 신규 등장 / ⚠️ 탈락 이유 상세", expanded=True):
                    if latest_new:
                        st.markdown("**🆕 신규 등장**")
                        for r in latest_new:
                            st.markdown(f"- **{r['종목']}** — {r['이유']}")
                    if latest_dropped:
                        st.markdown("**⚠️ 탈락**")
                        for r in latest_dropped:
                            st.markdown(f"- **{r['종목']}** — {r['이유']}")
                    st.caption(
                        "정확한 항목별 이유는 이 수정 이후 실행한 스크리닝부터 저장됩니다. "
                        "이전 기록은 '최종 통과 목록에 없음'처럼 단순 표시될 수 있습니다."
                    )

    st.stop()  # 이 뷰만 보여주고 아래 일반 분석 흐름은 실행 안 함


# ─────────── 점수 시뮬레이션 뷰 모드 ───────────
if st.session_state.get("_view_mode") == "verifier":
    import verifier

    st.subheader("📊 점수 시뮬레이션 — 점수가 실제 수익률과 맞았는가")
    st.caption(
        "스크리닝에서 발굴된 종목을 그 시점에 가상으로 매수했다고 가정하고 현재까지의 수익률을 추적합니다. "
        "현재가는 가장 최근 분석된 종가 기준 — 발굴 후 한 번도 다시 분석되지 않은 종목은 빠집니다."
    )

    if st.button("← 분석 화면으로 돌아가기", key="back_from_verifier", use_container_width=False):
        st.session_state.pop("_view_mode", None)
        st.rerun()

    tab_scout, tab_item = st.tabs([
        "💰 발굴 종목 가상 수익률",
        "🔬 항목별 예측력",
    ])

    def _bucket_table(bucket_stats: dict, value_label: str) -> pd.DataFrame:
        rows = []
        for label, st_ in bucket_stats.items():
            rows.append({
                "점수 구간": label,
                "종목 수": st_["count"],
                f"평균 {value_label}(%)": st_["avg_return"] if st_["avg_return"] is not None else "-",
                "승률(%)": st_["win_rate"] if st_["win_rate"] is not None else "-",
                "최고(%)": st_["best"] if st_["best"] is not None else "-",
                "최저(%)": st_["worst"] if st_["worst"] is not None else "-",
            })
        return pd.DataFrame(rows)

    # ─── 탭 1: 발굴 가상 수익률 ───
    with tab_scout:
        score_type = st.radio(
            "점수 종류 비교 — 어느 점수가 실제 수익률을 가장 잘 예측했는지",
            options=["total", "short_term", "mid_term"],
            format_func=lambda t: {
                "total": "종합 점수 (모든 항목 합)",
                "short_term": "단기 점수 (거래량·수급·공시·시장강도)",
                "mid_term": "중기 점수 (추세·가치·재무·성장성)",
            }[t],
            horizontal=True,
            key="sim_score_type",
            help=(
                "단기 점수는 1~4주 매매에 효과가 학술적으로 검증된 항목만 합산. "
                "중기 점수는 분기 이상에서 효과가 있는 항목만 합산. "
                "모멘텀(RSI)과 가격리스크는 의심 항목이라 종합점수에만 포함되고 부분합엔 빠짐."
            ),
        )
        result = verifier.verify_scouted(score_type=score_type)
        if result["total_count"] == 0:
            if score_type in ("short_term", "mid_term"):
                st.info(
                    f"`{score_type}` 점수가 기록된 발굴 종목이 없습니다. "
                    "단기·중기 점수는 항목별 점수 스냅샷이 있어야 계산되는데, 이 기능은 "
                    "최근 추가됐기 때문에 새로 발굴된 종목부터 데이터가 쌓입니다.\n\n"
                    "사이드바에서 **🔍 발굴 시작**을 한 번 다시 돌려보세요."
                )
            else:
                st.info(
                    "시뮬레이션할 발굴 종목이 없습니다. 사이드바에서 **🔍 발굴 시작**을 한 번 돌리면 "
                    "통과 종목들이 자동으로 발굴 추적에 등록되고, 이후 가격 변화가 가상 수익률로 추적됩니다."
                )
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("발굴 종목", f"{result['total_count']}개")
            avg = result.get("overall_avg")
            col2.metric("평균 수익률", f"{avg:+.2f}%" if avg is not None else "-")
            wr = result.get("overall_win_rate")
            col3.metric("승률", f"{wr}%" if wr is not None else "-")

            st.markdown("##### 발굴 시점 점수 구간별 평균 수익률")
            st.caption(
                "→ '8점 이상에서 발굴됐던 종목들이 평균 +N% 수익이었나' 같은 통계. "
                "위 구간 평균이 아래 구간 평균보다 일관되게 높으면 점수가 잘 맞고 있는 것."
            )
            st.dataframe(
                _bucket_table(result["bucket_stats"], "수익률"),
                hide_index=True, use_container_width=True,
            )

            st.markdown("##### 종목별 상세")
            detail_rows = []
            for r in result["rows"]:
                cur_score = r.get("current_score")
                detail_rows.append({
                    "종목": f"{stock_dict.get(r['code'], '?')} ({r['code']})",
                    "발굴일": r["added_at"] or "-",
                    "발굴점수": f"{r['added_score']:+d}" if r["added_score"] is not None else "-",
                    "현재점수": f"{cur_score:+d}" if cur_score is not None else "-",
                    "발굴가": f"{r['added_close']:,.0f}",
                    "현재가": f"{r['current_close']:,.0f}",
                    "수익률(%)": r["return_pct"],
                    "유니버스": r.get("universe", ""),
                })
            st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)

    # ─── 탭 2: 항목별 예측력 ───
    with tab_item:
        st.caption(
            "각 항목 점수가 며칠 뒤 수익률을 얼마나 잘 예측했는지. "
            "spread = (양수 점수 평균 수익률) − (음수 점수 평균 수익률). "
            "+면 항목이 예측력 있음, −면 반대로 작동, 0 근처면 무의미."
        )
        forward_days = st.radio(
            "예측 기간",
            options=[5, 20, 60],
            format_func=lambda d: f"{d}일",
            horizontal=True,
            key="verifier_forward_days",
            index=0,
        )
        item_result = verifier.verify_item_scores(forward_days=forward_days)
        st.caption(
            f"전체 샘플 {item_result['total_samples']:,}건 "
            f"(종목×시점 페어, 항목별 점수 기록이 있는 일자만)"
        )

        if item_result["total_samples"] == 0:
            st.info(
                f"분석에 필요한 데이터가 부족합니다. 항목별 점수 저장은 오늘부터 시작되므로 "
                f"적어도 {forward_days + 5}일 이상 매일 분석을 돌려야 의미 있는 통계가 나옵니다."
            )
        else:
            ranked = item_result["ranked_items"]
            spread_rows = []
            for item in ranked:
                st_ = item_result["item_stats"][item]
                spread_rows.append({
                    "항목": item,
                    "예측력 spread(%p)": st_["predictive_spread"] if st_["predictive_spread"] is not None else "-",
                    "샘플 수": st_["total_samples"],
                })
            st.markdown(f"##### 항목별 예측력 ({forward_days}일 뒤 수익률 기준)")
            st.dataframe(pd.DataFrame(spread_rows), hide_index=True, use_container_width=True)

            st.markdown("##### 항목 상세 — 점수값별 평균 수익률")
            for item in ranked:
                st_ = item_result["item_stats"][item]
                with st.expander(
                    f"{item}  (spread {st_['predictive_spread']:+.2f}%p, "
                    f"{st_['total_samples']}건)" if st_["predictive_spread"] is not None
                    else f"{item}  ({st_['total_samples']}건)"
                ):
                    rows = []
                    for score_val, stat in st_["by_score"].items():
                        rows.append({
                            "점수값": f"{score_val:+d}",
                            "케이스 수": stat["count"],
                            "평균 수익률(%)": stat["avg_return"] if stat["avg_return"] is not None else "-",
                            "승률(%)": stat["win_rate"] if stat["win_rate"] is not None else "-",
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.stop()


focus_code = st.session_state.pop("_focus_code", None)
analyze_favs_only = st.session_state.pop("_analyze_favs_only", False)
selected_screen_codes = st.session_state.pop("_selected_screen_codes", [])
screen_req = st.session_state.pop("_screen", None)
auto_run = st.session_state.pop("_auto_run_analyze", False)

results = None
_screen_runner = _get_screen_runner()

# 새 발굴 요청 — 진행 중인 러너가 없을 때만 새로 시작 (이중 클릭 방지)
if screen_req and (
    _screen_runner is None
    or _screen_runner.get("status") in ("done", "error", "cancelled")
):
    if _screen_runner is not None:
        _clear_screen_runner()
    _screen_runner = _start_screen_runner(
        screen_req["universe"], screen_req["min_score"], stock_dict,
    )

if _screen_runner is not None and _screen_runner.get("status") == "running":
    universe = _screen_runner["universe"]
    label = UNIVERSE_LABELS.get(universe, universe)
    st.subheader(f"🔍 발굴 진행 중 — {label}")
    st.info(
        "💡 백그라운드에서 진행됩니다. **브라우저 탭이 잠시 끊기거나 새로고침해도 "
        "스크리닝은 계속 돕니다.** 페이지로 돌아오면 진행 상황을 다시 볼 수 있어요."
    )

    stage = _screen_runner.get("stage")
    stage_label = _screen_runner.get("stage_label") or "진행 중"
    completed = _screen_runner.get("completed", 0)
    total = _screen_runner.get("total", 0) or 0
    current = _screen_runner.get("current", "") or ""
    errors = _screen_runner.get("errors", 0)

    art_box = st.empty()
    progress_bar = st.progress(0)
    status_box = st.empty()

    if stage == "fetch_universe":
        st.caption("유니버스 종목 목록 로딩 중...")
        progress_bar.progress(0)
        status_box.caption(stage_label)
    elif stage == "stage1":
        st.caption("1단계: 전체 분석 (공시 분류 + 잠정실적 + 뉴스)")
        _progress_update(
            progress_bar, status_box, completed, total,
            "1단계 스크리닝", f"완료: {current}",
            art_box, "screening-stage1",
        )
    elif stage == "stage2":
        st.caption(f"2단계: 통과 {total}개 종목 정밀 분석 중...")
        _progress_update(
            progress_bar, status_box, completed, total,
            "2단계 정밀 분석", f"완료: {current}",
            art_box, "screening-stage2",
        )
    elif stage == "save":
        st.caption("결과 저장 중...")
        progress_bar.progress(1.0)
        status_box.caption(stage_label)
    else:
        status_box.caption(stage_label)

    if errors:
        st.caption(f"⚠️ 1단계 에러 {errors}건 (계속 진행)")

    cancel_col, _spacer = st.columns([1, 5])
    if cancel_col.button("취소", key="cancel_screen_btn"):
        _screen_runner["_cancel"] = True
        st.caption("취소 요청 — 현재 진행 중인 종목까지만 처리하고 종료합니다.")

    # JS 기반 자동 새로고침 — Streamlit Cloud에서 time.sleep+st.rerun보다 안정적
    st_autorefresh(interval=2000, key="screen_runner_poll")

elif _screen_runner is not None and _screen_runner.get("status") in ("done", "error", "cancelled"):
    universe = _screen_runner["universe"]
    min_score = _screen_runner["min_score"]
    label = UNIVERSE_LABELS.get(universe, universe)
    st.subheader(f"🔍 발굴 결과 — {label}")

    if _screen_runner["status"] == "error":
        st.error(_screen_runner.get("error_msg") or "알 수 없는 오류가 발생했습니다.")
        ue = _screen_runner.get("universe_error")
        if ue:
            with st.expander("오류 상세 보기"):
                st.code(str(ue))
        if st.button("닫기 / 다시 시도", key="close_screen_err"):
            _clear_screen_runner()
            st.rerun()
        st.stop()

    if _screen_runner["status"] == "cancelled":
        st.warning("🛑 사용자 취소 — 중간 결과는 저장되지 않았습니다.")
        if st.button("닫기", key="close_screen_cancel"):
            _clear_screen_runner()
            st.rerun()
        st.stop()

    # status == "done"
    results = _screen_runner.get("results") or []
    universe_size = _screen_runner.get("universe_size", 0)
    errors = _screen_runner.get("errors", 0)
    excluded_stale = _screen_runner.get("excluded_stale", 0)
    excluded_surge = _screen_runner.get("excluded_surge", 0)
    excluded_score = _screen_runner.get("excluded_score", 0)
    dropped_after_deep = _screen_runner.get("dropped_after_deep", 0)
    top_score = _screen_runner.get("fresh_top_score")

    if results:
        msg = (
            f"✅ {universe_size}개 1차 분석 · **{len(results)}개 최종 통과** "
            f"(점수 ≥ {min_score:+d}, 에러 {errors}건"
        )
        if excluded_stale > 0:
            msg += f", stale 제외 {excluded_stale}"
        if excluded_surge > 0:
            msg += f", 급등 제외 {excluded_surge}"
        if dropped_after_deep > 0:
            msg += f", 정밀 분석 후 탈락 {dropped_after_deep}"
        msg += ")."
        st.success(msg)
        st.info(
            f"📌 최종 통과 {len(results)}개가 사이드바 **'최근 스크리닝 후보'** 에 "
            "자동 추가됐습니다. 매일 돌리면 점수 변화·주가 추이·이탈이 추적돼요."
        )
        _render_screen_diagnostics(_screen_runner)
    else:
        st.warning(
            f"🚫 **조건을 통과한 종목이 없습니다** "
            f"({universe_size}개 분석)"
        )
        with st.container(border=True):
            st.markdown("#### 📊 발굴 내역")
            lines = [
                f"- 분석 시도: **{universe_size}개**",
                f"- 에러: {errors}개",
            ]
            if excluded_stale > 0:
                lines.append(
                    f"- 재무 stale + 잠정실적 없음 자동 제외: **{excluded_stale}개**"
                )
            if excluded_surge > 0:
                lines.append(
                    f"- 발굴 시점 급등 자동 제외(고점 추격 회피): **{excluded_surge}개**"
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
        _render_screen_diagnostics(_screen_runner)
    st.session_state.results = results
    # 다음 트리거(분석/즐겨찾기/단독 등)가 이 분기에 막히지 않도록 러너 비움
    # — 결과는 session_state에 이미 저장됐고, 표 렌더는 마지막 `if results:`에서 처리
    _clear_screen_runner()

else:
    # ─── 일반 분석(관심종목/즐겨찾기/단독/선택) 트리거 → 백그라운드 러너 ───
    _analyze_runner = _get_analyze_runner()

    # 1) 새 트리거 처리 — 진행 중인 러너가 없을 때만 새로 시작
    _can_start_new = (
        _analyze_runner is None
        or _analyze_runner.get("status") in ("done", "error", "cancelled")
    )

    _new_trigger: tuple | None = None
    if _can_start_new:
        if selected_screen_codes:
            _ssc = [str(c).zfill(6) for c in selected_screen_codes if str(c).strip()]
            if _ssc:
                _new_trigger = (
                    "selected", _ssc, "선택 종목 분석",
                    False, 3, True,
                    "🔍 선택 종목 분석 결과",
                    f"📌 최근 스크리닝 후보에서 체크한 **{len(_ssc)}개** 종목만 분석했습니다. "
                    "다른 종목을 고르려면 사이드바의 **최근 스크리닝 후보**로 다시 들어가세요.",
                )
            else:
                st.warning("선택된 종목이 없습니다.")
        elif focus_code:
            _name = stock_dict.get(focus_code, focus_code)
            _new_trigger = (
                "focus", [focus_code], f"{_name} 단독 분석",
                False, 3, False,
                None,
                f"⭐ **{_name}** 단독 분석 결과입니다. 워치리스트 전체를 보려면 **🔍 분석하기**를 누르세요.",
            )
        elif analyze_favs_only:
            _favs = load_favorites()
            if _favs:
                _new_trigger = (
                    "favs", _favs, "즐겨찾기 분석",
                    False, 3, False,
                    None,
                    f"⭐ 즐겨찾기 **{len(_favs)}개**만 분석한 결과입니다. "
                    "워치리스트는 그대로이며, 워치리스트 전체를 보려면 **🔍 분석하기**를 누르세요.",
                )
            else:
                st.warning("⭐ 즐겨찾기가 비어있습니다. 종목 카드의 ☆ 버튼으로 추가하세요.")
        elif (run_analysis or auto_run) and selected_codes:
            _new_trigger = (
                "watchlist", selected_codes, "관심종목 분석",
                False, 3, False, None, None,
            )

    if _new_trigger:
        _k, _codes, _label, _lite, _dtop, _cc, _sub, _imsg = _new_trigger
        if _analyze_runner is not None:
            _clear_analyze_runner()
        _analyze_runner = _start_analyze_runner(
            kind=_k, codes=_codes, label=_label,
            lite=_lite, deep_top=_dtop, clear_cache=_cc,
            subheader=_sub, info_msg=_imsg,
            stock_dict_local=stock_dict,
        )
    elif (
        not _can_start_new
        and (focus_code or selected_screen_codes or analyze_favs_only or run_analysis or auto_run)
    ):
        # 분석 진행 중인데 새 분석 트리거가 또 눌렸음 — 안내
        st.caption("ℹ️ 이미 분석이 진행 중입니다. 완료 후 다시 시도하세요.")

    # 2) 러너 상태별 화면 표시
    if _analyze_runner is not None and _analyze_runner.get("status") == "running":
        if _analyze_runner.get("subheader"):
            st.subheader(_analyze_runner["subheader"])
        st.info(
            "💡 백그라운드에서 진행됩니다. **다른 화면에 갔다 와도, 새로고침해도 "
            "분석은 계속 돕니다.** 페이지로 돌아오면 진행 상황을 다시 볼 수 있어요."
        )
        _ar_label = _analyze_runner.get("label", "분석")
        _ar_completed = _analyze_runner.get("completed", 0)
        _ar_total = _analyze_runner.get("total", 0) or 0
        _ar_current = _analyze_runner.get("current", "") or ""
        _ar_errors = _analyze_runner.get("errors", 0)

        _ar_art = st.empty()
        _ar_pb = st.progress(0)
        _ar_sb = st.empty()
        _progress_update(
            _ar_pb, _ar_sb, _ar_completed, _ar_total,
            _ar_label, f"완료: {_ar_current}",
            _ar_art, f"{_ar_label}-analysis",
        )
        if _ar_errors:
            st.caption(f"⚠️ 에러 {_ar_errors}건 (계속 진행)")

        _cc_col, _ = st.columns([1, 5])
        if _cc_col.button("취소", key="cancel_analyze_btn"):
            _analyze_runner["_cancel"] = True
            st.caption("취소 요청 — 현재까지 분석된 종목으로 마무리합니다.")

        # JS 기반 자동 새로고침 — Streamlit Cloud에서 time.sleep+st.rerun보다 안정적
        st_autorefresh(interval=2000, key="analyze_runner_poll")

    elif _analyze_runner is not None and _analyze_runner.get("status") == "error":
        st.error(_analyze_runner.get("error_msg") or "분석 중 오류가 발생했습니다.")
        if st.button("닫기 / 다시 시도", key="close_analyze_err"):
            _clear_analyze_runner()
            st.rerun()

    elif _analyze_runner is not None and _analyze_runner.get("status") == "cancelled":
        _partial = _analyze_runner.get("results") or []
        if _partial:
            st.warning(f"🛑 분석 취소 — 취소 전까지 완료된 **{len(_partial)}개** 결과만 표시합니다.")
            results = _partial
            st.session_state.results = results
        else:
            st.warning("🛑 분석이 취소됐습니다.")
        if st.button("닫기", key="close_analyze_cancel"):
            _clear_analyze_runner()
            st.rerun()

    elif _analyze_runner is not None and _analyze_runner.get("status") == "done":
        if _analyze_runner.get("subheader"):
            st.subheader(_analyze_runner["subheader"])
        if _analyze_runner.get("info_msg"):
            st.info(_analyze_runner["info_msg"])
        results = _analyze_runner.get("results") or []
        st.session_state.results = results
        # 다음 트리거가 막히지 않도록 결과 채택 후 러너 비움 — 결과는 session_state에 남음
        _clear_analyze_runner()

    elif not selected_codes:
        st.info("👈 사이드바에서 분석할 종목을 선택하세요.")
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
