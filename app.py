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
    KOREAN_NAMES, SCREEN_MIN_SCORE, UNIVERSE_LABELS, all_korean_stocks, analyze,
    get_universe_codes, market_regime_state, position_size,
)
import dart
import holdings_monitor
import llm
import portfolio as port
import history as hist_module
import cloud_store
import screening_history
import scouted


DEFAULT_WATCHLIST = ["005930", "000660", "035420"]


def _stock_option(code: str, stock_dict: dict[str, str]) -> str:
    """multiselect option label. Unknown codes are still preserved safely."""
    code = str(code).strip().zfill(6)
    return f"{code} {stock_dict.get(code, '저장된 종목')}"


def build_watchlist_options(
    stock_dict: dict[str, str],
    saved_codes: list[str],
    extra_codes: list[str] | None = None,
) -> list[str]:
    """
    Build sidebar options while preserving saved/custom codes even when KRX
    listing lookup temporarily fails.
    """
    options = [f"{code} {name}" for code, name in sorted(stock_dict.items())]
    seen = {opt.split(maxsplit=1)[0] for opt in options}
    for code in [*saved_codes, *(extra_codes or [])]:
        clean = str(code).strip().zfill(6)
        if not clean or clean in seen:
            continue
        options.append(_stock_option(clean, stock_dict))
        seen.add(clean)
    return options


def _codes_from_options(items: list[str]) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for item in items:
        code = str(item).split(maxsplit=1)[0].strip().zfill(6)
        if not code or code in seen:
            continue
        codes.append(code)
        seen.add(code)
    return codes


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
def _full_stock_dict() -> dict[str, str]:
    """전체 KRX 종목 목록 (코드 → 이름). 성공했을 때만 24h 캐시된다.

    all_korean_stocks() 가 실패하면 예외를 그대로 올려서, st.cache_data 가 결과를
    캐시하지 않게 한다(예외는 캐시되지 않음) → 다음 호출 때 자동 재시도.
    """
    return all_korean_stocks()


def get_stock_dict() -> dict[str, str]:
    """전체 KRX 종목 이름맵. 전체 목록 실패 시에만 작은 fallback(이건 캐시 안 함).

    과거: 실패 시 15종목짜리 KOREAN_NAMES 를 24h 캐시에 박아, 그 15개 외 전 종목이
    '?'로 표시되던 버그. 이제 fallback 은 캐시하지 않아 KRX 가 살아나면 곧바로 복구된다.
    """
    try:
        return _full_stock_dict()
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
    saved_codes = load_watchlist()

    # 즐겨찾기 클릭으로 추가 요청된 코드를 multiselect 렌더 전에 주입
    pending_to_add = st.session_state.pop("_pending_add_to_watchlist", [])
    options = build_watchlist_options(stock_dict, saved_codes, pending_to_add)
    if "watchlist_select" in st.session_state:
        current_codes = _codes_from_options(st.session_state["watchlist_select"])
        options = build_watchlist_options(stock_dict, saved_codes, [*pending_to_add, *current_codes])
        st.session_state["watchlist_select"] = [
            _stock_option(c, stock_dict) for c in current_codes
        ]
    if pending_to_add:
        current = st.session_state.get("watchlist_select")
        if current is None:
            current = [_stock_option(c, stock_dict) for c in saved_codes]
        for c in pending_to_add:
            opt = _stock_option(c, stock_dict)
            if opt in options and opt not in current:
                current = current + [opt]
        st.session_state["watchlist_select"] = current

    default_options = [_stock_option(c, stock_dict) for c in saved_codes]

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
            name = stock_dict.get(code, code)
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

        _trail_pct = float(port.load_settings().get("trail_pct", 10.0))
        for code, h in portfolio.items():
            name = stock_dict.get(code, code)
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
                    if r:
                        _ev = holdings_monitor.evaluate_holding(
                            code, name, h, r, trail_pct=_trail_pct,
                        )
                        for _a in _ev["alerts"]:
                            _icon = holdings_monitor.LEVEL_ICON.get(_a["level"], "•")
                            st.markdown(
                                f"<small>{_icon} {_a['msg']}</small>",
                                unsafe_allow_html=True,
                            )
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
            # 분석된 종목이면 매수 시점 손절/목표가도 함께 저장 → 보유 점검 기준이 됨
            _plan = (results_map.get(code) or {}).get("trade_plan") or {}
            port.add_holding(
                code, port_qty, port_price,
                stop_loss=_plan.get("stop_loss"),
                target_1r=_plan.get("target_1r"),
                target_2r=_plan.get("target_2r"),
            )
            st.success(f"{stock_dict.get(code, code)} 저장됨")
            st.rerun()

    with st.expander("⚙️ 투자금·리스크 설정 (매수 수량 제안)"):
        _rs = port.load_settings()
        _eq = st.number_input(
            "총 투자금 (원)", min_value=0, step=1_000_000,
            value=int(_rs["account_equity"]), key="rs_equity",
            help="0이면 매수 수량 제안이 꺼집니다.",
        )
        _risk = st.number_input(
            "종목당 리스크 (%)", min_value=0.1, max_value=10.0, step=0.1,
            value=float(_rs["risk_per_trade_pct"]), key="rs_risk",
            help="한 종목이 손절되면 계좌의 몇 %를 잃을지. 보통 0.5~2%.",
        )
        _maxpos = st.number_input(
            "종목당 최대 비중 (%)", min_value=1.0, max_value=100.0, step=1.0,
            value=float(_rs["max_position_pct"]), key="rs_maxpos",
        )
        _trail = st.number_input(
            "트레일링 스톱 (고점 대비 %)", min_value=1.0, max_value=50.0, step=1.0,
            value=float(_rs["trail_pct"]), key="rs_trail",
            help="보유 종목이 고점 대비 이만큼 떨어지면 '이익 보호' 알림.",
        )
        _ro_on = st.checkbox(
            "하락장 방어 (코스피 200일선 아래면 진입 기준 상향)",
            value=bool(_rs.get("risk_off_enabled", True)), key="rs_ro_on",
            help="하락 추세에선 좋은 점수 종목도 같이 빠지기 쉬워, 신규 진입 기준점수를 높입니다.",
        )
        _ro_boost = st.number_input(
            "하락장 진입 기준 가산점", min_value=0, max_value=10, step=1,
            value=int(_rs.get("risk_off_score_boost", 2)), key="rs_ro_boost",
            help="리스크오프 때 min_score 에 더할 점수. 클수록 더 엄격(=후보 적어짐).",
        )
        if st.button("설정 저장", key="rs_save", use_container_width=True):
            port.save_settings({
                "account_equity": _eq, "risk_per_trade_pct": _risk,
                "max_position_pct": _maxpos, "trail_pct": _trail,
                "risk_off_enabled": _ro_on,
                "risk_off_score_boost": _ro_boost,
            })
            st.success("리스크 설정 저장됨")
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
    # 최근 스크리닝 실행 상태 — "오늘 돌았나?" 를 한눈에. (자동 스크리닝이 PC와
    # 무관하게 도는지 사용자가 직접 확인하는 용도. 날짜 키 존재 = 그날 실행됨.)
    # 상태 위젯은 어떤 경우에도 앱 전체를 중단시키면 안 되므로 방어적으로 감싼다 —
    # 예: 배포 직후 Streamlit 이 app.py 는 새로 실행했지만 import 된 모듈은 아직
    # 옛 버전이라 신규 함수가 없는 과도기(AttributeError) 에도 조용히 넘어가게.
    try:
        _last_run = screening_history.last_run()
        if _last_run:
            _when = (
                f"{_last_run['ran_at']} KST" if _last_run.get("ran_at")
                else _last_run["date"]
            )
            _passed = _last_run.get("passed_count", 0)
            if _last_run.get("ran_today"):
                st.success(f"🟢 오늘 스크리닝 완료 · {_when} · 통과 {_passed}개")
            else:
                _ago = _last_run.get("days_ago")
                _ago_txt = f" · {_ago}일 전" if isinstance(_ago, int) and _ago > 0 else ""
                st.warning(f"🟡 마지막 스크리닝 {_when}{_ago_txt} · 통과 {_passed}개")
        else:
            st.caption("⚪ 아직 스크리닝 기록이 없습니다 — 자동 스크리닝이 한 번 돌면 표시됩니다")
    except Exception:
        pass  # 상태 표시 실패는 조용히 무시 — 사이드바 한 줄 때문에 앱이 죽지 않게
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
    try:
        _regime = market_regime_state()
        if _regime.get("risk_off"):
            _rs_now = port.load_settings()
            if _rs_now.get("risk_off_enabled", True):
                st.warning(
                    f"🛡️ {_regime['label']} — 신규 진입 기준 "
                    f"+{int(_rs_now.get('risk_off_score_boost', 2))}점 상향 적용 중"
                )
            else:
                st.caption(f"🛡️ {_regime['label']} (하락장 방어 꺼짐)")
    except Exception:
        pass

    universe = st.selectbox(
        "유니버스",
        options=list(UNIVERSE_LABELS.keys()),
        format_func=lambda k: UNIVERSE_LABELS[k],
        key="screen_universe_select",
        help="🛡️ 안전 유니버스 = 시총 5조+ / 거래대금 500억+ / 관리종목 등 제외 — 작전주 위험 낮은 후보들",
    )

    # 기준 종합점수는 코드에 고정(analyzer.SCREEN_MIN_SCORE) — 사용자가 매번 고르지 않음.
    min_score = SCREEN_MIN_SCORE
    st.caption(
        f"📌 기준 종합점수 **{SCREEN_MIN_SCORE}점 이상** 자동 적용 "
        f"(하락장에선 자동 상향). 점수 기준은 시스템이 관리합니다."
    )

    # 시간 추정 — KRX 종목 목록 서버가 불안정해도 앱 전체가 멈추지 않도록 보호
    try:
        est_codes = len(get_universe_codes(universe))
    except Exception:
        est_codes = 0

    if est_codes > 0:
        est_pass1_sec = est_codes * 4 / 3
        est_min = max(1, round(est_pass1_sec / 60))
        st.caption(
            f"⏱️ 빠른 1차 스크리닝 약 {est_min}분 예상 "
            f"({est_codes}개 ÷ 3병렬). 통과 종목엔 선택적으로 깊이 분석 실행."
        )
    else:
        st.caption(
            "⏱️ 현재 KRX 종목 목록을 불러오지 못해 예상 시간을 계산하지 못했습니다. "
            "관심 종목 분석은 계속 사용할 수 있고, 종목 발굴은 잠시 후 다시 시도하세요."
        )

    screen_deep = st.checkbox(
        "통과 종목 정밀 분석까지 실행",
        value=True,
        key="screen_deep_analysis",
        help="끄면 1차 점수로 바로 결과를 보여줘서 훨씬 빠릅니다. 켜면 통과 종목만 공시 본문을 더 깊게 확인합니다.",
    )

    if st.button("🔍 발굴 시작", use_container_width=True, key="run_screen"):
        st.session_state["_screen"] = {
            "universe": universe,
            "min_score": min_score,
            "deep_screen": screen_deep,
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
ANALYZER_VERSION = "v31-2026-05-28-tone-balance-bucket-recalibration"
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
    if total >= 8:
        return "✨"
    if total >= 4:
        return "🟡"
    if total >= -1:
        return "👀"
    return "⚠️"


SCORE_COLUMNS = [
    "추세", "모멘텀", "거래량", "가격 리스크",
    "시장 상대강도", "시장 국면", "수급", "공시",
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
            f":gray[종합점수는 가격 반응 신호(상대강도·수급·거래량·공시 ×2)에 가중치 적용]"
        )
        m3.metric("의견", f"{opinion_emoji(r['total'])} {r['opinion'].split(' — ')[0]}")

        # 매매 가이드는 참고용 — 기본 접힘, 사용자가 명시적으로 펼쳤을 때만 표시.
        # "매수하세요/매도하세요" 같은 추천이 아니라 ATR 기반 손절/목표가 산정 참고치.
        plan = r.get("trade_plan") or {}
        if plan and plan.get("entry_price"):
            action = plan.get("action", "-")
            confidence = plan.get("confidence", "-")
            reason = plan.get("reason", "")
            # 라벨 첫 마디만 보여 expander 헤더에 — 펼치기 전에도 톤은 알 수 있게
            if "긍정" in action:
                emoji = "🟢"
            elif "위험" in action:
                emoji = "🔴"
            else:
                emoji = "⚪"

            with st.expander(f"📈 참고용 매매 가이드 ({emoji} {action}) — 추천 아님, 손절·목표가 산정용", expanded=False):
                st.caption(
                    "⚠️ **이것은 매수·매도 추천이 아닙니다.** "
                    "점수와 ATR(평균 변동폭) 기반의 일반적인 손절/목표가 *후보*일 뿐, "
                    "사용자 본인의 리스크 허용도·포지션 크기·전략에 맞게 직접 조정하세요."
                )
                st.markdown(
                    f"- **신호 강도**: {action} · 신뢰도 {confidence}\n"
                    f"- **근거**: {reason}"
                )
                if plan.get("stop_loss") and plan.get("target_1r") and plan.get("target_2r"):
                    risk_pct = plan.get("risk_pct")
                    risk_text = f" (현재가 대비 {risk_pct:.1f}%)" if isinstance(risk_pct, (int, float)) else ""
                    atr_text = (
                        f" · ATR(14일) {plan['atr14']:,.0f}원"
                        if plan.get("atr14") else ""
                    )
                    st.markdown(
                        f"- **손절가 후보**: {plan['stop_loss']:,.0f}원{risk_text}{atr_text}\n"
                        f"- **1R 목표**: {plan['target_1r']:,.0f}원 (리스크 1배 이익)\n"
                        f"- **2R 목표**: {plan['target_2r']:,.0f}원 (리스크 2배 이익)"
                    )
                    st.caption(
                        "*R(Risk) = 진입가 - 손절가*. "
                        "1R 목표는 손절폭만큼의 이익 (손익비 1:1), 2R은 2배 (1:2)."
                    )

                # 포지션 사이징 — 리스크 설정이 있으면 매수 수량 제안 (정보용, 추천 아님)
                _rs = port.load_settings()
                if _rs.get("account_equity", 0) and plan.get("stop_loss"):
                    _ps = position_size(
                        entry_price=plan.get("entry_price") or r["last_close"],
                        stop_loss=plan.get("stop_loss"),
                        account_equity=_rs["account_equity"],
                        risk_per_trade_pct=_rs["risk_per_trade_pct"],
                        max_position_pct=_rs["max_position_pct"],
                    )
                    if _ps["ok"]:
                        _cap = " (최대비중 상한 적용)" if _ps["capped"] else ""
                        st.markdown(
                            f"- **제안 매수 수량**: 약 **{_ps['shares']:,}주** "
                            f"(약 {_ps['position_value']:,.0f}원 · 계좌의 {_ps['position_pct']:.0f}%){_cap}\n"
                            f"- 손절 시 예상 손실 약 **{_ps['risk_amount']:,.0f}원** — {_ps['note']}"
                        )
                    elif _ps["note"]:
                        st.caption(f"💡 매수 수량 제안: {_ps['note']}")

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
                    # 캐시 사용 여부 표시 — DART 직접 호출 실패 시 옛 캐시로 fallback 가능
                    cache_note = ""
                    if src.get("fin_cache_stale"):
                        cache_note = (
                            f"  \n  ⚠️ DART 일시 오류로 **만료된 캐시 사용** "
                            f"(저장 시점: `{src.get('fin_cache_fetched_at', '?')}`)"
                        )
                    elif src.get("fin_cached"):
                        cache_note = (
                            f"  \n  ℹ️ Gist 캐시 사용 "
                            f"(저장 시점: `{src.get('fin_cache_fetched_at', '?')}`)"
                        )
                    lines.append(
                        f"- **가치 (PER/PBR) / 재무 건전성 / 성장성** — "
                        f"DART OPEN API · `{src['fin_report_label']}` 기준"
                        f"{fresh_note}{cache_note}"
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

    # 과열장 적응 통과 + 모멘텀 후보 — 통과 0개여도 시뮬에 쌓이고 주도주를 보여준다.
    if runner.get("overheated"):
        def _oh_rows(items):
            out = []
            for r in items:
                metrics = (r.get("recent_surge") or {}).get("metrics") or {}
                rel = next((s.get("relative") for s in (r.get("scores") or [])
                            if s.get("name") == "시장 상대강도"), None)
                out.append({
                    "종목": f'{r.get("name", "")}({r.get("code", "")})',
                    "종합점수": r.get("total"),
                    "20일(%)": metrics.get("d20"),
                    "시장대비(%p)": rel,
                    "종가": r.get("last_close"),
                })
            return out

        adaptive_picks = runner.get("adaptive_picks") or []
        obs_added = runner.get("observed_added")
        picks = runner.get("momentum_picks") or []

        # (1) 실제 통과 — 수익률 게이트(시장대비 5~25%p·거래량·펀더)를 통과한 적응 통과
        if adaptive_picks:
            st.success(
                f"✅ **과열장 적응 통과 {len(adaptive_picks)}개** — 시장 대비 초과가 적정"
                "(5~25%p)하고 거래량·펀더가 받쳐주는 건전 주도주입니다. 점수 시뮬레이션엔 "
                "일반 통과와 **분리(adaptive)** 기록돼 따로 검증됩니다(손절 전제)."
            )
            st.dataframe(
                pd.DataFrame(_oh_rows(adaptive_picks)),
                hide_index=True, use_container_width=True,
            )
        elif obs_added:
            st.info(
                f"🔭 이번엔 게이트를 통과한 적응 통과는 없지만, 관찰용으로 "
                f"**{obs_added}개**를 점수 시뮬레이션에 기록했습니다 (데이터 끊김 방지)."
            )

        # (2) 참고용 — 통과엔 못 들었지만 급등 중인 나머지 주도주(관찰)
        adaptive_codes = {a.get("code") for a in adaptive_picks}
        ref_picks = [r for r in picks if r.get("code") not in adaptive_codes]
        if ref_picks:
            st.markdown("#### 📈 과열장 모멘텀 후보 (참고용 · 자동 매수 아님)")
            st.caption(
                "통과 게이트엔 못 들었지만 급등 중인 주도주입니다. 추격 매수는 평균 회귀 "
                "위험이 크니 손절 전제로만 참고하세요. 관찰로 시뮬레이션에 기록됩니다."
            )
            st.dataframe(
                pd.DataFrame(_oh_rows(ref_picks)),
                hide_index=True, use_container_width=True,
            )

    log_to_show = captured_log or live_log
    with st.expander("🔍 진단: 저장 상태 / Gist 동기화 로그", expanded=bool(save_err) or save_ok is None or bool(scouted_err)):
        st.markdown(
            f"- `save_ok` (screening_history) = `{save_ok}`\n"
            f"- `save_error` = `{save_err}`\n"
            f"- `scouted_added` = `{scouted_added}`, `scouted_skipped` = `{scouted_skipped}`\n"
            f"- `observed_added` = `{runner.get('observed_added')}`, `overheated` = `{runner.get('overheated')}`\n"
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
            futures = {executor.submit(cached_analyze, c, True, 0): c for c in codes}
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

    # 하락장 리스크오프 — KOSPI가 200일선 아래면 진입 기준 상향 (auto_screen 과 동일)
    from analyzer import (
        market_regime_state as _mrs,
        effective_min_score,
        select_observation_targets,
        select_adaptive_picks,
    )
    _base_min = min_score
    regime = _mrs()
    min_score, _ro_boost = effective_min_score(_base_min, regime=regime)
    runner["regime"] = regime
    runner["min_score_base"] = _base_min
    runner["min_score_effective"] = min_score
    runner["risk_off_boost"] = _ro_boost

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

    if results and runner.get("deep_screen", True):
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

    # 과열장 적응 통과(adaptive) + 관찰(observed) — 통과 0개여도 시뮬 데이터가 끊기지 않게.
    runner["adaptive_added"] = None
    runner["adaptive_picks"] = []
    runner["observed_added"] = None
    runner["observed_error"] = None
    runner["overheated"] = bool(regime.get("overheated"))
    runner["momentum_picks"] = []
    passed_codes = {r.get("code") for r in results}

    # 과열장 적응 통과 — 시장 대비 초과가 적정한 건전 주도주를 소수 통과(수익률 게이트).
    try:
        adaptive = select_adaptive_picks(screened, regime, passed_codes=passed_codes)
        ad_added, _ad_skip = scouted.add_adaptive_from_analysis(adaptive, universe=universe)
        runner["adaptive_added"] = ad_added
        runner["adaptive_picks"] = adaptive
        passed_codes |= {r.get("code") for r in adaptive}
    except Exception as e:
        runner["adaptive_error"] = f"{type(e).__name__}: {e}"

    try:
        obs = select_observation_targets(
            screened, fresh_results, regime, passed_codes=passed_codes,
        )
        obs_added, _obs_skip = scouted.add_observed_from_analysis(obs, universe=universe)
        runner["observed_added"] = obs_added
        # 과열장 주도주(모멘텀 후보)만 UI 표시용으로 추림 — 관찰 대상 중 surge 인 것.
        if regime.get("overheated"):
            runner["momentum_picks"] = [
                r for r in obs if (r.get("recent_surge") or {}).get("is_surge")
            ]
    except Exception as e:
        runner["observed_error"] = f"{type(e).__name__}: {e}"
    try:
        import cloud_store as _cs
        runner["sync_log"] = _cs.get_sync_log()[-15:]
    except Exception:
        pass

    runner["results"] = results
    runner["dropped_details"] = dropped_details
    runner["status"] = "done"


def _start_screen_runner(
    universe: str,
    min_score: int,
    stock_dict_local: dict[str, str],
    *,
    deep_screen: bool = True,
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
        "deep_screen": deep_screen,
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
        "스크리닝에서 발굴된 종목을 그 시점에 가상으로 매수했다고 가정하고 수익률을 추적합니다. "
        "고정 시계(5/20/60일)로 비교하면 발굴일 다른 종목도 같은 잣대로 검증되고, "
        "시장 대비 초과수익(KOSPI·KOSDAQ 차감)으로 시장 방향성과 점수 능력을 분리합니다."
    )

    if st.button("← 분석 화면으로 돌아가기", key="back_from_verifier", use_container_width=False):
        st.session_state.pop("_view_mode", None)
        st.rerun()

    tab_scout, tab_item = st.tabs([
        "💰 발굴 종목 가상 수익률",
        "🔬 항목별 예측력",
    ])

    def _bucket_table(bucket_stats: dict) -> pd.DataFrame:
        rows = []
        for label, st_ in bucket_stats.items():
            rows.append({
                "점수 구간": label,
                "종목 수": st_["count"],
                "평균(%)": st_["avg_return"] if st_["avg_return"] is not None else "-",
                "중앙값(%)": st_["median_return"] if st_["median_return"] is not None else "-",
                "승률(%)": st_["win_rate"] if st_["win_rate"] is not None else "-",
                "초과 평균(%p)": st_["avg_excess"] if st_["avg_excess"] is not None else "-",
                "초과 승률(%)": st_["excess_win_rate"] if st_["excess_win_rate"] is not None else "-",
                "최고(%)": st_["best"] if st_["best"] is not None else "-",
                "최저(%)": st_["worst"] if st_["worst"] is not None else "-",
            })
        return pd.DataFrame(rows)

    # ─── 결과를 쉬운 말 한 줄 결론으로 ───
    # 표의 숫자를 비전문가도 바로 이해하게 "그래서 점수 높을 때 사면 이익?" 에
    # 답하는 문장을 자동 생성. 보수적으로 — 표본 적으면 단정 금지.
    def _emit_verdict(level: str, msg: str) -> None:
        {"success": st.success, "warning": st.warning, "info": st.info}.get(
            level, st.info
        )(msg)

    def _verdict_scout(result: dict) -> tuple[str, str]:
        n = result["total_count"]
        if n == 0:
            return ("info",
                "📌 아직 검증할 종목이 없어요. 사이드바에서 **🔍 발굴**을 한 번 돌리고 "
                "며칠 지나면 여기에 '점수가 수익으로 이어졌는지' 결과가 쌓입니다.")
        if n < 10:
            return ("warning",
                f"📌 지금은 검증된 종목이 **{n}개뿐**이라 결론 내기엔 매우 일러요. "
                "수십 개 이상 쌓이고 몇 달 지나야 믿을 만한 답이 됩니다.")

        buckets = result["bucket_stats"]
        top = next((v for k, v in buckets.items() if k.startswith("상위")), None)
        bot = next((v for k, v in buckets.items() if k.startswith("하위")), None)

        # 점수 구간(상위/하위) 비교는 구간당 표본이 충분해야 의미 → 전체 40개 이상에서만.
        if (n >= 40 and top and bot and top["avg_return"] is not None
                and bot["avg_return"] is not None):
            diff = top["avg_return"] - bot["avg_return"]
            ex = ""
            if top.get("avg_excess") is not None and top["avg_excess"] > 0:
                ex = f" (시장 대비로도 +{top['avg_excess']:.1f}%p)"
            if diff > 0.5 and top["avg_return"] > 0:
                return ("success",
                    f"📌 **점수가 높을수록 실제로 더 올랐어요.** 점수 상위 그룹은 "
                    f"평균 **{top['avg_return']:+.1f}%** (승률 {top['win_rate']}%){ex}로, "
                    f"하위 그룹({bot['avg_return']:+.1f}%)보다 {diff:.1f}%p 앞섰습니다. "
                    "→ 종합점수가 높을 때 사는 게 지금까지는 통했다는 신호입니다. "
                    "단, 과거 결과일 뿐 미래를 보장하진 않아요.")
            if diff > 0:
                return ("info",
                    f"📌 점수 상위 그룹(평균 {top['avg_return']:+.1f}%)이 "
                    f"하위({bot['avg_return']:+.1f}%)보다 조금 나았지만 차이가 작아요"
                    f"({diff:.1f}%p). 방향은 맞지만 '확실히 통한다'고 보긴 일러요.")
            return ("warning",
                f"📌 **아직은 점수가 높다고 더 오르진 않았어요.** 상위 그룹 평균 "
                f"{top['avg_return']:+.1f}%, 하위 그룹 {bot['avg_return']:+.1f}%. "
                "표본을 더 쌓아 보거나 점수 기준을 다시 볼 필요가 있습니다.")

        avg = result["overall_avg"]
        win = result["overall_win_rate"]
        if avg is not None:
            tail = (
                "아직 표본이 적어(또는 점수가 다양하지 않아) 점수 구간별 "
                "상/하위 비교는 보류합니다 — 수십 개 이상 쌓여야 합니다."
            )
            if avg > 0:
                return ("info",
                    f"📌 검증된 {n}개 종목의 평균 수익률은 **{avg:+.1f}%** "
                    f"(승률 {win}%)예요. {tail}")
            return ("warning",
                f"📌 검증된 {n}개 종목의 평균이 {avg:+.1f}%로 부진해요. {tail}")
        return ("info", "📌 결과를 계산했지만 요약할 수치가 부족합니다.")

    def _verdict_item(item_result: dict) -> tuple[str, str]:
        n = item_result["total_samples"]
        n_dates = item_result.get("n_dates", 0)
        if n == 0:
            return ("info",
                "📌 아직 항목별로 따질 데이터가 없어요. 매일 분석이 며칠 쌓이면 "
                "어떤 항목이 진짜 잘 맞는지 보입니다.")
        # rank-IC 는 표본·기간이 충분해야 신뢰. 점수가 거친 정수라 더 보수적으로 본다.
        if n < 100 or n_dates < 15:
            return ("warning",
                f"📌 표본이 아직 적어요(관측 {n:,}건·기간 {n_dates}일). 항목별 예측력은 "
                "**방향만 참고**하시고 숫자는 믿지 마세요. 몇 달 더 쌓여야 신뢰할 만합니다.")
        ics = [(it, item_result["item_stats"][it]["rank_ic"])
               for it in item_result["ranked_items"]
               if item_result["item_stats"][it]["rank_ic"] is not None]
        if not ics:
            return ("info",
                "📌 아직 어떤 항목도 예측력 통계(IC)를 낼 만큼 표본이 안 모였어요.")
        best_it, best_ic = ics[0]
        worst_it, worst_ic = ics[-1]
        if best_ic >= 0.05:
            extra = ""
            if worst_ic <= -0.05:
                extra = (f" 반대로 **'{worst_it}'**(IC {worst_ic:+.2f})는 거꾸로 작동하는 편이라 "
                         "비중 축소 후보예요.")
            return ("success",
                f"📌 예측력이 가장 좋은 항목은 **'{best_it}'**(IC {best_ic:+.2f})예요 "
                f"— IC가 +면 그 점수 높을수록 시장보다 더 올랐다는 뜻.{extra} "
                f"단, 표본 기간이 {n_dates}일로 아직 짧아 잠정 결과입니다.")
        return ("info",
            f"📌 아직 어떤 항목도 뚜렷한 예측력(IC≥0.05)을 못 보였어요(최고 '{best_it}' "
            f"IC {best_ic:+.2f}). 신호가 약하거나 데이터가 더 필요합니다.")

    # ─── 탭 1: 발굴 가상 수익률 ───
    with tab_scout:
        col_score, col_horizon = st.columns([2, 1])
        with col_score:
            score_type = st.radio(
                "점수 종류",
                options=["total", "short_term", "mid_term", "focus"],
                format_func=lambda t: {
                    "total": "종합 (가중)",
                    "short_term": "단기 (거래량·수급·공시·시장강도)",
                    "mid_term": "중기 (추세·가치·재무·성장성)",
                    "focus": "집중 (상대강도·재무·가치)",
                }[t],
                horizontal=True,
                key="sim_score_type",
                help=(
                    "단기 점수는 1~4주 매매에 중요도 높은 항목을 가중 합산. "
                    "중기 점수는 분기 이상에서 효과 있는 항목만 합산. "
                    "모멘텀·가격리스크는 의심 항목이라 종합점수에만 포함."
                ),
            )

        # 점수 종류를 바꾸면 보유 시계를 권장값으로 자동 맞춤 (안내형). 단기→5일,
        # 종합·집중→20일, 중기→60일. 사용자가 horizon 을 직접 바꾼 뒤 점수 종류를
        # 그대로 두면 그 선택은 유지된다(권장은 점수 종류가 바뀔 때만 다시 적용).
        _REC_HORIZON = {"short_term": "5d", "total": "20d", "focus": "20d", "mid_term": "60d"}
        if st.session_state.get("_prev_sim_score_type") != score_type:
            st.session_state["_prev_sim_score_type"] = score_type
            st.session_state["sim_horizon"] = _REC_HORIZON.get(score_type, "20d")

        with col_horizon:
            horizon = st.radio(
                "보유 시계",
                options=["5d", "20d", "60d", "all"],
                format_func=lambda h: {
                    "5d": "5영업일",
                    "20d": "20영업일",
                    "60d": "60영업일",
                    "all": "발굴일~현재",
                }[h],
                horizontal=False,
                key="sim_horizon",
                help=(
                    "5/20/60일은 같은 보유기간으로 비교 — 시계 도달 못 한 종목은 자동 제외. "
                    "'발굴일~현재'는 보유기간이 종목마다 다르고 시장 흐름에 영향 받음."
                ),
            )

        _rec = _REC_HORIZON.get(score_type)
        if _rec and horizon not in (_rec, "all"):
            _hl = {"5d": "5일", "20d": "20일", "60d": "60일", "all": "발굴~현재"}
            _sl = {"total": "종합", "short_term": "단기", "mid_term": "중기", "focus": "집중"}
            st.caption(
                f"⚠️ '{_sl.get(score_type, score_type)}' 점수는 **{_hl[_rec]} 보유**와 잘 맞아요. "
                f"지금 {_hl[horizon]}은 신호와 기간이 어긋나 결과가 약해 보일 수 있어요."
            )

        min_hold_days = 0
        if horizon == "all":
            min_hold_days = st.slider(
                "최소 보유 영업일 (이하 제외)",
                min_value=0, max_value=20,
                value=verifier.DEFAULT_MIN_HOLD_DAYS,
                step=1,
                key="sim_min_hold",
                help="발굴 직후 종목은 신호가 작동할 시간이 없어 통계에 노이즈를 만듦.",
            )

        track = st.radio(
            "추적 종류",
            options=["picked", "adaptive", "observed", "all"],
            format_func=lambda k: {
                "picked": "발굴(통과) — 정상장 매수 후보",
                "adaptive": "과열장 적응 통과 — 시장대비 게이트 통과",
                "observed": "관찰 — 통과 못 했지만 추적",
                "all": "전체",
            }[k],
            horizontal=True,
            key="sim_kind",
            help=(
                "발굴: 정상장에서 종합점수 기준을 통과해 가상 매수한 종목. "
                "과열장 적응 통과: 과열장에서 시장 대비 초과·거래량·펀더 게이트를 통과한 주도주. "
                "관찰: 통과 못 했지만 점수 검증용으로 추적하는 종목. "
                "서로 섞으면 성과가 희석되므로 기본은 '발굴'만 봅니다."
            ),
        )

        result = verifier.verify_scouted(
            score_type=score_type,
            horizon=horizon,
            min_hold_days=min_hold_days,
            kind=track,
        )

        _lvl, _msg = _verdict_scout(result)
        _emit_verdict(_lvl, _msg)

        if result["total_count"] == 0:
            short_hold = result.get("excluded_short_hold", 0)
            missing = result.get("missing_data_count", 0)
            if score_type in ("short_term", "mid_term") and missing > 0 and short_hold == 0:
                st.info(
                    f"`{score_type}` 점수가 기록된 발굴 종목이 없습니다. "
                    "단기·중기 점수는 항목별 스냅샷이 있어야 계산되는데, 새로 발굴된 종목부터 데이터가 쌓입니다.\n\n"
                    "사이드바에서 **🔍 발굴 시작**을 한 번 다시 돌려보세요."
                )
            elif horizon != "all" and short_hold > 0:
                st.info(
                    f"{horizon} 시계에 도달한 발굴 종목이 없습니다 "
                    f"({short_hold}개가 보유기간 부족으로 제외됨). "
                    "더 짧은 시계로 보거나 '발굴일~현재'를 선택해보세요."
                )
            else:
                st.info(
                    "시뮬레이션할 발굴 종목이 없습니다. 사이드바에서 **🔍 발굴 시작**을 한 번 돌리면 "
                    "통과 종목들이 자동으로 발굴 추적에 등록됩니다."
                )
        else:
            # 메인 메트릭 6칸: 절대 평균/중앙/승률 + 초과 평균/중앙/승률
            row1 = st.columns(3)
            row1[0].metric(
                "발굴 종목", f"{result['total_count']}개",
                help="검증에 사용한 발굴 종목 수. 수십 개 미만이면 결과를 믿기 어려움.",
            )
            avg = result.get("overall_avg")
            row1[1].metric(
                "평균 수익률", f"{avg:+.2f}%" if avg is not None else "-",
                help="발굴한 뒤 종목들이 평균 몇 % 올랐는지(전체 평균). 시장이 오르내린 "
                     "영향이 섞여 있어 — 진짜 실력은 아래 '시장 대비'로 봐야 함.",
            )
            wr = result.get("overall_win_rate")
            row1[2].metric(
                "승률", f"{wr}%" if wr is not None else "-",
                help="수익이 플러스로 끝난 종목의 비율.",
            )

            row2 = st.columns(3)
            med = result.get("overall_median")
            row2[0].metric(
                "중앙값",
                f"{med:+.2f}%" if med is not None else "-",
                help="평균은 outlier에 휘둘려서 중앙값과 함께 봐야 진짜 분포가 보임.",
            )
            ex_avg = result.get("overall_avg_excess")
            row2[1].metric(
                "시장 대비 평균",
                f"{ex_avg:+.2f}%p" if ex_avg is not None else "-",
                help="(종목 수익률) − (KOSPI/KOSDAQ 같은 기간 수익률). 양수면 시장 이긴 것.",
            )
            ex_wr = result.get("overall_excess_win_rate")
            row2[2].metric(
                "시장 이긴 비율",
                f"{ex_wr}%" if ex_wr is not None else "-",
                help="초과수익이 양수인 종목 비율. 50% 넘으면 점수가 시장 평균을 이긴다는 신호.",
            )

            trail_avg = result.get("overall_avg_trail")
            if trail_avg is not None:
                tp = result.get("trail_pct", 10.0)
                row3 = st.columns(3)
                row3[0].metric(
                    "손절 적용 평균", f"{trail_avg:+.2f}%",
                    help=f"고점 대비 {tp:.0f}% 빠지면 파는 '트레일링 스톱'을 적용한 평균 수익률. "
                         "끝까지 보유 대신 하락을 잘라낸 가상 성과. % 는 사이드바에서 조정.",
                )
                twr = result.get("overall_win_rate_trail")
                row3[1].metric("손절 적용 승률", f"{twr}%" if twr is not None else "-")
                stopped = result.get("trail_stopped_count", 0)
                row3[2].metric(
                    "손절 발동", f"{stopped}/{result['total_count']}건",
                    help="트레일링 스톱이 실제로 걸려 중간에 청산된 종목 수.",
                )

            excluded = result.get("excluded_short_hold", 0)
            if excluded > 0:
                if horizon == "all":
                    st.caption(f"ℹ️ 보유기간 {min_hold_days}일 미만인 발굴 {excluded}개 제외됨.")
                else:
                    st.caption(f"ℹ️ {horizon} 시계 미도달 발굴 {excluded}개 제외됨.")

            st.markdown("##### 점수 분위별 통계")
            st.caption(
                "상위/중위/하위는 발굴 종목들의 점수 분포에서 1/3씩 자동 분할. "
                "상위 분위의 '초과 평균'이 하위보다 일관되게 높으면 점수가 시장 대비 알파를 만듦."
            )
            st.dataframe(
                _bucket_table(result["bucket_stats"]),
                hide_index=True, use_container_width=True,
            )

            st.markdown("##### 종목별 상세")
            st.caption("초과수익 내림차순.")
            detail_rows = []
            for r in result["rows"]:
                cur_score = r.get("current_score")
                detail_rows.append({
                    "종목": f"{stock_dict.get(r['code'], r['code'])} ({r['code']})",
                    "발굴일": r["added_at"] or "-",
                    "보유일": r.get("days_held") if r.get("days_held") is not None else "-",
                    "발굴점수": f"{r['added_score']:+d}" if r["added_score"] is not None else "-",
                    "현재점수": f"{cur_score:+d}" if cur_score is not None else "-",
                    "발굴가": f"{r['added_close']:,.0f}",
                    "현재가": f"{r['current_close']:,.0f}",
                    "수익률(%)": r["return_pct"],
                    "손절적용(%)": r.get("trail_return_pct") if r.get("trail_return_pct") is not None else "-",
                    "시장(%)": r.get("market_return_pct") if r.get("market_return_pct") is not None else "-",
                    "초과(%p)": r.get("excess_return_pct") if r.get("excess_return_pct") is not None else "-",
                    "유니버스": r.get("universe", ""),
                })
            st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)

    # ─── 탭 2: 항목별 예측력 ───
    with tab_item:
        st.caption(
            "각 항목 점수가 며칠 뒤 **시장 대비** 수익률을 얼마나 잘 예측했는지. "
            "**IC**(순위상관)가 주 지표 — +면 점수 높을수록 더 오름(예측력 있음), "
            "−면 거꾸로 작동, 0 근처면 무의미. (보조 지표 spread = 양수점수 평균 − 음수점수 평균)"
        )
        forward_days = st.radio(
            "예측 기간",
            options=[5, 20, 60],
            format_func=lambda d: f"{d}일",
            horizontal=True,
            key="verifier_forward_days",
            index=0,
        )

        # ─── 종합점수 walk-forward 예측력 (헤드라인) ───
        # 새 모듈 함수 호출 — 배포 직후 Streamlit 이 verifier 옛 모듈을 캐시하면
        # AttributeError 로 탭 전체가 죽을 수 있어 try/except 로 감싼다(Reboot 시 정상).
        try:
            st.markdown("##### 🎯 walk-forward 예측력 — 종합 vs 집중")
            st.caption(
                "그날 점수로 *그 이후* 수익률을 **날짜별로 따로** 채점(rank-IC)한 뒤 평균낸 값. "
                "미래를 안 쓰는 out-of-sample 이라 가장 정직한 비교입니다. "
                "종합(9개 항목 가중)과 집중(상대강도·재무·가치)을 나란히 봅니다."
            )
            _wf = {}
            _wf_rows = []
            for _stp, _slabel in (("total", "종합"), ("focus", "집중")):
                _w = verifier.verify_composite_walk_forward(forward_days=forward_days, score_type=_stp)
                _wf[_stp] = _w
                if _w["n_periods"] == 0:
                    _wf_rows.append({"점수": _slabel, "평균 IC": None, "t값": None,
                                     "IC 양수%": None, "상·하위⅓ 수익차(%p)": None, "기간수": 0})
                else:
                    _wf_rows.append({
                        "점수": _slabel,
                        "평균 IC": round(_w["mean_ic"], 3),
                        "t값": round(_w["t_stat"], 2) if _w["t_stat"] is not None else None,
                        "IC 양수%": _w["pct_ic_positive"],
                        "상·하위⅓ 수익차(%p)": _w["mean_spread"],
                        "기간수": _w["n_periods"],
                    })
            st.dataframe(
                pd.DataFrame(_wf_rows), hide_index=True, use_container_width=True,
                column_config={
                    "평균 IC": st.column_config.Column(
                        help="점수 순서와 이후 수익 순서가 맞는 정도 (-1~+1). 0.03~0.10이면 쓸 만, "
                             "그 위면 좋음, 0 근처면 무관, 마이너스면 거꾸로."),
                    "t값": st.column_config.Column(
                        help="그 IC가 우연 아닌지. 절댓값 2 넘으면 '믿을 만'. 표본 적으면 작게 나옴."),
                    "IC 양수%": st.column_config.Column(
                        help="여러 날 중 점수가 제대로(플러스로) 작동한 날 비율. 50% 넘으면 대체로 맞는 쪽."),
                    "상·하위⅓ 수익차(%p)": st.column_config.Column(
                        help="점수 상위 1/3이 하위 1/3보다 평균 몇 %p 더 벌었는지. 클수록 잘 가름."),
                    "기간수": st.column_config.Column(
                        help="채점에 쓴 날짜(기간) 수. 적을수록 잠정이고, 4 미만이면 신뢰 어려움."),
                },
            )

            # 종합·집중 비교 + 표본 경고
            _tt, _tf = _wf["total"], _wf["focus"]
            _t_tot = _tt.get("t_stat") or 0.0
            _t_foc = _tf.get("t_stat") or 0.0
            _min_periods = min(_tt.get("n_periods", 0), _tf.get("n_periods", 0))
            if _tt["n_periods"] == 0 and _tf["n_periods"] == 0:
                st.info(
                    "검증 데이터가 부족합니다 — '한 날짜에 15종목 이상'이 "
                    f"'{verifier.WF_MIN_PERIODS}일 이상' 쌓여야 합니다. 매일 스크리닝이 돌면 자동 축적됩니다."
                )
            elif _tt.get("insufficient") or _tf.get("insufficient"):
                _better = "집중" if _t_foc > _t_tot else "종합"
                st.warning(
                    f"⚠️ 표본 부족(기간 {_min_periods}개) — **방향만** 참고하세요. "
                    f"지금은 '{_better}' 점수의 t값이 더 높지만, 숫자는 데이터가 더 쌓여야 믿을 만합니다."
                )
            elif _t_foc >= 2 and _t_foc > _t_tot:
                st.success(
                    f"✅ '집중' 점수가 종합보다 더 잘 예측합니다 "
                    f"(집중 t={_t_foc:+.2f} vs 종합 t={_t_tot:+.2f}). 집중 가설을 데이터가 지지합니다."
                )
            elif _t_tot >= 2:
                st.success(
                    f"✅ '종합' 점수가 {forward_days}일 수익을 유의하게 예측합니다 "
                    f"(종합 t={_t_tot:+.2f}, 집중 t={_t_foc:+.2f})."
                )
            else:
                st.info(
                    f"➖ 둘 다 통계적으로 뚜렷하지 않습니다 (종합 t={_t_tot:+.2f}, 집중 t={_t_foc:+.2f}). "
                    "선별 필터로만 쓰고 단일 종목 신호로 과신하지 마세요."
                )
            st.caption(
                f"{forward_days}일 뒤 시장 대비 기준. t값 |2| 이상이면 유의, IC 0.03~0.10이면 쓸 만한 신호. "
                "기간(날짜) 수가 적을수록 잠정이며, 같은 종목이 여러 날 겹쳐 실제 유효 표본은 더 작습니다."
            )
            st.divider()
        except Exception as _wf_err:
            st.caption(
                f"종합점수 walk-forward 검증을 표시하지 못했습니다 ({type(_wf_err).__name__}). "
                "방금 배포했다면 앱을 **Reboot** 하면 나타납니다."
            )

        item_result = verifier.verify_item_scores(forward_days=forward_days)

        _lvl_i, _msg_i = _verdict_item(item_result)
        _emit_verdict(_lvl_i, _msg_i)

        st.caption(
            f"전체 관측 {item_result['total_samples']:,}건 · 기간 {item_result.get('n_dates', 0)}일 · "
            f"종목 {item_result.get('n_stocks', 0)}개 "
            f"({'시장 대비 초과수익' if item_result.get('use_excess') else '절대수익'} 기준). "
            "기간이 짧으면 같은 종목의 겹치는 구간이 많아 실제 유효 표본은 더 작습니다."
        )

        if item_result["total_samples"] == 0:
            st.info(
                f"분석에 필요한 데이터가 부족합니다. 적어도 {forward_days + 5}일 이상 "
                f"매일 분석을 돌리거나, 분석된 종목 수를 늘려야 의미 있는 통계가 나옵니다."
            )
        else:
            ranked = item_result["ranked_items"]
            spread_rows = []
            for item in ranked:
                st_ = item_result["item_stats"][item]
                spread_rows.append({
                    "항목": item,
                    "예측력 IC": st_["rank_ic"] if st_.get("rank_ic") is not None else "표본부족",
                    "spread(%p)": st_["predictive_spread"] if st_["predictive_spread"] is not None else "-",
                    "관측 수": st_.get("n", st_["total_samples"]),
                })
            st.markdown(f"##### 항목별 예측력 ({forward_days}일 뒤, 시장 대비 기준)")
            st.dataframe(
                pd.DataFrame(spread_rows), hide_index=True, use_container_width=True,
                column_config={
                    "예측력 IC": st.column_config.Column(
                        help="그 항목 점수와 이후 수익이 순서대로 맞는 정도 (-1~+1). "
                             "+면 점수 높을수록 더 오름, 0 근처면 무관, 마이너스면 거꾸로. "
                             "0.03~0.10이면 쓸 만.",
                    ),
                    "spread(%p)": st.column_config.Column(
                        help="그 항목 점수가 플러스일 때 평균수익에서 마이너스일 때 "
                             "평균수익을 뺀 값. 클수록 그 항목이 수익을 잘 가른다는 뜻.",
                    ),
                    "관측 수": st.column_config.Column(
                        help="계산에 쓴 (종목×날짜) 기록 수. 30건 미만이면 '표본부족'으로 "
                             "표시되고 믿기 어려움.",
                    ),
                },
            )

            st.markdown("##### 항목 상세 — 점수값별 수익률")
            for item in ranked:
                st_ = item_result["item_stats"][item]
                _ic = st_.get("rank_ic")
                _ic_txt = f"IC {_ic:+.2f}" if _ic is not None else "IC 표본부족"
                with st.expander(f"{item}  ({_ic_txt}, {st_.get('n', st_['total_samples'])}건)"):
                    rows = []
                    for score_val, stat in st_["by_score"].items():
                        rows.append({
                            "점수값": f"{score_val:+d}",
                            "케이스 수": stat["count"],
                            "평균(%)": stat["avg_return"] if stat["avg_return"] is not None else "-",
                            "중앙값(%)": stat.get("median_return") if stat.get("median_return") is not None else "-",
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
        screen_req["universe"],
        screen_req["min_score"],
        stock_dict,
        deep_screen=screen_req.get("deep_screen", True),
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
        st.caption("1단계: 빠른 스크리닝 (가격·수급·재무·룰 기반 공시)")
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

            st.markdown("#### 💡 통과 0개 — 왜, 그리고 무엇이 추적되나")
            _reg = _screen_runner.get("regime") or {}
            _overheated = bool(_screen_runner.get("overheated") or _reg.get("overheated"))
            _risk_off = bool(_reg.get("risk_off"))
            _adaptive = _screen_runner.get("adaptive_picks") or []
            _momentum = _screen_runner.get("momentum_picks") or []
            _base_min = _screen_runner.get("min_score_base", min_score)

            _notes: list[str] = []
            if _overheated:
                _notes.append(
                    f"- **과열 국면**입니다 ({_reg.get('label', 'KOSPI 장기선 위 과열')}). "
                    "대부분 종목이 이미 급등해 **고점 추격 회피 필터에서 점수 이전에 자동 제외**되므로, "
                    "통과 0개는 비정상이 아니라 **보수적으로 맞는 신호**일 때가 많습니다."
                )
                if _adaptive:
                    _names = ", ".join(r.get("name", r.get("code", "")) for r in _adaptive)
                    _notes.append(
                        f"- 대신 시장을 적정하게 이기는 **건전 주도주 {len(_adaptive)}개**를 "
                        f"`적응 통과(adaptive)`로 선별해 추적 중입니다 — {_names}. "
                        "**📊 점수 시뮬레이션**에서 성과가 검증돼요."
                    )
                else:
                    _notes.append(
                        "- 이번엔 시장 대비 초과가 적정한 **건전 주도주(adaptive)도 없어**, "
                        "추격 가치가 낮다고 본 상태입니다."
                    )
                if _momentum:
                    _notes.append(
                        f"- 급등으로 제외됐지만 주도주 성격인 **{len(_momentum)}개**는 "
                        "`관찰`로 함께 추적합니다 (매수 추천 아님, 사후 검증용)."
                    )
            elif _risk_off:
                _notes.append(
                    f"- **하락장 리스크오프**라 진입 기준을 {_base_min:+d} → {min_score:+d}점으로 "
                    "의도적으로 높였습니다. 추세 역행 매수를 줄여 **손실을 방어**하려는 설계된 "
                    "동작이라, 통과 0개가 정상입니다."
                )

            _notes.append("- 더 넓은 유니버스(KOSPI TOP 50 등)로 그물을 넓혀볼 수 있습니다.")
            if excluded_stale > 0:
                _notes.append(
                    "- 분기 보고서가 등록되면 재무 stale로 제외된 종목들이 후보에 다시 들어옵니다."
                )
            _notes.append(
                f"- 기준 종합점수는 수익률 보호를 위해 **{_base_min:+d}점으로 고정**"
                "(하락장에선 자동 상향)되어 있어 화면에서 임의로 낮출 수 없습니다 — "
                "기준을 낮추면 약한 후보가 들어와 오히려 손실 위험이 커지기 때문입니다."
            )
            st.markdown("\n".join(_notes))
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
