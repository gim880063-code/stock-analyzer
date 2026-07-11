"""매매기록 · 수익률 페이지 — 한국+미국 주식 매매 장부와 월간/연간 수익률.

app.py 와 독립된 새 페이지: 여기서 크래시가 나도 메인 분석 화면은 영향 없다.
데이터는 Gist(trades.json)에 저장되어 재배포 후에도 계속 누적된다.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

import cloud_store
import journal
import quotes


st.set_page_config(
    page_title="매매기록 · 수익률",
    page_icon="https://abs.twimg.com/emoji/v2/72x72/1f4d2.png",
    layout="wide",
)


def _check_password() -> bool:
    """app.py 와 동일한 비밀번호 보호 — 세션 키를 공유해 한 번만 입력."""
    try:
        configured = st.secrets.get("APP_PASSWORD", "")
    except (FileNotFoundError, AttributeError, Exception):
        return True
    if not configured:
        return True
    if st.session_state.get("_authenticated"):
        return True
    st.markdown("## 🔒 매매기록 · 수익률")
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

st.title("📒 매매기록 · 수익률")
st.caption(
    "한국·미국 주식 매매를 기록하면 보유 현황과 월간·연간 수익률을 자동 계산합니다. "
    "미국 주식은 체결 환율로 원화 환산(환차손익 포함). 데이터는 Gist에 저장되어 계속 누적됩니다."
)


@st.cache_data(ttl=86400, show_spinner=False)
def _krx_names() -> dict[str, str]:
    """KRX 전 종목 이름맵 (코드→이름). 실패 시 예외 → 캐시 안 됨 → 다음에 재시도."""
    from analyzer import all_korean_stocks
    return all_korean_stocks()


GREEN, RED = "#1f7a3a", "#a3201a"


def _color_pnl(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return f"color: {GREEN}" if v > 0 else (f"color: {RED}" if v < 0 else "")


# ─────────── 데이터 로드·계산 ───────────
try:
    trades = journal.load_trades()
    splits = journal.load_splits()
    incomes = journal.load_incomes()
except Exception as e:
    st.error(f"매매내역을 불러오지 못했습니다: {type(e).__name__}: {e}")
    st.stop()

# 액면분할 반영: 분할 이전 매매를 현재 주수 기준으로 환산 (금액 보존)
# → 시세(수정주가)와 기준이 맞아 보유현황·수익률이 증권사 앱과 일치.
adj_trades = journal.adjust_trades_for_splits(trades, splits)
positions, realized, warnings = journal.compute_positions(adj_trades)
fx_now = quotes.current_usdkrw()

monthly = []
if trades:
    first_date = datetime.strptime(journal.sorted_trades(trades)[0]["date"], "%Y-%m-%d").date()
    with st.spinner("가격·환율 데이터 조회 중..."):
        fx_series = quotes.usdkrw_history(first_date - timedelta(days=10))

        def _price_fn(market: str, code: str):
            return quotes.price_history(market, code, first_date - timedelta(days=10))

        try:
            monthly = journal.compute_monthly_returns(
                adj_trades, _price_fn, fx_series, incomes=incomes,
            )
        except Exception as e:
            st.warning(f"수익률 계산 실패 (가격 데이터 문제일 수 있음): {type(e).__name__}: {e}")

yearly = journal.yearly_returns(monthly)
realized_monthly, realized_yearly = journal.realized_by_period(realized)
incomes_monthly, incomes_yearly = journal.incomes_by_period(incomes)
# '확정손익' = 매매 실현손익 + 세후 배당 + 기타 수입 − 세금·비용
confirmed_monthly = journal.merge_period_sums(realized_monthly, incomes_monthly)
confirmed_yearly = journal.merge_period_sums(realized_yearly, incomes_yearly)
total_dividends = sum(
    journal.income_net_krw(e) for e in incomes if e["type"] == "dividend"
)

# 보유 현황 평가 (현재가 기준)
rows = []
total_cost_krw = 0.0
total_value_krw = 0.0
unpriced = 0
for (market, code), p in sorted(positions.items(), key=lambda kv: (kv[0][0], kv[1]["name"])):
    px = quotes.current_price(market, code)
    fx = (fx_now or 0.0) if market == "US" else 1.0
    cost_krw = p["qty"] * p["avg_krw"]
    if px is None or (market == "US" and not fx):
        unpriced += 1
        value_krw = cost_krw  # 가격 조회 실패 시 매입가로 평가 (손익 0 처리)
        px_disp = None
    else:
        value_krw = p["qty"] * px * fx
        px_disp = px
    total_cost_krw += cost_krw
    total_value_krw += value_krw
    rows.append({
        "시장": "🇰🇷 한국" if market == "KR" else "🇺🇸 미국",
        "종목": f"{p['name']} ({code})",
        "수량": p["qty"],
        "평균단가": p["avg_local"],
        "현재가": px_disp,
        "매입금(원)": cost_krw,
        "평가액(원)": value_krw,
        "평가손익(원)": value_krw - cost_krw,
        "수익률%": (value_krw / cost_krw - 1) * 100 if cost_krw > 0 else None,
    })
for r in rows:
    r["비중%"] = (r["평가액(원)"] / total_value_krw * 100) if total_value_krw > 0 else None

# ─────────── 요약 ───────────
total_pnl = total_value_krw - total_cost_krw
total_realized = sum(r["pnl_krw"] for r in realized)
total_confirmed = total_realized + sum(journal.income_net_krw(e) for e in incomes)
this_year = journal.today_kst().strftime("%Y")
year_now = next((y for y in yearly if y["year"] == this_year), None)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총 평가액", f"{total_value_krw:,.0f}원")
c2.metric(
    "평가손익 (보유분)",
    f"{total_pnl:+,.0f}원",
    f"{(total_value_krw / total_cost_krw - 1) * 100:+.2f}%" if total_cost_krw > 0 else None,
)
c3.metric(
    "누적 확정손익 (매매+배당)", f"{total_confirmed:+,.0f}원",
    f"{confirmed_yearly.get(this_year, 0.0):+,.0f}원 (올해)" if (realized or incomes) else None,
)
c4.metric(
    f"{this_year}년 수익률",
    f"{year_now['ret'] * 100:+.2f}%" if year_now and year_now["ret"] is not None else "-",
    f"{year_now['pnl_krw']:+,.0f}원" if year_now else None,
)
c5.metric("환율 USD/KRW", f"{fx_now:,.2f}원" if fx_now else "조회 실패")
st.caption(
    "가격은 최근 종가 기준(실시간 아님). 미국 주식 평가는 현재 시장환율로 환산합니다 — "
    "증권사(삼성증권) 앱의 평가금액과는 고시환율·스프레드 차이로 소폭 다를 수 있습니다. "
    "매매 자체는 입력한 체결 환율로 계산되므로 실현손익은 계좌와 일치합니다."
)
if unpriced:
    st.warning(f"⚠️ {unpriced}개 종목의 현재가 조회에 실패해 매입가로 평가했습니다. (티커 확인 필요)")
for w in warnings:
    st.warning(f"⚠️ {w}")

# ─────────── 매매 입력 ───────────
if not trades:
    st.info(
        "**처음 시작하기** — 지금 보유 중인 종목부터 등록하세요. "
        "실제 매수일과 평균단가로 '매수' 기록을 넣으면 그 시점부터의 수익률이 자동 계산됩니다. "
        "미국 주식은 매수일의 환율이 자동으로 채워집니다(수정 가능)."
    )

with st.expander("➕ 매매 입력 (매수/매도)", expanded=not trades):
    # 저장 성공 시 세대(gen)를 올려 수량·단가·수수료 입력칸을 깨끗이 비운다
    # (직전 입력값이 남아 실수로 중복 저장·잘못된 환율 역산되는 걸 방지).
    gen = st.session_state.setdefault("trade_form_gen", 0)
    in1, in2, in3 = st.columns([1, 1, 2])
    with in1:
        market_kor = st.radio("시장", ["🇰🇷 한국", "🇺🇸 미국"], horizontal=True, key="in_market")
        market = "KR" if market_kor.endswith("한국") else "US"
        side_kor = st.radio("구분", ["매수", "매도"], horizontal=True, key="in_side")
        side = "buy" if side_kor == "매수" else "sell"
    with in2:
        trade_date = st.date_input(
            "체결일", value=journal.today_kst(),
            min_value=datetime(2000, 1, 1).date(), max_value=journal.today_kst(),
            key="in_date",
        )
        qty = st.number_input("수량 (주)", min_value=0.0, step=1.0, value=0.0, key=f"in_qty_{gen}")
    with in3:
        code, name = "", ""
        if market == "KR":
            try:
                stock_dict = _krx_names()
            except Exception:
                stock_dict = {}
            if stock_dict:
                opts = [f"{n} {c}" for c, n in sorted(stock_dict.items(), key=lambda kv: kv[1])]
                sel = st.selectbox(
                    "종목 검색 (한글 이름 또는 코드)", options=opts, key="in_kr_stock",
                    help="입력창에 '삼성', '005930' 등을 타이핑하면 검색됩니다",
                )
                code = sel.rsplit(maxsplit=1)[-1]
                name = stock_dict.get(code, code)
            else:
                code = st.text_input("종목코드 (6자리)", max_chars=6, key="in_kr_code").strip()
                name = st.text_input("종목명", key="in_kr_name").strip() or code
            price = st.number_input("단가 (원)", min_value=0.0, step=100.0, value=0.0, key=f"in_price_kr_{gen}")
            fx, fee_label = 1.0, "수수료+세금 (원)"
        else:
            code = st.text_input("티커 (예: AAPL, TSLA)", key="in_us_ticker").strip().upper()
            name = code
            price = st.number_input("단가 (달러)", min_value=0.0, step=0.01, value=0.0,
                                    format="%.2f", key=f"in_price_us_{gen}")
            fx_default = quotes.usdkrw_at(trade_date) or fx_now or 1300.0
            # key 에 날짜를 넣어 체결일을 바꾸면 그 날짜의 환율이 새로 채워지게 함
            fx = st.number_input(
                "체결 환율 (원/달러) — 시장환율 자동 조회, 수정 가능",
                min_value=0.0, step=0.1, value=float(fx_default), format="%.2f",
                key=f"in_fx_{gen}_{trade_date.isoformat()}",
            )
            # 삼성증권 등 증권사는 자체 고시환율(스프레드 포함)로 환전하므로 시장환율과
            # 다르다. 증권사 앱의 원화 결제금액을 그대로 넣으면 실제 적용 환율을 역산.
            krw_amount = st.number_input(
                "원화 결제금액 (선택) — 증권사 앱에 표시된 원화 금액",
                min_value=0.0, step=1000.0, value=0.0, format="%.0f",
                key=f"in_krw_{gen}",
                help="삼성증권 앱의 체결내역에 나오는 원화 금액(수수료 제외 체결금액)을 입력하면 "
                     "환율을 자동 역산해 실제 계좌와 정확히 일치시킵니다. "
                     "매수는 결제한 원화, 매도는 수령한 원화 기준. 비워두면 위 환율을 사용.",
            )
            if krw_amount > 0 and qty > 0 and price > 0:
                fx = krw_amount / (qty * price)
                st.caption(f"→ 적용 환율 {fx:,.2f}원/달러 (원화 결제금액에서 역산)")
            fee_label = "수수료 (달러)"
        fee = st.number_input(fee_label, min_value=0.0, step=0.01, value=0.0, key=f"in_fee_{gen}")
        note = st.text_input("메모 (선택)", key=f"in_note_{gen}")

    if st.button("💾 기록 저장", type="primary", use_container_width=True):
        try:
            t = journal.add_trade({
                "date": trade_date.isoformat(), "market": market, "code": code,
                "name": name, "side": side, "qty": qty, "price": price,
                "fx": fx, "fee": fee, "note": note,
            })
            st.success(f"저장됨: {t['date']} {t['name']} {side_kor} {t['qty']:g}주 @ {t['price']:,g}")
            st.session_state["trade_form_gen"] = gen + 1  # 입력칸 초기화
            st.rerun()
        except ValueError as e:
            st.error(f"입력 오류: {e}")
        except Exception as e:
            st.error(f"저장 실패: {type(e).__name__}: {e}")

# ─────────── 배당금·세금·기타 입력 ───────────
with st.expander(f"💵 배당금·세금·기타 입력 ({len(incomes)}건)"):
    st.caption(
        "배당금은 세전 금액을 넣으면 원천징수세(한국 15.4% / 미국 15%)를 자동 계산해 "
        "세후 원화로 집계됩니다. 해외주식 양도소득세, 출금·이체 수수료 같은 비용도 "
        "'세금·비용'으로 기록하면 확정손익에서 차감됩니다."
    )
    igen = st.session_state.setdefault("income_form_gen", 0)
    inc_type_kor = st.radio(
        "유형", ["💰 배당금", "🧾 세금·비용", "➕ 기타 수입"],
        horizontal=True, key="inc_type",
    )
    j1, j2 = st.columns([1, 2])
    with j1:
        inc_date = st.date_input(
            "날짜", value=journal.today_kst(),
            min_value=datetime(2000, 1, 1).date(), max_value=journal.today_kst(),
            key="inc_date",
        )
    inc_payload = None
    if inc_type_kor.endswith("배당금"):
        with j1:
            inc_market_kor = st.radio("시장", ["🇰🇷 한국", "🇺🇸 미국"], horizontal=True, key="inc_market")
            inc_market = "KR" if inc_market_kor.endswith("한국") else "US"
        with j2:
            inc_code, inc_name = "", ""
            if inc_market == "KR":
                try:
                    _sd2 = _krx_names()
                except Exception:
                    _sd2 = {}
                if _sd2:
                    _opts2 = [f"{n} {c}" for c, n in sorted(_sd2.items(), key=lambda kv: kv[1])]
                    _sel2 = st.selectbox("종목", options=_opts2, key="inc_kr_stock")
                    inc_code = _sel2.rsplit(maxsplit=1)[-1]
                    inc_name = _sd2.get(inc_code, inc_code)
                else:
                    inc_code = st.text_input("종목코드 (6자리)", max_chars=6, key="inc_kr_code").strip()
                    inc_name = inc_code
            else:
                inc_code = st.text_input("티커 (예: AAPL)", key="inc_us_ticker").strip().upper()
                inc_name = inc_code
            inc_amount = st.number_input(
                "배당금 총액 (세전, " + ("원)" if inc_market == "KR" else "달러)"),
                min_value=0.0, step=1.0, value=0.0, format="%.2f", key=f"inc_amt_{igen}",
            )
            _rate = journal.DIVIDEND_TAX_RATE[inc_market]
            auto_tax = st.checkbox(
                f"원천징수세 자동 계산 ({_rate * 100:.1f}%)", value=True, key="inc_autotax",
            )
            if auto_tax:
                inc_tax = round(inc_amount * _rate, 2)
            else:
                inc_tax = st.number_input(
                    "원천징수세 (실제 떼인 금액)", min_value=0.0, step=0.01,
                    value=0.0, format="%.2f", key=f"inc_tax_{igen}",
                )
            inc_fx = 1.0
            if inc_market == "US":
                _fx_def = quotes.usdkrw_at(inc_date) or fx_now or 1300.0
                inc_fx = st.number_input(
                    "환율 (원/달러) — 자동 조회, 수정 가능",
                    min_value=0.0, step=0.1, value=float(_fx_def), format="%.2f",
                    key=f"inc_fx_{igen}_{inc_date.isoformat()}",
                )
            if inc_amount > 0:
                _net = (inc_amount - inc_tax) * inc_fx
                st.caption(f"→ 세후 수령액 {_net:,.0f}원 (세금 {inc_tax:,.2f} 차감)")
        inc_payload = {
            "type": "dividend", "date": inc_date.isoformat(), "market": inc_market,
            "code": inc_code, "name": inc_name, "amount": inc_amount,
            "tax": inc_tax, "fx": inc_fx,
        }
    else:
        is_expense = "세금" in inc_type_kor
        with j2:
            inc_label = st.text_input(
                "설명",
                placeholder="예: 해외주식 양도소득세, 출금 수수료" if is_expense else "예: 이벤트 지원금, 이자",
                key=f"inc_label_{igen}",
            )
            inc_cur_kor = st.radio("통화", ["원화", "달러"], horizontal=True, key="inc_cur")
            inc_currency = "KRW" if inc_cur_kor == "원화" else "USD"
            inc_amount = st.number_input(
                "금액 (" + ("원)" if inc_currency == "KRW" else "달러)"),
                min_value=0.0, step=1.0, value=0.0, format="%.2f", key=f"inc_amt2_{igen}",
            )
            inc_fx = 1.0
            if inc_currency == "USD":
                _fx_def2 = quotes.usdkrw_at(inc_date) or fx_now or 1300.0
                inc_fx = st.number_input(
                    "환율 (원/달러) — 자동 조회, 수정 가능",
                    min_value=0.0, step=0.1, value=float(_fx_def2), format="%.2f",
                    key=f"inc_fx2_{igen}_{inc_date.isoformat()}",
                )
        inc_payload = {
            "type": "expense" if is_expense else "income",
            "date": inc_date.isoformat(), "name": inc_label,
            "amount": inc_amount, "currency": inc_currency, "fx": inc_fx,
        }

    if st.button("💾 내역 저장", type="primary", use_container_width=True, key="inc_save"):
        try:
            e = journal.add_income(inc_payload)
            st.success(f"저장됨: {e['date']} {e['name']} {journal.income_net_krw(e):+,.0f}원")
            st.session_state["income_form_gen"] = igen + 1
            st.rerun()
        except ValueError as e:
            st.error(f"입력 오류: {e}")
        except Exception as e:
            st.error(f"저장 실패: {type(e).__name__}: {e}")

# ─────────── 액면분할 기록 ───────────
with st.expander(f"🔀 액면분할 기록 ({len(splits)}건) — 분할·병합된 종목이 있으면 등록"):
    st.caption(
        "예: 테슬라 2022-08-25 1주→3주, 삼성전자 2018-05-04 1주→50주. "
        "등록하면 **분할일 이전에 입력한 매매**의 수량·단가를 현재 기준으로 자동 환산합니다 "
        "(투자금액·손익은 그대로 보존). 분할일 이후 입력분은 이미 새 기준이므로 건드리지 않습니다. "
        "병합(감자)은 1주→0.1주처럼 1보다 작은 값으로 입력하세요."
    )
    sp1, sp2, sp3 = st.columns([1, 2, 1])
    with sp1:
        sp_market_kor = st.radio("시장", ["🇰🇷 한국", "🇺🇸 미국"], horizontal=True, key="sp_market")
        sp_market = "KR" if sp_market_kor.endswith("한국") else "US"
    with sp2:
        sp_code, sp_name = "", ""
        if sp_market == "KR":
            try:
                _sd = _krx_names()
            except Exception:
                _sd = {}
            if _sd:
                _opts = [f"{n} {c}" for c, n in sorted(_sd.items(), key=lambda kv: kv[1])]
                _sel = st.selectbox("종목", options=_opts, key="sp_kr_stock")
                sp_code = _sel.rsplit(maxsplit=1)[-1]
                sp_name = _sd.get(sp_code, sp_code)
            else:
                sp_code = st.text_input("종목코드 (6자리)", max_chars=6, key="sp_kr_code").strip()
                sp_name = sp_code
        else:
            sp_code = st.text_input("티커 (예: TSLA)", key="sp_us_ticker").strip().upper()
            sp_name = sp_code
    with sp3:
        sp_date = st.date_input(
            "분할 기준일", value=journal.today_kst(),
            min_value=datetime(2000, 1, 1).date(), max_value=journal.today_kst(),
            key="sp_date",
        )
        sp_ratio = st.number_input(
            "1주 → 몇 주?", min_value=0.0001, step=1.0, value=2.0, format="%.4f",
            key="sp_ratio", help="분할 3:1이면 3, 50:1이면 50. 병합 1:10이면 0.1",
        )
    if st.button("🔀 분할 기록 저장", use_container_width=True, key="sp_save"):
        try:
            s = journal.add_split({
                "market": sp_market, "code": sp_code, "name": sp_name,
                "date": sp_date.isoformat(), "ratio": sp_ratio,
            })
            st.success(f"저장됨: {s['name']} {s['date']} 1주→{s['ratio']:g}주")
            st.rerun()
        except ValueError as e:
            st.error(f"입력 오류: {e}")
        except Exception as e:
            st.error(f"저장 실패: {type(e).__name__}: {e}")

    if splits:
        st.divider()
        for s in sorted(splits, key=lambda x: x.get("date", "")):
            c_txt, c_del = st.columns([6, 1])
            c_txt.markdown(
                f"<small>{'🇰🇷' if s['market'] == 'KR' else '🇺🇸'} "
                f"<b>{s['name']}</b> ({s['code']}) · {s['date']} · 1주→{s['ratio']:g}주</small>",
                unsafe_allow_html=True,
            )
            if c_del.button("✖", key=f"sp_del_{s['id']}", help="분할 기록 삭제"):
                journal.delete_splits({s["id"]})
                st.rerun()

# ─────────── 보유 현황 ───────────
st.subheader("💼 보유 현황")
if rows:
    df_pos = pd.DataFrame(rows)
    styled = df_pos.style.map(_color_pnl, subset=["평가손익(원)", "수익률%"]).format({
        "수량": "{:,.0f}",
        "평균단가": "{:,.2f}",
        "현재가": lambda v: "-" if v is None or pd.isna(v) else f"{v:,.2f}",
        "매입금(원)": "{:,.0f}",
        "평가액(원)": "{:,.0f}",
        "평가손익(원)": "{:+,.0f}",
        "수익률%": "{:+.2f}",
        "비중%": "{:.1f}",
    }, na_rep="-")
    st.dataframe(styled, use_container_width=True, hide_index=True)
    _split_note = " 액면분할 기록이 반영된 수량·평단입니다." if splits else ""
    st.caption(
        "평균단가·현재가는 매매 통화 기준(한국=원, 미국=달러). 매입금·평가액·손익은 원화 환산."
        + _split_note
    )
else:
    st.caption("보유 중인 종목이 없습니다. 위에서 매매를 입력하세요.")

# ─────────── 실현손익 (매매·배당 확정 손익) ───────────
st.subheader("💰 실현손익 — 매매·배당으로 확정한 손익")
if realized or incomes:
    _ym_now = journal.today_kst().strftime("%Y-%m")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("이번 달 확정손익", f"{confirmed_monthly.get(_ym_now, 0.0):+,.0f}원")
    r2.metric(f"{this_year}년 확정손익", f"{confirmed_yearly.get(this_year, 0.0):+,.0f}원")
    r3.metric("누적 확정손익", f"{total_confirmed:+,.0f}원")
    r4.metric("누적 배당 (세후)", f"{total_dividends:+,.0f}원")
    st.caption("확정손익 = 매매 실현손익 + 세후 배당 + 기타 수입 − 세금·비용 (원화)")

    tab_rt, tab_rg, tab_rs, tab_rd, tab_ri = st.tabs(
        ["월간·연간 표", "누적·월별 그래프", "종목별", "매도 내역", "배당·기타 내역"]
    )

    with tab_rt:
        # 연도 × 월 피벗 + 연간 열 (단위: 원)
        r_years = sorted({k[:4] for k in confirmed_monthly})
        r_table = {}
        for y in r_years:
            row = {f"{mm}월": confirmed_monthly.get(f"{y}-{mm:02d}") for mm in range(1, 13)}
            row["연간"] = confirmed_yearly.get(y)
            r_table[y] = row
        df_rt = pd.DataFrame(r_table).T
        styled_rt = df_rt.style.map(_color_pnl).format("{:+,.0f}", na_rep="")
        st.dataframe(styled_rt, use_container_width=True)
        st.caption(
            "단위: 원. 매매 실현손익(이동평균법·수수료 차감·환차손익 포함) + 세후 배당 + "
            "기타 수입 − 세금·비용. 내역이 있는 달만 값이 표시됩니다."
        )

    with tab_rg:
        s_rm = journal.period_series(confirmed_monthly)
        if len(s_rm) > 0:
            st.markdown("**누적 확정손익 (원)** — 매매·배당으로 확정한 손익이 쌓여온 흐름")
            st.line_chart(s_rm.cumsum().rename("누적 확정손익(원)"), height=260)
            st.markdown("**월별 확정손익 (원)**")
            bar = pd.Series(
                s_rm.values, index=[d.strftime("%Y-%m") for d in s_rm.index],
                name="확정손익(원)",
            )
            st.bar_chart(bar, height=220)

    with tab_rs:
        by_sym = journal.realized_by_symbol(realized)
        div_sym = journal.dividends_by_symbol(incomes)
        _seen = set()
        rs_rows = []
        for a in by_sym:
            key = (a["market"], a["code"])
            _seen.add(key)
            div = div_sym.get(key, {}).get("net_krw", 0.0)
            rs_rows.append({
                "시장": "🇰🇷" if a["market"] == "KR" else "🇺🇸",
                "종목": f"{a['name']} ({a['code']})",
                "매도 횟수": a["sells"],
                "매매손익(원)": a["pnl_krw"],
                "배당 세후(원)": div or None,
                "합계(원)": a["pnl_krw"] + div,
                "실현수익률%": a["ret"] * 100 if a["ret"] is not None else None,
            })
        for key, dv in div_sym.items():  # 매도 없이 배당만 받은 종목
            if key in _seen:
                continue
            rs_rows.append({
                "시장": "🇰🇷" if key[0] == "KR" else "🇺🇸",
                "종목": f"{dv['name']} ({key[1]})",
                "매도 횟수": 0,
                "매매손익(원)": None,
                "배당 세후(원)": dv["net_krw"],
                "합계(원)": dv["net_krw"],
                "실현수익률%": None,
            })
        rs_rows.sort(key=lambda r: r["합계(원)"], reverse=True)
        if rs_rows:
            df_rs = pd.DataFrame(rs_rows)
            styled_rs = df_rs.style.map(
                _color_pnl, subset=["매매손익(원)", "배당 세후(원)", "합계(원)", "실현수익률%"],
            ).format(
                {"매매손익(원)": "{:+,.0f}", "배당 세후(원)": "{:+,.0f}",
                 "합계(원)": "{:+,.0f}", "실현수익률%": "{:+.2f}"}, na_rep="-",
            )
            st.dataframe(styled_rs, use_container_width=True, hide_index=True)
            st.caption("어떤 종목에서 벌고 잃었는지 — 실현수익률 = 매매손익 ÷ 매도한 수량의 매입원가(원화).")
        else:
            st.caption("매도나 배당이 있는 종목이 아직 없습니다.")

    with tab_rd:
        if realized:
            df_r = pd.DataFrame([
                {"매도일": r["date"],
                 "시장": "🇰🇷" if r["market"] == "KR" else "🇺🇸",
                 "종목": f"{r['name']} ({r['code']})",
                 "수량": r["qty"],
                 "매도금액(원)": r["proceeds_krw"],
                 "실현손익(원)": r["pnl_krw"],
                 "실현수익률%": (r["pnl_krw"] / r["cost_krw"] * 100) if r["cost_krw"] > 0 else None}
                for r in reversed(realized)
            ])
            styled_r = df_r.style.map(_color_pnl, subset=["실현손익(원)", "실현수익률%"]).format(
                {"수량": "{:,.0f}", "매도금액(원)": "{:,.0f}",
                 "실현손익(원)": "{:+,.0f}", "실현수익률%": "{:+.2f}"}, na_rep="-",
            )
            st.dataframe(styled_r, use_container_width=True, hide_index=True)
            st.caption("매도 1건마다의 확정 손익입니다.")
        else:
            st.caption("아직 매도 기록이 없습니다.")

    with tab_ri:
        if incomes:
            _type_kor = {"dividend": "💰 배당", "expense": "🧾 세금·비용", "income": "➕ 기타 수입"}
            inc_ordered = sorted(incomes, key=lambda e: e.get("date", ""), reverse=True)
            df_i = pd.DataFrame([
                {"삭제": False,
                 "날짜": e["date"],
                 "유형": _type_kor.get(e["type"], e["type"]),
                 "종목/설명": e["name"],
                 "금액": e["amount"],
                 "세금": e["tax"] if e["type"] == "dividend" else None,
                 "통화": "원" if e["currency"] == "KRW" else "달러",
                 "환율": e["fx"] if e["currency"] == "USD" else None,
                 "원화 효과": journal.income_net_krw(e)}
                for e in inc_ordered
            ])
            edited_i = st.data_editor(
                df_i,
                use_container_width=True,
                hide_index=True,
                disabled=[c for c in df_i.columns if c != "삭제"],
                column_config={
                    "삭제": st.column_config.CheckboxColumn(width="small"),
                    "금액": st.column_config.NumberColumn(format="%,.2f"),
                    "세금": st.column_config.NumberColumn(format="%,.2f"),
                    "환율": st.column_config.NumberColumn(format="%,.2f"),
                    "원화 효과": st.column_config.NumberColumn(format="%,.0f"),
                },
                key=f"inc_editor_{hash(tuple(e['id'] for e in inc_ordered))}",
            )
            checked_i = [i for i, v in enumerate(edited_i["삭제"].tolist()) if v]
            if checked_i:
                if st.button(f"🗑️ 선택한 {len(checked_i)}건 삭제", key="inc_del_btn"):
                    _ids = {inc_ordered[i]["id"] for i in checked_i}
                    journal.delete_incomes(_ids)
                    st.rerun()
            st.caption("'원화 효과'는 확정손익에 더해지는 값 — 배당은 세후(+), 세금·비용은 (−).")
        else:
            st.caption("아직 배당·세금 기록이 없습니다 — 위의 '배당금·세금·기타 입력'에서 추가하세요.")
else:
    st.caption("아직 매도·배당 기록이 없습니다 — 매도나 배당을 입력하면 여기에 월간·연간·종목별로 집계됩니다.")

# ─────────── 수익률 ───────────
st.subheader("📈 수익률 (월간·연간)")
if monthly:
    tab_ret, tab_cum, tab_pnl = st.tabs(["월간 수익률 표", "누적 수익률", "월간 손익(원)"])

    with tab_ret:
        # 연도 × 월 피벗 + 연간 열
        years = sorted({m["ym"][:4] for m in monthly})
        table = {}
        for y in years:
            row = {f"{mm}월": None for mm in range(1, 13)}
            for m in monthly:
                if m["ym"][:4] == y and m["ret"] is not None:
                    row[f"{int(m['ym'][5:7])}월"] = m["ret"] * 100
            yr = next((v for v in yearly if v["year"] == y), None)
            row["연간"] = yr["ret"] * 100 if yr and yr["ret"] is not None else None
            table[y] = row
        df_ret = pd.DataFrame(table).T
        styled_ret = df_ret.style.map(_color_pnl).format("{:+.2f}%", na_rep="")
        st.dataframe(styled_ret, use_container_width=True)
        st.caption(
            "월중 매수·매도 금액을 기간 가중해 반영한 수익률(Modified Dietz) — "
            "'주식에 들어가 있던 돈' 대비 성과입니다. 이번 달은 오늘까지 기준."
        )

    with tab_cum:
        curve = journal.cumulative_curve(monthly)
        if len(curve) > 0:
            st.line_chart(curve, height=280)
            st.caption(f"첫 기록({monthly[0]['ym']}) 이후 월간 수익률을 누적 연결한 값입니다.")

    with tab_pnl:
        pnl_series = pd.Series(
            [m["pnl_krw"] for m in monthly],
            index=[m["ym"] for m in monthly], name="월간 손익(원)",
        )
        st.bar_chart(pnl_series, height=280)
        df_y = pd.DataFrame([
            {"연도": y["year"],
             "손익(원)": y["pnl_krw"],
             "수익률": y["ret"] * 100 if y["ret"] is not None else None}
            for y in yearly
        ])
        styled_y = df_y.style.map(_color_pnl, subset=["손익(원)", "수익률"]).format(
            {"손익(원)": "{:+,.0f}", "수익률": "{:+.2f}%"}, na_rep="-",
        )
        st.dataframe(styled_y, use_container_width=True, hide_index=True)
        st.caption("손익 = 실현손익 + 보유분 평가손익 변동 (원화).")
else:
    st.caption("매매를 입력하면 첫 매매가 있는 달부터 수익률이 계산됩니다.")

# ─────────── 매매내역 ───────────
st.subheader(f"📋 매매내역 ({len(trades)}건)")
if trades:
    realized_by_id = {r["trade_id"]: r["pnl_krw"] for r in realized}
    ordered = list(reversed(journal.sorted_trades(trades)))  # 최신이 위로
    df_hist = pd.DataFrame([
        {"삭제": False,
         "날짜": t["date"],
         "시장": "🇰🇷" if t["market"] == "KR" else "🇺🇸",
         "종목": f"{t['name']} ({t['code']})",
         "구분": "매수" if t["side"] == "buy" else "매도",
         "수량": t["qty"],
         "단가": t["price"],
         "환율": t["fx"] if t["market"] == "US" else None,
         "수수료": t["fee"],
         "실현손익(원)": realized_by_id.get(t["id"]),
         "메모": t["note"]}
        for t in ordered
    ])
    edited = st.data_editor(
        df_hist,
        use_container_width=True,
        hide_index=True,
        disabled=[c for c in df_hist.columns if c != "삭제"],
        column_config={
            "삭제": st.column_config.CheckboxColumn(width="small"),
            "수량": st.column_config.NumberColumn(format="%,.0f"),
            "단가": st.column_config.NumberColumn(format="%,.2f"),
            "환율": st.column_config.NumberColumn(format="%,.2f"),
            "실현손익(원)": st.column_config.NumberColumn(format="%,.0f"),
        },
        # 매매내역이 바뀌면 키도 바뀜 → 삭제 후 체크 상태가 다른 행에 남는 사고 방지
        key=f"hist_editor_{hash(tuple(t['id'] for t in ordered))}",
    )
    checked = [i for i, v in enumerate(edited["삭제"].tolist()) if v]
    if checked:
        if st.button(f"🗑️ 선택한 {len(checked)}건 삭제", type="secondary"):
            ids = {ordered[i]["id"] for i in checked}
            n = journal.delete_trades(ids)
            st.success(f"{n}건 삭제됨")
            st.rerun()
    st.caption(
        "잘못 입력한 기록은 왼쪽 체크박스로 선택해 삭제한 뒤 다시 입력하세요."
        + (" 여기 표시되는 수량·단가는 입력 당시 원본이며, 액면분할 환산은 보유 현황·수익률 계산에만 적용됩니다." if splits else "")
    )
else:
    st.caption("아직 기록이 없습니다.")

# ─────────── 진단 ───────────
with st.expander("🔍 진단: 저장 상태 / Gist 동기화 로그"):
    if cloud_store.is_configured():
        st.success("Gist 영구 저장소 연결됨 — 매매내역이 재배포 후에도 유지됩니다.")
    else:
        st.warning("Gist 미설정 — 로컬 파일에만 저장됩니다 (Streamlit Cloud 재배포 시 초기화).")
    for line in cloud_store.get_sync_log():
        st.text(line)
