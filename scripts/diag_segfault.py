"""배포 Segmentation fault 진단 — 서버와 같은 리눅스에서 의심 경로를 단계별 실행.

GitHub Actions(diag-segfault.yml)가 각 단계를 별도 프로세스로 호출한다.
segfault 가 나면 faulthandler 가 죽은 위치의 파이썬 스택을 stderr 에 남기고,
그 단계의 exit code 가 139 로 기록되어 범인 경로가 특정된다.
"""
import faulthandler
faulthandler.enable()

import json
import sys
import traceback
from pathlib import Path

# 윈도우 콘솔(cp949) 등 어디서 돌려도 한글·특수문자 출력이 죽지 않게
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(msg, flush=True)


def seed_data() -> None:
    """Gist 없이도 페이지 계산 경로가 전부 돌도록 가짜 매매기록을 로컬에 심는다."""
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    trades = [
        {"id": "t1", "date": "2026-05-04", "market": "KR", "code": "005930",
         "name": "삼성전자", "side": "buy", "qty": 10, "price": 61000.0,
         "fx": 1.0, "fee": 150.0, "note": ""},
        {"id": "t2", "date": "2026-05-15", "market": "US", "code": "AAPL",
         "name": "AAPL", "side": "buy", "qty": 3, "price": 210.5,
         "fx": 1385.2, "fee": 0.5, "note": ""},
        {"id": "t3", "date": "2026-06-10", "market": "KR", "code": "005930",
         "name": "삼성전자", "side": "sell", "qty": 4, "price": 64500.0,
         "fx": 1.0, "fee": 210.0, "note": ""},
        {"id": "t4", "date": "2026-06-22", "market": "US", "code": "NVDA",
         "name": "NVDA", "side": "buy", "qty": 2, "price": 155.0,
         "fx": 1390.0, "fee": 0.3, "note": ""},
    ]
    (data / "trades.json").write_text(
        json.dumps(trades, ensure_ascii=False), encoding="utf-8")


def stage_imports() -> None:
    import analyzer, journal, quotes, cloud_store, history, portfolio  # noqa
    import screening_history, scouted, holdings_monitor, naver, dart, llm, verifier  # noqa
    log("모든 모듈 임포트 OK")


def stage_krx() -> None:
    from analyzer import all_korean_stocks
    d = all_korean_stocks()
    log(f"KRX 종목 {len(d)}개")


def stage_etf() -> None:
    import FinanceDataReader as fdr
    etf = fdr.StockListing("ETF/KR")
    log(f"ETF {len(etf)}개, cols={list(etf.columns)[:6]}")
    sym = "Symbol" if "Symbol" in etf.columns else etf.columns[0]
    nm = "Name" if "Name" in etf.columns else etf.columns[1]
    names = {}
    for _, r in etf.iterrows():
        names[str(r[sym]).strip()] = str(r[nm]).strip()
    log(f"병합 루프 {len(names)}개 OK")


def stage_fx() -> None:
    from datetime import date
    import quotes
    s = quotes.usdkrw_history(date(2026, 4, 1))
    log(f"환율 이력 {0 if s is None else len(s)}건, 현재 {quotes.current_usdkrw()}")


def stage_price() -> None:
    from datetime import date
    import quotes
    for mkt, code in (("KR", "005930"), ("US", "AAPL")):
        s = quotes.price_history(mkt, code, date(2026, 4, 1))
        log(f"{code}: {0 if s is None else len(s)}건")


def stage_journal() -> None:
    seed_data()
    from datetime import datetime, timedelta
    import journal
    import quotes
    trades = journal.load_trades()
    splits = journal.load_splits()
    adj = journal.adjust_trades_for_splits(trades, splits)
    pos, realized, warns = journal.compute_positions(adj)
    log(f"positions={len(pos)} realized={len(realized)} warns={len(warns)}")
    first = datetime.strptime(
        journal.sorted_trades(trades)[0]["date"], "%Y-%m-%d").date()
    fx = quotes.usdkrw_history(first - timedelta(days=10))

    def pfn(m, c):
        return quotes.price_history(m, c, first - timedelta(days=10))

    monthly = journal.compute_monthly_returns(adj, pfn, fx)
    log(f"monthly={len(monthly)} OK")


def stage_page() -> None:
    """매매기록 페이지 전체를 실제 Streamlit 렌더 경로(Arrow 직렬화 포함)로 실행."""
    seed_data()
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(
        str(ROOT / "pages" / "1_매매기록_수익률.py"), default_timeout=600)
    at.run()
    log(f"페이지 렌더 OK — 예외 요소 {len(at.exception)}개")
    for e in at.exception:
        log(f"  페이지 내부 예외: {e.value}")


def stage_main() -> None:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    log(f"메인 렌더 OK — 예외 요소 {len(at.exception)}개")
    for e in at.exception:
        log(f"  메인 내부 예외: {e.value}")


def stage_parallel() -> None:
    """스크리닝과 동일한 3병렬 분석 — 스레드 동시성(lxml 등) segfault 점검."""
    from concurrent.futures import ThreadPoolExecutor
    from analyzer import analyze
    codes = ["005930", "000660", "035420"]
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(lambda c: analyze(c, lite=True), codes):
            log(f"analyze {getattr(r, 'code', '?')} OK")


STAGES = {
    "imports": stage_imports,
    "krx": stage_krx,
    "etf": stage_etf,
    "fx": stage_fx,
    "price": stage_price,
    "journal": stage_journal,
    "page": stage_page,
    "main": stage_main,
    "parallel": stage_parallel,
}

if __name__ == "__main__":
    name = sys.argv[1]
    try:
        STAGES[name]()
        log(f"[{name}] 정상 종료")
    except Exception:
        traceback.print_exc()
        log(f"[{name}] 파이썬 예외로 종료 (segfault 아님)")
