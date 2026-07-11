"""
보유 종목 청산 점검 — 매수(스크리닝)와 대칭으로, '나갈 때'를 놓치지 않게 한다.

매일(자동 스크리닝과 함께) 보유 종목을 재평가해서 아래 조건이 충족되면 알림:
  1. 손절 조건 — 현재가가 손절선 이하 (저장된 손절가 없으면 평단 -8% 기본)
  2. 목표 도달 — 현재가가 1R / 2R 목표 이상
  3. 트레일링 스톱 — 이익 구간에서 고점 대비 N% 이상 하락
  4. 근거 붕괴 — 종합 신호가 '위험'으로 전환 / 종합점수 급락

앱 철학에 맞춰 '팔아라'가 아니라 '이 조건이 충족됐으니 점검하세요' 정보형 알림.
"""
from __future__ import annotations

from datetime import datetime

import cloud_store
import portfolio as port


# 손절선이 저장 안 된 보유 종목의 기본 손절 (평단 대비 %)
DEFAULT_STOP_PCT = 8.0

# 시간 손절 점검 — 이 일수 이상 보유했는데 손익이 없으면(현재가 ≤ 평단) 자본 회전 알림.
# 90일 근거: 발굴 검증의 최장 호라이즌이 60영업일(≈3개월)이라, 그 기간을 지나도
# 성과가 없으면 발굴 논리가 틀렸을 가능성이 크고 자본만 묶인다. 손절선과 달리
# '가격이 안 빠졌지만 안 가는' 죽은 돈을 잡는 보완 장치 (정보형 알림).
TIME_STOP_DAYS = 90

# 알림 심각도 표시
LEVEL_ICON = {"high": "🔴", "medium": "🟡", "info": "🔵"}

ALERTS_FILE = "holdings_alerts.json"


def evaluate_holding(
    code: str,
    name: str,
    holding: dict,
    analysis: dict,
    trail_pct: float = 10.0,
) -> dict:
    """단일 보유 종목 점검. analysis 는 analyzer.analyze 결과(dict).

    반환: code, name, current, score, action, pnl_pct, alerts[], new_peak
    """
    current = float(analysis.get("last_close") or 0)
    avg = float(holding.get("avg_price") or 0)
    plan = analysis.get("trade_plan") or {}
    total = analysis.get("total")
    action = str(plan.get("action") or "")
    alerts: list[tuple[str, str]] = []

    # 1) 손절 조건 — 저장된 손절선 우선, 없으면 평단 -DEFAULT_STOP_PCT%
    stop_ref = holding.get("stop_loss")
    stop_src = "설정 손절선"
    if stop_ref is None and avg > 0:
        stop_ref = avg * (1 - DEFAULT_STOP_PCT / 100.0)
        stop_src = f"기본 손절선(평단 -{DEFAULT_STOP_PCT:.0f}%)"
    if stop_ref and current and current <= float(stop_ref):
        alerts.append((
            "high",
            f"손절 조건 도달 — 현재가 {current:,.0f}원 ≤ {stop_src} {float(stop_ref):,.0f}원",
        ))

    # 2) 목표 도달 (2R 우선 표기)
    t1 = holding.get("target_1r")
    t2 = holding.get("target_2r")
    if t2 and current and current >= float(t2):
        alerts.append((
            "info",
            f"2R 목표 도달 — 현재가 {current:,.0f}원 ≥ {float(t2):,.0f}원 (이익 실현 점검)",
        ))
    elif t1 and current and current >= float(t1):
        alerts.append((
            "info",
            f"1R 목표 도달 — 현재가 {current:,.0f}원 ≥ {float(t1):,.0f}원 (일부 실현 점검)",
        ))

    # 3) 트레일링 스톱 — 이익 구간에서만 (평단 위)
    prev_peak = float(holding.get("peak_price") or avg or 0)
    new_peak = max(prev_peak, current) if current else prev_peak
    if current and avg and current > avg and new_peak > 0:
        drop = (new_peak - current) / new_peak * 100
        if drop >= trail_pct:
            alerts.append((
                "medium",
                f"고점 대비 -{drop:.1f}% (고점 {new_peak:,.0f}→현재 {current:,.0f}원) — 이익 보호 점검",
            ))

    # 4) 근거 붕괴 — 신호 전환 / 점수 급락
    if action.startswith("위험"):
        alerts.append(("medium", f"근거 약화 — 현재 신호 '{action}' (재평가 점검)"))
    elif isinstance(total, (int, float)) and total <= -1:
        alerts.append(("medium", f"근거 약화 — 종합점수 {total} (재평가 점검)"))

    # 5) 시간 손절 — 오래 들고 있는데 수익이 없는 '죽은 돈' 점검 (자본 회전)
    added_at = holding.get("added_at")
    if added_at and current and avg and current <= avg:
        try:
            held_days = (datetime.now() - datetime.strptime(str(added_at), "%Y-%m-%d")).days
        except ValueError:
            held_days = None
        if held_days is not None and held_days >= TIME_STOP_DAYS:
            pct = (current / avg - 1) * 100
            alerts.append((
                "medium",
                f"시간 손절 점검 — 보유 {held_days}일째 수익 없음({pct:+.1f}%). "
                f"자본을 더 좋은 후보로 회전할지 검토",
            ))

    pnl = port.compute_pnl(holding, current) if current else {}
    return {
        "code": code,
        "name": name,
        "current": current,
        "score": total,
        "action": action,
        "pnl_pct": round(pnl.get("profit_pct", 0), 2) if pnl else None,
        "alerts": [{"level": lv, "msg": m} for lv, m in alerts],
        "new_peak": new_peak,
    }


def monitor_holdings(analyze_fn=None, persist: bool = True) -> list[dict]:
    """모든 보유 종목 점검. analyze_fn(code)->dict 미지정 시 analyzer.analyze(lite).

    트레일링용 고점(peak_price)은 갱신되면 1회만 저장.
    """
    if analyze_fn is None:
        import analyzer
        def analyze_fn(c):  # noqa: E306
            return analyzer.analyze(c, lite=True, deep_top=0)

    settings = port.load_settings()
    trail_pct = float(settings.get("trail_pct", 10.0) or 10.0)
    p = port.load_portfolio()

    results: list[dict] = []
    changed = False
    for code, holding in p.items():
        try:
            analysis = analyze_fn(code)
        except Exception as e:
            results.append({"code": code, "name": code, "error": str(e), "alerts": []})
            continue
        if analysis.get("error"):
            results.append({
                "code": code, "name": analysis.get("name", code),
                "error": analysis["error"], "alerts": [],
            })
            continue

        ev = evaluate_holding(
            code, analysis.get("name", code), holding, analysis, trail_pct=trail_pct,
        )
        new_peak = ev.pop("new_peak", None)
        if new_peak and new_peak != float(holding.get("peak_price") or 0):
            holding["peak_price"] = new_peak
            changed = True
        results.append(ev)

    if persist and changed:
        port.save_portfolio(p)
    return results


def save_alerts_snapshot(results: list[dict]) -> None:
    """최근 점검 결과를 저장 — 헤드리스(cron)에서 점검한 걸 앱이 보여줄 수 있게."""
    payload = {
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "holdings": results,
    }
    try:
        cloud_store.save(ALERTS_FILE, payload)
    except Exception:
        pass


def load_alerts_snapshot() -> dict:
    data = cloud_store.load(ALERTS_FILE, {})
    return data if isinstance(data, dict) else {}


def count_actionable(results: list[dict]) -> int:
    """high/medium 알림이 하나라도 있는 종목 수 (info=목표도달은 제외)."""
    n = 0
    for r in results:
        if any(a.get("level") in ("high", "medium") for a in r.get("alerts", [])):
            n += 1
    return n
