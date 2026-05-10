"""두 종목의 모든 공시 (90일) 조회"""
import dart

for code, name in [("007660", "이수페타시스"), ("003230", "삼양식품")]:
    print(f"\n=== {name} ({code}) — 90일 모든 공시 ===")
    discs = dart.get_recent_disclosures(code, days=90, max_count=50)
    print(f"개수: {len(discs)}")
    for d in discs:
        print(f"  {d['date']}  {d['title']}")
