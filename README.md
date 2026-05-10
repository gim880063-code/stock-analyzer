# 📈 주식 분석 리포트

KOSPI 종목을 자동 분석해서 추세·모멘텀·거래량·가격 리스크·수급·공시·가치·재무 건전성·성장성 9개 항목으로 점수화하는 개인용 분석 도구.

## 주요 기능

- 🔍 **종목 발굴** — 안전 유니버스(시총 5조+ / 거래대금 500억+ / 관리종목 제외)에서 종합점수 높은 종목 자동 검색
- 📋 **DART 공시 자동 분류** — Gemini로 critical/negative/positive/neutral 분류 + 본문 깊이 분석
- 📢 **잠정실적공시 반영** — 정식 분기보고서 등록 전이라도 잠정실적으로 성장성 점수 갱신
- 📊 **120일 차트** — 종가 + 20일선 + 60일선
- ⭐ **즐겨찾기** — 종목 클릭 한 번에 단독 분석
- 💾 **영구 캐시** — `rcept_no` 기반으로 같은 공시는 한 번만 LLM 호출

## 데이터 출처

- 가격·거래량: [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader) (KRX 일봉)
- 재무·공시: [DART OPEN API](https://opendart.fss.or.kr) (무료)
- 외국인·기관 수급, 종목뉴스: 네이버 금융 (HTML 스크래핑)
- LLM 분석: Google Gemini API (무료 티어)

## 로컬 실행

```powershell
pip install -r requirements.txt
# .env 파일 생성 후 API 키 입력
python -m streamlit run app.py
```

`.env` 예시:
```
DART_API_KEY=발급받은_DART_키
GEMINI_API_KEY=발급받은_Gemini_키
```

## Streamlit Cloud 배포

1. 이 repo를 본인 GitHub 계정으로 fork/push
2. https://share.streamlit.io 에서 "New app" → repo 선택
3. App settings → Secrets에 `DART_API_KEY`, `GEMINI_API_KEY` 입력 (TOML 형식)

## 면책

이 앱은 **매수·매도 추천이 아닙니다**. 여러 지표를 정리해서 보여주는 분석 보조 도구이며, 최종 투자 판단은 본인의 책임입니다.
