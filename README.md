# 🤖 DEAR TICKET — AI Server

> 실시간 티켓팅 시스템을 위한 AI 마이크로서비스  
> FastAPI 기반 봇 탐지 / 개인화 추천 / 수요 예측 / 챗봇

---

## 🛠 Tech Stack

| 분류 | 기술 |
|------|------|
| Framework | FastAPI |
| ML | PyTorch, scikit-learn |
| 언어 | Python 3.11 |

---

## 📁 프로젝트 구조

```
ai/
├── bot_detection_server.py    # 봇 탐지 (port 8000)
├── recommendation_server.py   # 개인화 추천 (port 8001)
├── demand_forecast_server.py  # 수요 예측 (port 8002)
├── chatbot_server.py          # 고객센터 챗봇 (port 8003)
└── requirements.txt
```

---

## ✨ 서버별 기능

### 🛡 봇 탐지 (`bot_detection_server.py` — port 8000)
- **모델**: GraphSAGE (PyTorch 2-layer)
- 8차원 피처 기반 실시간 봇 스코어 산출
  - 60초 내 요청 수, 요청 간격 표준편차, 동시 접속 수
  - 대기열 통과 후 예매 소요 시간, 취소율
  - User-Agent 패턴, 마우스/터치 인터랙션, 동일 이벤트 반복 접근
- 임계값 0.7 이상 → 봇 판정 및 예매 차단
- **Fail-Open**: 서버 장애 시 메인 서비스 중단 없음

### 🎯 개인화 추천 (`recommendation_server.py` — port 8001)
- **모델**: RALLRec (TF-IDF + 코사인 유사도)
- 예매 내역 기반 취향 벡터 생성 → 유사 공연 추천
- 카테고리 가중치(0.7 : 0.3) 적용으로 선호 장르 강화
- Cold-start 처리: 예매 이력 없는 신규 유저는 현재 공연 기반 추천
- Spring 서버 기동 시 `/index` 엔드포인트로 전체 공연 자동 인덱싱

### 📈 수요 예측 (`demand_forecast_server.py` — port 8002)
- **모델**: Lag-Llama (시계열 파운데이션 모델 경량 구현)
- lag 피처(1·2·4·8·24시간 전) 기반 향후 12시간 혼잡도 예측
- 혼잡도 레벨: `LOW` / `MEDIUM` / `HIGH` / `VERY_HIGH`
- 매진 예상 일시 및 AI 한줄 인사이트 반환
- Cold-start: 예매 이력 없는 신규 공연은 시간대별 패턴 시뮬레이션

### 💬 챗봇 (`chatbot_server.py` — port 8003)
- 키워드 매칭 기반 질문 응답
- 마이페이지 / 검색 / 포인트 충전 등 바로가기 연동

---

## 🚀 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
python bot_detection_server.py    # port 8000
python recommendation_server.py   # port 8001
python demand_forecast_server.py  # port 8002
python chatbot_server.py          # port 8003
```

> Spring Boot 백엔드 실행 전에 AI 서버를 먼저 기동하는 것을 권장합니다.  
> AI 서버가 꺼져 있어도 메인 서비스는 정상 동작합니다. (Fail-Open)

---

## 📡 API 엔드포인트

| 서버 | 엔드포인트 | 설명 |
|------|-----------|------|
| 봇 탐지 | `POST /detect` | 봇 스코어 및 차단 여부 반환 |
| 추천 | `POST /index` | 전체 공연 인덱싱 |
| 추천 | `POST /recommend` | 유저별 맞춤 공연 추천 |
| 수요 예측 | `POST /forecast` | 12시간 혼잡도 + 매진 예측 |
| 전체 | `GET /health` | 서버 상태 확인 |

---

## 🔗 관련 레포지토리

- 백엔드: [DEAR TICKET Backend](링크)
- 프론트엔드: [DEAR TICKET Frontend](링크)

---

## 📦 References

- Hamilton et al. (2017) — GraphSAGE. NeurIPS 2017. [arxiv](https://arxiv.org/abs/1706.02216)
- Yao et al. (2023) — RALLRec. [arxiv](https://arxiv.org/abs/2312.02445)
- Rasul et al. (2024) — Lag-Llama. [arxiv](https://arxiv.org/abs/2310.08278)
