# DEAR TICKET — AI Server

![CI](https://github.com/hyeonu8745/ticketing-ai/actions/workflows/ci.yml/badge.svg)

> 실시간 티켓팅 시스템을 위한 AI 마이크로서비스
> FastAPI 기반 봇 탐지 / 개인화 추천 / 수요 예측 / 챗봇

---

## Tech Stack

| 분류 | 기술 |
|------|------|
| Framework | FastAPI |
| ML | PyTorch, scikit-learn |
| 언어 | Python 3.11 |
| LLM | Ollama (Qwen3 8B) |
| CI/CD | GitHub Actions |

---

## 프로젝트 구조

```
ai/
├── bot_detection_server.py    # 봇 탐지 (port 8000)
├── recommendation_server.py   # 개인화 추천 (port 8001)
├── demand_forecast_server.py  # 수요 예측 (port 8002)
├── chatbot_server.py          # 고객센터 챗봇 (port 8003)
└── requirements.txt
```

---

## 서버별 기능

### 봇 탐지 (`bot_detection_server.py` — port 8000)
- 모델: GraphSAGE (PyTorch 2-layer)
- 8차원 피처 기반 실시간 봇 스코어 산출
- 임계값 0.7 이상 → 봇 판정 및 예매 차단
- Fail-Open: 서버 장애 시 메인 서비스 중단 없음

### 개인화 추천 (`recommendation_server.py` — port 8001)
- 모델: RALLRec (TF-IDF + 코사인 유사도)
- 예매 내역 기반 취향 벡터 생성 → 유사 공연 추천
- 카테고리 가중치(0.7 : 0.3) 적용으로 선호 장르 강화
- Cold-start 처리: 예매 이력 없는 신규 유저는 현재 공연 기반 추천
- Spring 서버 기동 시 `/index` 엔드포인트로 전체 공연 자동 인덱싱

### 수요 예측 (`demand_forecast_server.py` — port 8002)
- 모델: Lag-Llama (시계열 파운데이션 모델 경량 구현)
- lag 피처(1·2·4·8·24시간 전) 기반 향후 12시간 혼잡도 예측
- 혼잡도 레벨: LOW / MEDIUM / HIGH / VERY_HIGH
- 매진 예상 일시 및 AI 한줄 인사이트 반환

### 챗봇 (`chatbot_server.py` — port 8003)
- Ollama Qwen3 8B 기반 FAQ 자동 응답
- DEAR TICKET 서비스 전용 시스템 프롬프트 적용
- Fail-Open: Ollama 장애 시 fallback 메시지 반환

---

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
uvicorn bot_detection_server:app --host 0.0.0.0 --port 8000
uvicorn recommendation_server:app --host 0.0.0.0 --port 8001
uvicorn demand_forecast_server:app --host 0.0.0.0 --port 8002
uvicorn chatbot_server:app --host 0.0.0.0 --port 8003
```

> Spring Boot 백엔드 실행 **전에** AI 서버를 먼저 기동하는 것을 권장합니다.
> AI 서버가 꺼져 있어도 메인 서비스는 정상 동작합니다. (Fail-Open)

---

## API 엔드포인트

| 서버 | 엔드포인트 | 설명 |
|------|-----------|------|
| 봇 탐지 | `POST /detect` | 봇 스코어 및 차단 여부 반환 |
| 추천 | `POST /index` | 전체 공연 인덱싱 |
| 추천 | `POST /recommend` | 유저별 맞춤 공연 추천 |
| 수요 예측 | `POST /forecast` | 12시간 혼잡도 + 매진 예측 |
| 전체 | `GET /health` | 서버 상태 확인 |

---

## 관련 레포지토리

- 백엔드: [DEAR TICKET Backend](https://github.com/hyeonu8745/ticketing-server)
- 프론트엔드: [DEAR TICKET Frontend](https://github.com/hyeonu8745/ticketing-frontend)

---

## References

- Hamilton et al. (2017) — GraphSAGE. NeurIPS 2017. [arxiv](https://arxiv.org/abs/1706.02216)
- Yao et al. (2023) — RALLRec. [arxiv](https://arxiv.org/abs/2312.02445)
- Rasul et al. (2024) — Lag-Llama. [arxiv](https://arxiv.org/abs/2310.08278)