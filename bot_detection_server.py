"""
bot_detection_server.py
========================
VIVID HW — GraphSAGE 기반 봇 탐지 FastAPI 서버

[동작 원리]
1. Spring 백엔드가 예매 요청 시 /detect 엔드포인트를 호출
2. 사용자 행동 피처(요청 속도, IP, 디바이스 등)를 그래프 노드로 표현
3. Inductive GraphSAGE로 임베딩 → 봇 확률 스코어 반환
4. Spring은 스코어가 임계값(0.7) 초과면 예매 차단

[파이참에서 실행 방법]
1. 터미널에서: pip install -r requirements.txt
2. 실행: uvicorn bot_detection_server:app --host 0.0.0.0 --port 8000 --reload
3. Swagger 확인: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VIVID HW Bot Detection API",
    description="GraphSAGE 기반 실시간 봇 탐지 서버",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# 1. GraphSAGE 모델 정의
# ──────────────────────────────────────────────────────────────

class GraphSAGELayer(nn.Module):
    """
    Inductive GraphSAGE Layer.
    핵심: 학습 때 보지 못한 새 노드(신규 IP, 신규 디바이스)도
    이웃 집계(mean aggregation)로 즉시 임베딩 생성 가능.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # 자기 자신(in_dim) + 이웃 평균(in_dim) → concat → Linear
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, neighbor_mean: torch.Tensor) -> torch.Tensor:
        aggregated = torch.cat([x, neighbor_mean], dim=-1)
        out = self.linear(aggregated)
        # 배치 크기가 1인 경우 BatchNorm 스킵
        if out.size(0) > 1:
            out = self.bn(out)
        return F.relu(out)


class BotDetectionGNN(nn.Module):
    """
    2-layer GraphSAGE → Binary Classifier (정상 유저 vs 봇)

    입력 피처 8차원:
      [0] 60초 내 요청 수           — 많을수록 봇 의심
      [1] 요청 간격 표준편차         — 낮을수록 봇 의심 (너무 규칙적)
      [2] 동일 IP 동시 접속 수       — 많을수록 봇 의심
      [3] 대기열 → 예매까지 소요 시간 — 너무 짧으면 봇 의심
      [4] 이전 예매 취소율           — 높을수록 봇 의심
      [5] User-Agent 다양성 점수     — 낮을수록 봇 의심
      [6] 마우스/터치 이벤트 여부     — 없으면(0) 봇 의심
      [7] 동일 이벤트 반복 접근 횟수  — 많을수록 봇 의심
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.sage1 = GraphSAGELayer(input_dim, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, neighbor_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 이웃 피처가 없으면 자기 자신을 이웃으로 사용 (단독 노드 처리)
        if neighbor_feats is None:
            neighbor_feats = x
        h1 = self.sage1(x, neighbor_feats)
        h2 = self.sage2(h1, h1)
        return self.classifier(h2)


# 모델 초기화
# 실제 서비스: model.load_state_dict(torch.load("bot_detection_weights.pth"))
model = BotDetectionGNN(input_dim=8, hidden_dim=64)
model.eval()
logger.info("✅ GraphSAGE 봇 탐지 모델 초기화 완료")


# ──────────────────────────────────────────────────────────────
# 2. 행동 이력 추적 (In-Memory 슬라이딩 윈도우)
#    실제 운영 환경에서는 Redis로 교체 권장
# ──────────────────────────────────────────────────────────────

# IP별 최근 요청 타임스탬프 (최근 200개까지만 보관)
ip_request_log: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))

# IP별 누적 취소 / 성공 횟수
ip_cancel_count:  dict[str, int] = defaultdict(int)
ip_success_count: dict[str, int] = defaultdict(int)

# IP별 접근한 이벤트 ID 집합 (다양성 측정용)
ip_event_access:  dict[str, set] = defaultdict(set)

# userId별 동일 이벤트 접근 횟수
user_event_count: dict[str, int] = defaultdict(int)


# ──────────────────────────────────────────────────────────────
# 3. 요청/응답 스키마
# ──────────────────────────────────────────────────────────────

class BotDetectionRequest(BaseModel):
    user_id: int
    event_id: int
    ip_address: str
    user_agent: str
    queue_wait_seconds: float       # 대기열 진입 후 예매 시도까지 걸린 시간(초)
    has_interaction: bool = True    # 마우스/터치 이벤트 존재 여부
    cancel_count: int = 0           # 해당 유저의 이전 취소 횟수
    success_count: int = 0          # 해당 유저의 이전 성공 횟수


class BotDetectionResponse(BaseModel):
    user_id: int
    is_bot: bool
    bot_score: float                # 0.0(정상) ~ 1.0(봇)
    reason: str
    blocked: bool


# ──────────────────────────────────────────────────────────────
# 4. 피처 추출
# ──────────────────────────────────────────────────────────────

def extract_features(req: BotDetectionRequest) -> torch.Tensor:
    now = time.time()
    ip = req.ip_address
    log = ip_request_log[ip]
    log.append(now)

    # 이벤트 접근 기록
    ip_event_access[ip].add(req.event_id)
    user_event_key = f"{req.user_id}:{req.event_id}"
    user_event_count[user_event_key] += 1

    # 피처 0: 60초 내 요청 수 정규화 (30회 = 1.0)
    recent_requests = sum(1 for t in log if now - t < 60)
    f0 = min(recent_requests / 30.0, 1.0)

    # 피처 1: 요청 간격 표준편차 역수 (규칙적일수록 높음 = 봇 의심)
    if len(log) >= 3:
        intervals = [log[i] - log[i-1] for i in range(1, len(log))]
        std = float(np.std(intervals))
        f1 = 1.0 - min(std / 5.0, 1.0)  # std가 0에 가까울수록 1.0
    else:
        f1 = 0.0

    # 피처 2: 동일 IP 동시 접속 수 — 현재 단순화하여 최근 5초 요청 수 사용
    concurrent = sum(1 for t in log if now - t < 5)
    f2 = min(concurrent / 10.0, 1.0)

    # 피처 3: 대기열 → 예매 소요 시간 역수 (너무 짧으면 봇)
    # 정상 유저: 10~60초, 봇: 1초 이하
    f3 = 1.0 - min(req.queue_wait_seconds / 10.0, 1.0)
    f3 = max(f3, 0.0)

    # 피처 4: 취소율 (취소 / (취소 + 성공 + 1))
    total = req.cancel_count + req.success_count + 1
    f4 = min(req.cancel_count / total, 1.0)

    # 피처 5: User-Agent 다양성 점수 (단순화: 알려진 봇 패턴 체크)
    ua_lower = req.user_agent.lower()
    bot_keywords = ["bot", "crawler", "spider", "curl", "python-requests", "java/", "go-http"]
    f5 = 1.0 if any(kw in ua_lower for kw in bot_keywords) else 0.0

    # 피처 6: 마우스/터치 이벤트 없으면 봇 의심
    f6 = 0.0 if req.has_interaction else 1.0

    # 피처 7: 동일 이벤트 반복 접근 횟수 (10회 이상 = 1.0)
    f7 = min(user_event_count[user_event_key] / 10.0, 1.0)

    features = [f0, f1, f2, f3, f4, f5, f6, f7]
    return torch.tensor([features], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# 5. 봇 여부 판단 이유 생성
# ──────────────────────────────────────────────────────────────

def get_reason(features: list[float], score: float) -> str:
    reasons = []
    if features[0] > 0.6:
        reasons.append("단시간 대량 요청")
    if features[1] > 0.8:
        reasons.append("요청 간격이 비정상적으로 규칙적")
    if features[3] > 0.9:
        reasons.append("대기열 통과 후 즉시 예매 시도")
    if features[5] > 0.5:
        reasons.append("봇 User-Agent 감지")
    if features[6] > 0.5:
        reasons.append("마우스/터치 이벤트 없음")
    if features[7] > 0.5:
        reasons.append("동일 이벤트 반복 접근")

    if not reasons:
        return "정상 요청으로 판단됨"
    return ", ".join(reasons)


# ──────────────────────────────────────────────────────────────
# 6. API 엔드포인트
# ──────────────────────────────────────────────────────────────

BOT_THRESHOLD = 0.7  # 봇 판단 임계값 (0.0 ~ 1.0)


@app.post("/detect", response_model=BotDetectionResponse)
async def detect_bot(req: BotDetectionRequest):
    """
    Spring 백엔드에서 예매 요청 시 호출.
    봇 확률 스코어와 차단 여부를 반환합니다.
    """
    try:
        feature_tensor = extract_features(req)
        features_list = feature_tensor[0].tolist()

        with torch.no_grad():
            bot_score = float(model(feature_tensor).item())

        is_bot = bot_score >= BOT_THRESHOLD
        reason = get_reason(features_list, bot_score)

        if is_bot:
            logger.warning(
                f"🚨 [BOT DETECTED] userId={req.user_id} ip={req.ip_address} "
                f"score={bot_score:.3f} reason={reason}"
            )
        else:
            logger.info(
                f"✅ [NORMAL] userId={req.user_id} score={bot_score:.3f}"
            )

        return BotDetectionResponse(
            user_id=req.user_id,
            is_bot=is_bot,
            bot_score=round(bot_score, 4),
            reason=reason,
            blocked=is_bot
        )

    except Exception as e:
        logger.error(f"봇 탐지 오류: {e}")
        # 탐지 서버 장애 시 → 예매 허용 (fail-open 전략)
        return BotDetectionResponse(
            user_id=req.user_id,
            is_bot=False,
            bot_score=0.0,
            reason="탐지 서버 오류 — fail-open 처리",
            blocked=False
        )


@app.get("/health")
async def health_check():
    """Spring Actuator처럼 서버 상태 확인용"""
    return {"status": "UP", "model": "GraphSAGE BotDetection v1.0"}


@app.delete("/reset/{ip_address}")
async def reset_ip_history(ip_address: str):
    """관리자용: 특정 IP 이력 초기화"""
    ip_request_log.pop(ip_address, None)
    ip_cancel_count.pop(ip_address, None)
    ip_success_count.pop(ip_address, None)
    return {"message": f"{ip_address} 이력 초기화 완료"}


@app.get("/stats/{ip_address}")
async def get_ip_stats(ip_address: str):
    """관리자용: IP별 행동 통계 조회"""
    log = ip_request_log.get(ip_address, deque())
    now = time.time()
    return {
        "ip": ip_address,
        "requests_last_60s": sum(1 for t in log if now - t < 60),
        "total_logged": len(log),
        "cancel_count": ip_cancel_count.get(ip_address, 0),
        "success_count": ip_success_count.get(ip_address, 0),
        "event_diversity": len(ip_event_access.get(ip_address, set())),
    }
