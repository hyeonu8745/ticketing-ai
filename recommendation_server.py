"""
recommendation_server.py
=========================
VIVID HW — RALLRec (RAG + LLM) 기반 공연 추천 FastAPI 서버

[동작 원리]
1. 유저의 예매 내역(카테고리 + 공연 텍스트)을 임베딩
2. 전체 공연 임베딩과 코사인 유사도 계산 (RAG 방식)
3. 유저가 본 공연 제외 후 상위 N개 추천

[실행 방법]
uvicorn recommendation_server:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VIVID HW Recommendation API",
    description="RALLRec 기반 공연 추천 서버",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# 1. RALLRec 모델 핵심 컴포넌트
# ──────────────────────────────────────────────────────────────

class TextEmbedder:
    """
    경량 TF-IDF 기반 텍스트 임베더.
    실제 운영 시 sentence-transformers로 교체하면 품질 대폭 향상.
    (pip install sentence-transformers 후 교체 가능)
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.fitted = False

    def _tokenize(self, text: str) -> list[str]:
        """한국어 + 영어 단순 토크나이저 (공백/특수문자 기준 분리)"""
        import re
        text = text.lower()
        tokens = re.findall(r'[가-힣]+|[a-z]+|\d+', text)
        return tokens

    def fit(self, documents: list[str]):
        """전체 공연 텍스트로 어휘 사전 + IDF 구축"""
        # 어휘 사전 구축
        all_tokens = set()
        doc_tokens = []
        for doc in documents:
            tokens = set(self._tokenize(doc))
            doc_tokens.append(tokens)
            all_tokens.update(tokens)

        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        vocab_size = len(self.vocab)

        # IDF 계산
        idf = np.zeros(vocab_size)
        n_docs = len(documents)
        for tokens in doc_tokens:
            for token in tokens:
                if token in self.vocab:
                    idf[self.vocab[token]] += 1

        self.idf = np.log((n_docs + 1) / (idf + 1)) + 1
        self.fitted = True
        logger.info(f"✅ TextEmbedder 학습 완료: 어휘 수={vocab_size}, 문서 수={n_docs}")

    def embed(self, text: str) -> np.ndarray:
        """텍스트 → TF-IDF 벡터"""
        if not self.fitted or not self.vocab:
            # 미학습 상태: 랜덤 임베딩 반환 (cold start)
            return np.random.rand(128)

        tokens = self._tokenize(text)
        vec = np.zeros(len(self.vocab))
        token_counts = defaultdict(int)
        for token in tokens:
            token_counts[token] += 1

        for token, count in token_counts.items():
            if token in self.vocab:
                idx = self.vocab[token]
                tf = count / max(len(tokens), 1)
                vec[idx] = tf * self.idf[idx]

        # L2 정규화
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class RALLRecModel:
    """
    RAG + LLM 기반 추천 모델 (RALLRec 간소화 구현)

    [RALLRec 핵심 아이디어]
    - 공연 텍스트(제목+설명+카테고리)를 임베딩 → 벡터 DB 역할
    - 유저 행동(예매 내역)을 쿼리 벡터로 변환
    - 코사인 유사도로 가장 가까운 공연 검색 (RAG)
    - 카테고리 가중치로 개인화 강화 (LLM 역할 대체)
    """

    def __init__(self):
        self.embedder = TextEmbedder()
        self.event_embeddings: dict[int, np.ndarray] = {}   # eventId → 임베딩
        self.event_metadata: dict[int, dict] = {}           # eventId → 메타데이터
        self.category_weights: dict[str, float] = {}        # 카테고리 → 가중치
        self.is_ready = False

    def index_events(self, events: list[dict]):
        """
        전체 공연 목록을 임베딩하여 인덱싱.
        Spring 서버 시작 시 또는 주기적으로 호출.
        """
        if not events:
            logger.warning("인덱싱할 공연이 없습니다.")
            return

        # 공연 텍스트 구성 (제목 + 카테고리 + 설명 앞 200자)
        documents = []
        for e in events:
            text = f"{e.get('title', '')} {e.get('category', '')} {e.get('description', '')[:200]}"
            documents.append(text)

        # TF-IDF 학습
        self.embedder.fit(documents)

        # 각 공연 임베딩 저장
        for e, doc in zip(events, documents):
            event_id = e['id']
            self.event_embeddings[event_id] = self.embedder.embed(doc)
            self.event_metadata[event_id] = e

        self.is_ready = True
        logger.info(f"✅ RALLRec 인덱싱 완료: {len(events)}개 공연")

    def build_user_profile(
        self,
        reserved_event_ids: list[int],
        reserved_categories: list[str]
    ) -> np.ndarray:
        """
        유저 예매 내역 → 프로필 벡터 생성
        예매한 공연들의 임베딩 평균 = 유저 취향 벡터
        """
        vectors = []
        for eid in reserved_event_ids:
            if eid in self.event_embeddings:
                vectors.append(self.event_embeddings[eid])

        if not vectors:
            # Cold start: 카테고리 기반 텍스트로 프로필 생성
            category_text = " ".join(reserved_categories) * 3  # 반복으로 가중치 강화
            return self.embedder.embed(category_text)

        # 예매 내역 평균 벡터
        profile = np.mean(vectors, axis=0)

        # 카테고리 가중치 적용 (최근 예매한 카테고리 강화)
        category_boost = self.embedder.embed(" ".join(reserved_categories))
        if len(profile) == len(category_boost):
            profile = profile * 0.7 + category_boost * 0.3

        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def recommend(
        self,
        user_profile: np.ndarray,
        exclude_ids: list[int],
        top_k: int = 6
    ) -> list[dict]:
        """
        유저 프로필 벡터와 전체 공연 임베딩의 코사인 유사도 계산
        → 상위 top_k 공연 반환 (이미 예매한 공연 제외)
        """
        if not self.event_embeddings:
            return []

        exclude_set = set(exclude_ids)
        scores = []

        for event_id, embedding in self.event_embeddings.items():
            if event_id in exclude_set:
                continue
            if len(embedding) != len(user_profile):
                continue

            # 코사인 유사도
            similarity = float(np.dot(user_profile, embedding) / (
                np.linalg.norm(user_profile) * np.linalg.norm(embedding) + 1e-8
            ))
            scores.append((event_id, similarity))

        # 유사도 내림차순 정렬
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for event_id, score in scores[:top_k]:
            meta = self.event_metadata[event_id]
            results.append({
                "id": event_id,
                "title": meta.get("title", ""),
                "location": meta.get("location", ""),
                "posterUrl": meta.get("posterUrl", ""),
                "category": meta.get("category", ""),
                "priceRange": meta.get("priceRange", ""),
                "remainingSeats": meta.get("remainingSeats", 0),
                "totalSeats": meta.get("totalSeats", 0),
                "similarityScore": round(score, 4),
            })

        return results


# 전역 모델 인스턴스
rallrec = RALLRecModel()
logger.info("✅ RALLRec 추천 모델 초기화 완료")


# ──────────────────────────────────────────────────────────────
# 2. 요청/응답 스키마
# ──────────────────────────────────────────────────────────────

class EventInfo(BaseModel):
    """Spring에서 보내는 공연 정보"""
    id: int
    title: str
    category: str
    location: Optional[str] = ""
    posterUrl: Optional[str] = ""
    description: Optional[str] = ""
    priceRange: Optional[str] = ""
    remainingSeats: Optional[int] = 0
    totalSeats: Optional[int] = 0


class IndexRequest(BaseModel):
    """전체 공연 인덱싱 요청"""
    events: list[EventInfo]


class RecommendRequest(BaseModel):
    """추천 요청"""
    userId: int
    currentEventId: int                     # 현재 보고 있는 공연 ID
    reservedEventIds: list[int] = []        # 예매한 공연 ID 목록
    reservedCategories: list[str] = []      # 예매한 카테고리 목록
    topK: int = 6                           # 추천 개수


class RecommendedEvent(BaseModel):
    id: int
    title: str
    location: str
    posterUrl: str
    category: str
    priceRange: str
    remainingSeats: int
    totalSeats: int
    similarityScore: float


class RecommendResponse(BaseModel):
    userId: int
    recommendations: list[RecommendedEvent]
    reason: str


# ──────────────────────────────────────────────────────────────
# 3. API 엔드포인트
# ──────────────────────────────────────────────────────────────

@app.post("/index")
async def index_events(req: IndexRequest):
    """
    Spring 서버 시작 시 전체 공연 인덱싱.
    공연 데이터가 바뀔 때마다 호출.
    """
    events = [e.model_dump() for e in req.events]
    rallrec.index_events(events)
    return {"message": f"{len(events)}개 공연 인덱싱 완료", "status": "OK"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """
    유저 예매 내역 + 현재 공연 기반 추천.
    공연 상세 페이지 진입 시 Spring에서 호출.
    """
    try:
        # 현재 보고 있는 공연도 제외 목록에 추가
        exclude_ids = list(set(req.reservedEventIds + [req.currentEventId]))

        # 예매 내역이 없으면 현재 공연 기반으로 추천
        if not req.reservedEventIds and req.currentEventId in rallrec.event_embeddings:
            user_profile = rallrec.event_embeddings[req.currentEventId]
            reason = "현재 공연과 유사한 공연"
        else:
            user_profile = rallrec.build_user_profile(
                req.reservedEventIds,
                req.reservedCategories
            )
            reason = "예매 내역 + 선호 카테고리 기반 추천"

        recommendations = rallrec.recommend(user_profile, exclude_ids, req.topK)

        logger.info(
            f"✅ [RECOMMEND] userId={req.userId} "
            f"추천 {len(recommendations)}개 반환"
        )

        return RecommendResponse(
            userId=req.userId,
            recommendations=recommendations,
            reason=reason
        )

    except Exception as e:
        logger.error(f"추천 오류: {e}")
        return RecommendResponse(
            userId=req.userId,
            recommendations=[],
            reason="추천 서버 오류"
        )


@app.get("/health")
async def health_check():
    return {
        "status": "UP",
        "model": "RALLRec v1.0",
        "indexed_events": len(rallrec.event_embeddings),
        "is_ready": rallrec.is_ready
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("recommendation_server:app", host="0.0.0.0", port=8001, reload=True)
