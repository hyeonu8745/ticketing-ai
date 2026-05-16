"""
demand_forecast_server.py
==========================
VIVID HW — Lag-Llama 기반 공연 수요 예측 FastAPI 서버

[동작 원리]
1. 공연의 잔여석 감소 이력 → 시계열 데이터로 구성
2. Lag-Llama (시계열 파운데이션 모델) 로 미래 수요 예측
3. 시간대별 혼잡도 + 매진 예측일 반환

[실행 방법]
uvicorn demand_forecast_server:app --host 0.0.0.0 --port 8002 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VIVID HW Demand Forecast API",
    description="Lag-Llama 기반 공연 수요 예측 서버",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# 1. Lag-Llama 핵심 컴포넌트
# ──────────────────────────────────────────────────────────────

class LagFeatureExtractor:
    """
    Lag-Llama의 핵심: 시계열을 lag(과거 시점) 피처로 변환.
    예) lag=[1,2,4,8,24] → 1시간 전, 2시간 전, 4시간 전... 예매 수를 피처로 사용
    """
    def __init__(self, lags: list[int] = None):
        self.lags = lags or [1, 2, 4, 8, 24, 48]

    def transform(self, series: list[float]) -> np.ndarray:
        """시계열 → lag 피처 행렬"""
        n = len(series)
        max_lag = max(self.lags)

        if n <= max_lag:
            # 데이터 부족 시 패딩
            series = [series[0]] * (max_lag - n + 1) + list(series)

        features = []
        for i in range(max_lag, len(series)):
            row = [series[i - lag] for lag in self.lags]
            features.append(row)

        return np.array(features) if features else np.zeros((1, len(self.lags)))


class LagLlamaForecaster:
    """
    Lag-Llama 간소화 구현.

    [실제 Lag-Llama 동작]
    - 트랜스포머 기반 시계열 파운데이션 모델
    - lag 피처를 컨텍스트로 사용해 확률적 예측 수행
    - Zero-shot: 새 공연 데이터도 별도 학습 없이 바로 예측 가능

    [여기서의 구현]
    - lag 피처 추출 + 선형 회귀 기반 트렌드 예측
    - 확률적 불확실성 구간 추가 (Lag-Llama의 분포 예측 모방)
    """

    def __init__(self):
        self.extractor = LagFeatureExtractor(lags=[1, 2, 4, 8, 24])

    def forecast(
        self,
        history: list[float],   # 시간별 누적 예매 수 이력
        horizon: int = 24       # 앞으로 몇 시간 예측할지
    ) -> dict:
        """
        시계열 이력 → 미래 수요 예측
        반환: 예측값 리스트 + 상/하한 구간
        """
        if len(history) < 2:
            # Cold start: 균등 분포로 예측
            base = history[0] if history else 10
            predictions = [base * (1 + 0.05 * i) for i in range(horizon)]
            return {
                "predictions": predictions,
                "upper": [p * 1.2 for p in predictions],
                "lower": [p * 0.8 for p in predictions],
            }

        series = np.array(history, dtype=float)

        # 1. 트렌드 추출 (선형 회귀)
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series, deg=1)
        trend_slope = coeffs[0]

        # 2. 계절성 패턴 (주말/공연 시간대 가중치)
        seasonality = self._extract_seasonality(series)

        # 3. Lag 피처 기반 단기 예측
        lag_features = self.extractor.transform(history)
        last_lag_mean = float(np.mean(lag_features[-1])) if len(lag_features) > 0 else series[-1]

        # 4. 미래 예측값 생성
        last_val = float(series[-1])
        predictions = []
        upper = []
        lower = []

        for h in range(1, horizon + 1):
            # 트렌드 + 계절성 + lag 가중 평균
            trend_component = last_val + trend_slope * h
            seasonal_factor = seasonality[h % len(seasonality)] if seasonality else 1.0
            lag_component = last_lag_mean + trend_slope * h

            pred = trend_component * 0.6 + lag_component * 0.4
            pred = pred * seasonal_factor
            pred = max(0, pred)  # 음수 방지

            # 불확실성 구간 (horizon이 길수록 넓어짐)
            uncertainty = pred * 0.1 * np.sqrt(h)
            predictions.append(round(pred, 2))
            upper.append(round(pred + uncertainty, 2))
            lower.append(round(max(0, pred - uncertainty), 2))

        return {
            "predictions": predictions,
            "upper": upper,
            "lower": lower,
            "trend_slope": round(float(trend_slope), 4),
        }

    def _extract_seasonality(self, series: np.ndarray) -> list[float]:
        """시간대별 계절성 패턴 추출 (24시간 주기)"""
        if len(series) < 24:
            return [1.0] * 24

        # 24시간 주기 평균
        n_complete = (len(series) // 24) * 24
        if n_complete == 0:
            return [1.0] * 24

        reshaped = series[:n_complete].reshape(-1, 24)
        hourly_mean = reshaped.mean(axis=0)
        overall_mean = hourly_mean.mean()

        if overall_mean == 0:
            return [1.0] * 24

        return (hourly_mean / overall_mean).tolist()


# 전역 인스턴스
forecaster = LagLlamaForecaster()

# 공연별 예매 이력 저장소 (실제 운영 시 Redis/DB 사용 권장)
reservation_history: dict[int, list[dict]] = defaultdict(list)

logger.info("✅ Lag-Llama 수요 예측 모델 초기화 완료")


# ──────────────────────────────────────────────────────────────
# 2. 요청/응답 스키마
# ──────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    eventId: int
    totalSeats: int
    remainingSeats: int
    startTime: str                          # ISO 형식: "2026-06-20T19:30:00"
    hourlyReservations: list[int] = []      # 시간대별 예매 수 이력 (없으면 시뮬레이션)


class HourlyDemand(BaseModel):
    hour: str           # "14:00"
    demand: int         # 예측 예매 수
    level: str          # "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH"
    label: str          # "여유" | "보통" | "혼잡" | "매우 혼잡"


class ForecastResponse(BaseModel):
    eventId: int
    soldOutPrediction: Optional[str]        # 매진 예측일시 (없으면 null)
    soldOutDaysLeft: Optional[int]          # 매진까지 남은 일수
    currentDemandLevel: str                 # 현재 혼잡도
    hourlyDemands: list[HourlyDemand]       # 시간대별 혼잡도 (다음 12시간)
    reservationRate: float                  # 현재 예매율 (%)
    insight: str                            # AI 한줄 인사이트


# ──────────────────────────────────────────────────────────────
# 3. 핵심 로직
# ──────────────────────────────────────────────────────────────

def simulate_history(total_seats: int, remaining_seats: int) -> list[float]:
    """
    예매 이력이 없을 때 현재 예매율 기반으로 시계열 시뮬레이션.
    실제 서비스에서는 DB에서 시간별 예매 로그를 가져와 대체.
    """
    sold = total_seats - remaining_seats
    if sold <= 0:
        return [0.0] * 24

    # 공연 특성상 오후~저녁에 예매가 몰리는 패턴
    hourly_weights = [
        0.5, 0.3, 0.2, 0.2, 0.3, 0.5,   # 00~05시 (새벽: 적음)
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5,   # 06~11시 (오전: 증가)
        4.0, 4.5, 4.0, 3.5, 3.0, 4.0,   # 12~17시 (점심~오후)
        5.0, 5.5, 4.5, 3.5, 2.0, 1.0,   # 18~23시 (저녁: 피크)
    ]
    total_weight = sum(hourly_weights)
    history = [sold * (w / total_weight) for w in hourly_weights]
    return history


def classify_demand(predicted: float, total_seats: int) -> tuple[str, str]:
    """예측 수요 → 혼잡도 레벨 분류"""
    rate = predicted / max(total_seats * 0.1, 1)  # 시간당 10% 기준
    if rate < 0.3:
        return "LOW", "여유"
    elif rate < 0.6:
        return "MEDIUM", "보통"
    elif rate < 0.9:
        return "HIGH", "혼잡"
    else:
        return "VERY_HIGH", "매우 혼잡"


def predict_sold_out(
    remaining: int,
    predictions: list[float],
    base_time: datetime
) -> tuple[Optional[str], Optional[int]]:
    """남은 좌석 + 예측 수요로 매진 시점 계산"""
    if remaining <= 0:
        return "이미 매진", 0

    cumulative = 0
    for h, pred in enumerate(predictions):
        cumulative += pred
        if cumulative >= remaining:
            sold_out_time = base_time + timedelta(hours=h + 1)
            days_left = (sold_out_time - base_time).days
            return sold_out_time.strftime("%Y년 %m월 %d일 %H시"), max(0, days_left)

    return None, None  # 예측 기간 내 매진 안 됨


def generate_insight(
    reservation_rate: float,
    demand_level: str,
    sold_out_days: Optional[int]
) -> str:
    """AI 한줄 인사이트 생성"""
    if reservation_rate >= 95:
        return "🔴 거의 매진! 지금 바로 예매하세요."
    if sold_out_days is not None and sold_out_days == 0:
        return "🔴 오늘 안에 매진될 것으로 예측됩니다!"
    if sold_out_days is not None and sold_out_days <= 3:
        return f"🟠 {sold_out_days}일 내 매진 예측 — 서두르세요!"
    if demand_level == "VERY_HIGH":
        return "🟠 현재 예매가 매우 몰리고 있어요. 빠른 예매를 권장합니다."
    if demand_level == "HIGH":
        return "🟡 예매 속도가 빨라지고 있어요. 관심 있으시면 서두르세요."
    if reservation_rate >= 50:
        return "🟢 절반 이상 예매됐어요. 원하는 좌석을 미리 선점하세요."
    return "🟢 아직 여유가 있어요. 천천히 좌석을 골라보세요."


# ──────────────────────────────────────────────────────────────
# 4. API 엔드포인트
# ──────────────────────────────────────────────────────────────

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_demand(req: ForecastRequest):
    """
    공연 수요 예측 메인 엔드포인트.
    공연 상세 페이지 진입 시 Spring에서 호출.
    """
    try:
        # 1. 예매 이력 구성 (없으면 시뮬레이션)
        if req.hourlyReservations:
            history = [float(x) for x in req.hourlyReservations]
        else:
            history = simulate_history(req.totalSeats, req.remainingSeats)

        # 2. Lag-Llama 예측 (다음 12시간)
        forecast_result = forecaster.forecast(history, horizon=12)
        predictions = forecast_result["predictions"]

        # 3. 현재 시각 기준 시간대별 혼잡도 구성
        now = datetime.now()
        hourly_demands = []
        for h, pred in enumerate(predictions):
            target_time = now + timedelta(hours=h)
            level, label = classify_demand(pred, req.totalSeats)
            hourly_demands.append(HourlyDemand(
                hour=target_time.strftime("%H:00"),
                demand=max(0, int(pred)),
                level=level,
                label=label,
            ))

        # 4. 매진 예측
        sold_out_time_str, sold_out_days = predict_sold_out(
            req.remainingSeats,
            predictions * 10,  # 12시간 → 120시간(5일)으로 확장해서 매진 예측
            now
        )

        # 5. 현재 예매율 & 혼잡도
        reservation_rate = round(
            (req.totalSeats - req.remainingSeats) / max(req.totalSeats, 1) * 100, 1
        )
        current_level, _ = classify_demand(
            float(np.mean(predictions[:3])), req.totalSeats
        )

        # 6. 인사이트
        insight = generate_insight(reservation_rate, current_level, sold_out_days)

        logger.info(
            f"✅ [FORECAST] eventId={req.eventId} "
            f"예매율={reservation_rate}% 매진예측={sold_out_time_str}"
        )

        return ForecastResponse(
            eventId=req.eventId,
            soldOutPrediction=sold_out_time_str,
            soldOutDaysLeft=sold_out_days,
            currentDemandLevel=current_level,
            hourlyDemands=hourly_demands,
            reservationRate=reservation_rate,
            insight=insight,
        )

    except Exception as e:
        logger.error(f"수요 예측 오류: {e}")
        return ForecastResponse(
            eventId=req.eventId,
            soldOutPrediction=None,
            soldOutDaysLeft=None,
            currentDemandLevel="LOW",
            hourlyDemands=[],
            reservationRate=0.0,
            insight="예측 데이터를 불러오는 중입니다.",
        )


@app.get("/health")
async def health_check():
    return {"status": "UP", "model": "Lag-Llama Demand Forecast v1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demand_forecast_server:app", host="0.0.0.0", port=8002, reload=True)
