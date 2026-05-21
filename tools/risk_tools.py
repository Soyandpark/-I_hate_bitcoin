"""
Risk Assessment Tools — CVaR, ATR, 변동성 기반 리스크 평가.
"""
from typing import Dict, Any, List, Optional
import numpy as np


def calculate_atr(
    high: List[float],
    low: List[float],
    close: List[float],
    period: int = 14,
) -> float:
    """Average True Range 계산"""
    if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
        return 0.0

    tr_list = []
    for i in range(1, len(high)):
        high_low = high[i] - low[i]
        high_close = abs(high[i] - close[i - 1])
        low_close = abs(low[i] - close[i - 1])
        tr_list.append(max(high_low, high_close, low_close))

    return float(np.mean(tr_list[-period:])) if tr_list else 0.0


def calculate_cvar(
    returns: List[float],
    confidence_level: float = 0.95,
) -> float:
    """
    Conditional Value at Risk (CVaR) 계산.
    returns: 직전 N期間の 수익률 리스트.
    """
    if not returns:
        return 0.0
    sorted_ret = sorted(returns)
    cutoff_idx = int(len(sorted_ret) * (1 - confidence_level))
    cvar = -np.mean(sorted_ret[:cutoff_idx]) if cutoff_idx > 0 else 0.0
    return float(cvar)


def check_volatility_break(
    current_atr: float,
    prev_atr: float,
    threshold_multiplier: float = 1.5,
) -> bool:
    """ATR 급증 감지 (변동성 돌파 조건 1)"""
    if prev_atr <= 0:
        return False
    return current_atr > prev_atr * threshold_multiplier


def check_signal_conflict(
    news_direction: int,      # 1=bullish, -1=bearish, 0=neutral
    cached_direction: int,     # 1=bullish, -1=bearish, 0=neutral
    threshold_conflict: bool = True,
) -> bool:
    """방향 충돌 감지 (조건 2)"""
    if news_direction == 0 or cached_direction == 0:
        return False
    return news_direction != cached_direction and threshold_conflict


def check_uncertainty_threshold(
    pred_upper: float,
    pred_lower: float,
    pred_mean: float,
    uncertainty_threshold: float = 0.05,
) -> bool:
    """
    예측 불확실성 임계치 초과 감지 (조건 3).
    pred_upper - pred_lower > uncertainty_threshold * pred_mean 이면 트리거.
    """
    if pred_mean <= 0:
        return False
    return (pred_upper - pred_lower) / pred_mean > uncertainty_threshold


def assess_overall_risk(
    cvar: float,
    atr_stop_pct: float,
    position_weight: float,
    cvar_max: float = 0.15,
) -> Dict[str, Any]:
    """
    종합 리스크 평가 결과 반환.
    """
    risk_exceeded = cvar > cvar_max or atr_stop_pct > 0.05
    return {
        "risk_level": "HIGH" if risk_exceeded else "NORMAL",
        "cvar_exceeded": cvar > cvar_max,
        "atr_stop_exceeded": atr_stop_pct > 0.05,
        "recommendation": "HOLD_or_SELL" if risk_exceeded else "PROCEED",
        "override_weight": position_weight * 0.5 if risk_exceeded else position_weight,
    }


def should_trigger_on_demand(
    market_data: Dict[str, Any],
    prev_episode_data: Optional[Dict] = None,
) -> tuple[bool, str]:
    """
    온디맨드 추론 트리거 조건 3가지를 검사하고 트리거 여부를 반환한다.

    Returns:
        (should_trigger, reason)
        reason: "volatility_break" | "signal_conflict" | "uncertainty" | "none"
    """
    preds = market_data.get("base_predictions", {})
    if not preds:
        return True, "initial_run"  # 최초 실행시는 항상 예측

    pred_upper = preds.get("pred_upper", 0)
    pred_lower = preds.get("pred_lower", 0)
    pred_mean = preds.get("pred_mean", 1)

    # 조건 3: 불확실성
    if check_uncertainty_threshold(pred_upper, pred_lower, pred_mean):
        return True, "uncertainty"

    # 조건 1: 변동성 돌파 (ATR 계산)
    if prev_episode_data:
        current_atr = market_data.get("atr", 0)
        prev_atr = prev_episode_data.get("atr", 0)
        if check_volatility_break(current_atr, prev_atr):
            return True, "volatility_break"

    # 조건 2: 시그널 충돌
    news_dir = market_data.get("news_direction", 0)
    cached_dir = preds.get("direction", 0)
    if check_signal_conflict(news_dir, cached_dir):
        return True, "signal_conflict"

    return False, "none"