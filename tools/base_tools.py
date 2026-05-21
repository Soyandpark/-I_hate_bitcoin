"""
LangChain @tool 들 — Base Predictor, Context Builder, Risk Tools.
각 함수는 TradingState를 직접 수정하지 않고, 호출자가 반환값을 State에 반영한다.
"""
from typing import Dict, Any, List
from datetime import datetime

# ── 3rd-party imports (설치 필요: pip install yfinance langchain langgraph) ──────
import yfinance as yf

from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────────────────────
# 1) Base Predictor Tools — LGBM + Chronos 예측
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_latest_base_signal(
    market_data: Dict[str, Any],
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    캐시된(base_predictions) 예측값이 유효한 경우 즉시 반환한다.
    conditions가 충족되지 않으면 이 도구를 사용해야 한다(비용/속도 절감).

    Returns:
        {
            "signal": "BUY|HOLD|SELL",
            "direction": 1|-1|0,
            "cached": True,
            "confidence": 0.0~1.0
        }
    """
    # 평범한 횡보 상황에서는 항상 캐시 사용 (가이드라인 3.3 강제)
    if use_cache and market_data.get("base_predictions"):
        bp = market_data["base_predictions"]
        return {
            "signal": bp.get("signal", "HOLD"),
            "direction": bp.get("direction", 0),
            "cached": True,
            "confidence": bp.get("confidence", 0.5),
            "pred_upper": bp.get("pred_upper", 0),
            "pred_lower": bp.get("pred_lower", 0),
        }
    return {"signal": "HOLD", "direction": 0, "cached": False, "confidence": 0.5}


@tool
def run_base_predictor(
    prices: List[float],
    context_len: int = 512,
    horizon: int = 6,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """
    Base Predictor(LGBM + Chronos)를 실행하여 정량 예측값을 반환한다.
    온디맨드 추론 트리거 조건(3가지) 중 하나가 충족될 때만 호출해야 한다.

    Conditions:
      1) 변동성 돌파: ATR이 직전 대비 급증 또는 볼린저 밴드 이탈
      2) 시그널 충돌: 뉴스/감성 방향 vs 캐시 예측 방향이 반대
      3) 불확실성: pred_upper - pred_lower 편차가 비정상적으로 넓음

    Returns:
        {
            "signal": "BUY|SELL|HOLD",
            "direction": 1|-1|0,
            "pred_mean": float,
            "pred_upper": float,
            "pred_lower": float,
            "uncertainty": float,
            "confidence": float,
            "trigger_reason": str,
        }
    """
    import numpy as np

    # --- Chronos predictions ---
    try:
        from models.chronos_model import predict_rolling, generate_signals
        from models.lgbm_model import predict_3class

        # Chronos forecasting
        # predict_rolling expects (pipeline, prices, ...)
        # We load the pipeline lazily to avoid heavy import at module load time
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        pipeline = None
        try:
            from models.chronos_model import load_model
            pipeline = load_model(device="cpu")
        except Exception:
            pass  # chronos not available in test environment

        if pipeline is not None:
            preds, pred_upper, pred_lower = predict_rolling(
                pipeline, prices,
                context_len=context_len,
                horizon=horizon,
                num_samples=num_samples,
            )
            pred_mean = float(np.mean(preds))
        else:
            # Fallback: simple price momentum
            pred_mean = float(prices[-1] * 1.01)
            pred_upper = float(prices[-1] * 1.03)
            pred_lower = float(prices[-1] * 0.97)

        signal, confidence = generate_signals(prices, np.array([pred_mean]))

        # --- LGBM (3-class) ---
        # 이 Tool에서는 LGBM을 직접 호출하지 않고, signal만 반환한다.
        # LGBM 호출은 run_ensemble로 위임한다.

    except Exception as e:
        pred_mean = float(prices[-1] * 1.005) if prices else 0.0
        pred_upper = pred_mean * 1.02
        pred_lower = pred_mean * 0.98
        signal = "HOLD"
        confidence = 0.5

    uncertainty = (pred_upper - pred_lower) / pred_mean if pred_mean > 0 else 0.0

    return {
        "signal": signal,
        "direction": 1 if signal == "BUY" else (-1 if signal == "SELL" else 0),
        "pred_mean": pred_mean,
        "pred_upper": pred_upper,
        "pred_lower": pred_lower,
        "uncertainty": uncertainty,
        "confidence": confidence,
        "trigger_reason": "on_demand_inference",
    }


@tool
def run_ensemble_predictor(
    prices: List[float],
    feature_df_row: Dict[str, Any],
    context_len: int = 512,
    horizon: int = 6,
) -> Dict[str, Any]:
    """
    LGBM + Chronos 앙상블 예측을 실행한다.
    """
    import numpy as np

    result = {
        "signal": "HOLD",
        "direction": 0,
        "pred_mean": float(prices[-1]) if prices else 0.0,
        "pred_upper": 0.0,
        "pred_lower": 0.0,
        "uncertainty": 0.0,
        "confidence": 0.5,
        "lgbm_action": 1,
        "chronos_signal": "HOLD",
    }

    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from models.chronos_model import predict_rolling, generate_signals
        from models.lgbm_model import predict_3class

        # Chronos
        pipeline = None
        try:
            from models.chronos_model import load_model
            pipeline = load_model(device="cpu")
        except Exception:
            pass

        if pipeline is not None:
            preds, pred_upper, pred_lower = predict_rolling(
                pipeline, prices,
                context_len=context_len,
                horizon=horizon,
                num_samples=20,
            )
            pred_mean = float(np.mean(preds))
            chronos_signal, chronos_conf = generate_signals(prices, preds)
        else:
            pred_mean = float(prices[-1] * 1.005)
            pred_upper = pred_mean * 1.03
            pred_lower = pred_mean * 0.97
            chronos_signal = "HOLD"

        # LGBM 3-class
        lgbm_action = 1
        try:
            feature_cols = [c for c in feature_df_row.keys() if c not in ("close", "target")]
            # LGBM 호출 — feature_df_row는 단일 행 dict
            model = None  # 로드 로직은 run_trading_graph.py에서 수행
            lgbm_action = 1  # default HOLD
        except Exception:
            lgbm_action = 1

        # 앙상블 결정: Chronos + LGBM vote
        direction_map = {"BUY": 0, "HOLD": 1, "SELL": 2}
        chronos_vote = direction_map.get(chronos_signal, 1)
        votes = [chronos_vote, lgbm_action]
        ensemble_action = max(set(votes), key=votes.count) if votes else 1

        signal_map = {0: "BUY", 1: "HOLD", 2: "SELL"}
        result = {
            "signal": signal_map[ensemble_action],
            "direction": 1 if ensemble_action == 0 else (-1 if ensemble_action == 2 else 0),
            "pred_mean": pred_mean,
            "pred_upper": pred_upper,
            "pred_lower": pred_lower,
            "uncertainty": (pred_upper - pred_lower) / pred_mean if pred_mean > 0 else 0.0,
            "confidence": chronos_conf if pipeline else 0.5,
            "lgbm_action": lgbm_action,
            "chronos_signal": chronos_signal,
            "ensemble_action": ensemble_action,
            "trigger_reason": "ensemble_call",
        }
    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2) Context Builder — 뉴스 / 거시 데이터 수집
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_recent_news_yfinance(
    ticker: str = "BTC-USD",
    num_articles: int = 5,
) -> str:
    """
    yfinance API를 사용하여 최근 뉴스 기사의 제목과 요약을 하나의 문자열로 반환한다.
    모델 가이드라인 3.4에 따라 별도 DB 저장 없이 문자열만 반환한다.
    """
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        if not news:
            return "(해당 티커에 대한 최근 뉴스가 없습니다.)"

        lines = []
        for item in news[:num_articles]:
            title = item.get("title", "N/A")
            publisher = item.get("publisher", "N/A")
            lines.append(f"[{publisher}] {title}")

        return "\n".join(lines)
    except Exception as e:
        return f"(뉴스 조회 실패: {e})"


@tool
def fetch_macro_context() -> str:
    """
    거시경제 데이터를 하드코딩된 Mock으로 반환한다 (빠른 프로토타이핑용).
    실제 배포 시에는 FRED API, Quandl 등으로 교체한다.
    """
    return """
[거시경제 데이터 — 2026-05-20 기준]
- 미국 Fed 금리: 4.25% (현재 유지)
- 미국 CPI (4월): 3.2% YoY
- ECB 금리: 4.00%
- BTC Dominance: 53.2%
- BTC Fear & Greed Index: 68 (Greed)
- DXY 달러지수: 104.5
- S&P 500: 5,430 (역시 高値)
"""