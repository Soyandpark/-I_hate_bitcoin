"""
TradingState — LangGraph를 관통하는 전역 상태 (TypedDict)
CVRF(Conceptual Verbal Reinforcement)에 의해 current_prompts가 동적으로 업데이트된다.
"""
from typing import TypedDict, List, Dict, Any, Optional


class TradingState(TypedDict, total=False):
    # ── 시점 / 시장 데이터 ─────────────────────────────────────────────────────
    timestamp: str                                   # 현재 분석 시점 (ISO 8601)
    market_data: Dict[str, Any]                     # OHLCV + 기술지표
    base_predictions: Dict[str, Any]                 # LGBM/Chronos 예측 결과 + 분산도

    # ── 애널리스트 레포트 ─────────────────────────────────────────────────────
    analyst_reports: Dict[str, str]                 # {"technical": str, "macro": str, "onchain": str}

    # ── 매니저 계층 결과 ──────────────────────────────────────────────────────
    hypotheses: str                                  # 가설 에이전트의 강세/약세 시나리오
    risk_assessment: Dict[str, Any]                 # CVaR, ATR, volatility 등 리스크 평가
    final_decision: int                             # 0=Buy, 1=Hold, 2=Sell
    position_weight: float                          # 0.0 ~ 1.0 자본 투입 비중

    # ── CVRF에 의해 동적으로 변하는 에이전트별 시스템 프롬프트 ─────────────────
    current_prompts: Dict[str, str]                 # {"analyst_technical": str, ...}

    # ── 온디맨드 추론 트리거 ─────────────────────────────────────────────────
    trigger_custom_inference: bool                  # True면 Base Predictor 재호출
    trigger_reason: Optional[str]                   # 트리거 이유 ("volatility_break", "signal_conflict", "uncertainty")

    # ── 에피소드 메모리 (CVRF 학습 결과) ─────────────────────────────────────
    episodic_memory: Dict[str, Any]                  # {"profitable_rules": [], "losing_rules": []}

    # ── 내부 메타데이터 ───────────────────────────────────────────────────────
    episode_id: str                                 # 현재 에피소드 식별자
    news_context: str                              # yfinance 뉴스 컨텍스트
    run_reason: str                                 # 현재 실행 이유 (테스트/실전)