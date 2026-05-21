"""
Analyst Layer — 3명의 병렬 Analyst 노드 (LangGraph Node 함수)
각 노드는 순수 함수: State를 입력 → 수정할 부분만 반환

⚠️ ChatAnthropic 지연 로딩 — API 키 없으면 Mock JSON 응답 반환
"""
import json
import random
from typing import Dict, Any

from prompts.prompt_templates import (
    SYSTEM_PROMPT_ANALYST_TECHNICAL,
    SYSTEM_PROMPT_ANALYST_MACRO,
    SYSTEM_PROMPT_ANALYST_ONCHAIN,
)
from tools.base_tools import get_recent_news_yfinance


# ─────────────────────────────────────────────────────────────────────────────
# Node Functions ─────────────────────────────────────────────────────────────

def node_analyst_technical(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    기술적 분석가 노드: MACD, RSI, SMA, VWAP, ATR 등 기술지표를 분석한다.
    """
    market_data = state.get("market_data", {})
    indicators = market_data.get("indicators", {})
    price_data = market_data.get("price_data", {})

    market_context = (
        f"[기술지표]\n"
        f"- MACD: {indicators.get('macd', 'N/A')}, hist={indicators.get('macdh', 'N/A')}\n"
        f"- RSI(14): {indicators.get('rsi', 'N/A')}\n"
        f"- SMA(10/30/60): {indicators.get('sma_10', 'N/A')} / {indicators.get('sma_30', 'N/A')} / {indicators.get('sma_60', 'N/A')}\n"
        f"- EMA(12/26): {indicators.get('ema_12', 'N/A')} / {indicators.get('ema_26', 'N/A')}\n"
        f"- ATR(14): {indicators.get('atr', 'N/A')}\n"
        f"- Bollinger Upper/Lower: {indicators.get('bb_upper', 'N/A')} / {indicators.get('bb_lower', 'N/A')}\n"
        f"- CCI: {indicators.get('cci', 'N/A')}\n"
        f"- Stochastic %K/%D: {indicators.get('stoch_k', 'N/A')} / {indicators.get('stoch_d', 'N/A')}\n"
        f"- 현재가격: ${price_data.get('close', 'N/A')}\n"
    )

    user_message = (
        f"[시장 데이터]\n{market_context}\n\n"
        f"[뉴스 컨텍스트]\n{state.get('news_context', '(없음)')}\n"
    )

    report = _invoke_analyst_llm(
        SYSTEM_PROMPT_ANALYST_TECHNICAL,
        user_message,
        "analyst_technical",
    )

    return {
        "analyst_reports": {
            **state.get("analyst_reports", {}),
            "technical": report,
        }
    }


def node_analyst_macro(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    거시경제/뉴스 분석가 노드: 금리, Fed 정책, ETF 뉴스, 규제 뉴스 등을 분석한다.
    """
    market_data = state.get("market_data", {})
    price_data = market_data.get("price_data", {})

    market_context = (
        f"[가격 데이터]\n"
        f"- 현재가: ${price_data.get('close', 'N/A')}\n"
        f"- 시간: {state.get('timestamp', 'N/A')}\n"
    )

    # yfinance 뉴스 가져오기
    news_str = ""
    try:
        news_str = get_recent_news_yfinance.invoke({"ticker": "BTC-USD", "num_articles": 5})
    except Exception:
        news_str = state.get("news_context", "")

    user_message = (
        f"[시장 데이터]\n{market_context}\n\n"
        f"[뉴스 컨텍스트]\n{news_str}\n\n"
        f"[거시경제 데이터]\n{state.get('macro_context', '(없음)')}\n"
    )

    report = _invoke_analyst_llm(
        SYSTEM_PROMPT_ANALYST_MACRO,
        user_message,
        "analyst_macro",
    )

    return {
        "analyst_reports": {
            **state.get("analyst_reports", {}),
            "macro": report,
        }
    }


def node_analyst_onchain(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    온체인/감성 분석가 노드: 공포탐욕지수, Whale 데이터, SNS 감성 등을 분석한다.
    """
    market_data = state.get("market_data", {})
    indicators = market_data.get("indicators", {})
    sentiment_data = market_data.get("sentiment", {})

    market_context = (
        f"[온체인/감성 데이터]\n"
        f"- Fear & Greed Index: {sentiment_data.get('fear_greed', 'N/A')}\n"
        f"- BTC Dominance: {sentiment_data.get('btc_dominance', 'N/A')}\n"
        f"- 활성 주소수 변화: {sentiment_data.get('active_addresses', 'N/A')}\n"
        f"- 대형 Whale 잔액 변화: {sentiment_data.get('whale_ratio', 'N/A')}\n"
        f"- SNS 긍정/부정 비율: {sentiment_data.get('social_sentiment', 'N/A')}\n"
        f"- Google Trends BTC: {sentiment_data.get('google_trends', 'N/A')}\n"
        f"- RSI(14): {indicators.get('rsi', 'N/A')} (역발전 신호 감지)\n"
    )

    user_message = (
        f"[시장 데이터]\n{market_context}\n\n"
        f"[뉴스 컨텍스트]\n{state.get('news_context', '(없음)')}\n"
    )

    report = _invoke_analyst_llm(
        SYSTEM_PROMPT_ANALYST_ONCHAIN,
        user_message,
        "analyst_onchain",
    )

    return {
        "analyst_reports": {
            **state.get("analyst_reports", {}),
            "onchain": report,
        }
    }