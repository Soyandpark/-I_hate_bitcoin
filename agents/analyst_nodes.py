"""
Analyst Layer -- 3명의 병렬 Analyst 노드 (LangGraph Node 함수)
각 노드는 순수 함수: State를 입력 -> 수정할 부분만 반환

툴 매핑:
  - Technical Analyst -> get_polygon_aggregates (1시간봉 + 마크다운 테이블)
  - Macro Analyst     -> get_pol_news (BTC/USD 최신 뉴스 마크다운)
  - On-chain Analyst  -> 기존 get_recent_news_yfinance

⚠️ ChatOpenAI 지연 로딩 -- API 키 없으면 Mock JSON 응답 반환
"""
import os
from dotenv import load_dotenv
load_dotenv()

import json
import random
from typing import Dict, Any

from prompts.prompt_templates import (
    SYSTEM_PROMPT_ANALYST_TECHNICAL,
    SYSTEM_PROMPT_ANALYST_MACRO,
    SYSTEM_PROMPT_ANALYST_ONCHAIN,
)
from tools.base_tools import get_recent_news_yfinance
from tools.polygon_tools import get_polygon_aggregates, get_polygon_news


# ─────────────────────────────────────────────────────────────────────────────
# Node Functions ─────────────────────────────────────────────────────────────

def _invoke_analyst_llm(
    system_prompt: str,
    user_content: str,
    agent_type: str = "analyst_technical",
) -> str:
    """공통 LLM 호출 헬퍼 — API 키 없으면 Mock JSON 반환"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            max_tokens=1024,
            temperature=0.2,
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        return str(response.content)
    except (ImportError, Exception) as e:
        if "API key" in str(e) or "auth" in str(e).lower():
            return _mock_analyst_report(agent_type)
        return json.dumps({"report": f"[LLM error: {e}]", "signal": "NEUTRAL", "confidence": 0.5})


def _mock_analyst_report(agent_type: str) -> str:
    mock_data = {
        "analyst_technical": {
            "report": "Mock: RSI 과매도 구간 — MACD 골든크로스 예상, 볼린저 하단 지지 확인",
            "signal": "BULL",
            "confidence": 0.72,
        },
        "analyst_macro": {
            "report": "Mock: Fed 금리 동결 + BTC ETF 승인 기대로 강세 심리 유입 중",
            "signal": "BULL",
            "confidence": 0.65,
        },
        "analyst_onchain": {
            "report": "Mock: Fear & Greed 68 (Greed), Whale 유출량 증가 — 단기 조정 가능성",
            "signal": "NEUTRAL",
            "confidence": 0.58,
        },
    }
    return json.dumps(mock_data.get(agent_type, mock_data["analyst_technical"]))


def node_analyst_technical(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    기술적 분석가 노드: Polygon Aggregates 툴로 1시간봉 데이터를 조회한 뒤
    MACD, RSI, SMA, VWAP, ATR 등 기술지표를 분석한다.

    Tool: get_polygon_aggregates
    Output key: analyst_reports["technical"]
    """
    # ── 1) Polygon Aggregates 툴 호출 (1시간봉 + 마크다운 변환) ──────────────
    price_data = state.get("market_data", {}).get("price_data", {})
    current_price = price_data.get("close", 65000.0)

    # Polygon: 최근 72시간(3일) 데이터 조회
    from datetime import datetime, timedelta
    to_date   = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

    agg_report: str = get_polygon_aggregates.invoke({
        "ticker": "X:BTCUSD",
        "from_date": from_date,
        "to_date": to_date,
        "multiplier": 1,
        "timespan": "hour",
    })

    # ── 2) 기존 기술지표 데이터 ────────────────────────────────────────────────
    market_data = state.get("market_data", {})
    indicators = market_data.get("indicators", {})
    price_data_full = market_data.get("price_data", {})

    tech_context = (
        f"[Polygon Aggregates — 최근 1시간봉]\n{agg_report}\n\n"
        f"[TA 라이브러리 기술지표]\n"
        f"- MACD: {indicators.get('macd', 'N/A')}, hist={indicators.get('macdh', 'N/A')}\n"
        f"- RSI(14): {indicators.get('rsi', 'N/A')}\n"
        f"- SMA(10/30/60): {indicators.get('sma_10', 'N/A')} / {indicators.get('sma_30', 'N/A')} / {indicators.get('sma_60', 'N/A')}\n"
        f"- EMA(12/26): {indicators.get('ema_12', 'N/A')} / {indicators.get('ema_26', 'N/A')}\n"
        f"- ATR(14): {indicators.get('atr', 'N/A')}\n"
        f"- Bollinger Upper/Lower: {indicators.get('bb_upper', 'N/A')} / {indicators.get('bb_lower', 'N/A')}\n"
        f"- CCI: {indicators.get('cci', 'N/A')}\n"
        f"- Stochastic %K/%D: {indicators.get('stoch_k', 'N/A')} / {indicators.get('stoch_d', 'N/A')}\n"
        f"- 현재가: ${price_data_full.get('close', 'N/A')}\n"
    )

    user_message = (
        f"[시장 데이터]\n{tech_context}\n\n"
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
    거시경제/뉴스 분석가 노드: Polygon News 툴로 BTC/USD 최신 뉴스를 조회한 뒤
    금리, Fed 정책, ETF 뉴스, 규제 뉴스 등을 분석한다.

    Tool: get_polygon_news
    Output key: analyst_reports["macro"]
    """
    # ── 1) Polygon News 툴 호출 (BTC/USD 최근 뉴스) ───────────────────────────
    from datetime import datetime, timedelta
    to_date   = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    news_report: str = get_polygon_news.invoke({
        "ticker": "X:BTCUSD",
        "from_date": from_date,
        "to_date": to_date,
        "limit": 10,
        "order": "desc",
    })

    # ── 2) 거시경제 데이터 ───────────────────────────────────────────────────
    market_data = state.get("market_data", {})
    price_data = market_data.get("price_data", {})

    macro_context = (
        f"[Polygon News — BTC/USD 최신]\n{news_report}\n\n"
        f"[가격 데이터]\n"
        f"- 현재가: ${price_data.get('close', 'N/A')}\n"
        f"- 시간: {state.get('timestamp', 'N/A')}\n"
    )

    user_message = (
        f"[시장 데이터]\n{macro_context}\n\n"
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