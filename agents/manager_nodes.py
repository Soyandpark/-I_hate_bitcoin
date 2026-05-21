"""
Manager Layer — 3단계 순차 에이전트 노드 (LangGraph Node 함수)
  1. Hypothesis Agent       : 강세/약세 시나리오 도출
  2. Investment Decision Agent : risk_tools 기반 CVaR/ATR 리스크 제어 + 비중 조절
  3. Final Judgment Agent   : 환각 검증 후 final_decision(0|1|2) + position_weight 최종 결정

각 노드는 TradingState를 입력받아 수정할 필드만 반환하는 순수 함수이다.

⚠️ ChatAnthropic은 지연 로딩(lazy init) — API 키 없으면 Mock 응답 반환
"""
import json
import re
import os
import random
from typing import Dict, Any

from graphs.trading_state import TradingState
from prompts.prompt_templates import (
    SYSTEM_PROMPT_HYPOTHESIS,
    SYSTEM_PROMPT_INVESTMENT_DECISION,
    SYSTEM_PROMPT_FINAL_JUDGMENT,
    build_agent_system_prompt,
    build_cvrf_rules_str,
    RISK_THRESHOLDS,
)
from tools.risk_tools import (
    calculate_cvar,
    calculate_atr,
    assess_overall_risk,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM helper (지연 로딩 — ANTHROPIC_API_KEY 없을 시 Mock 반환)
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm():
    """ChatAnthropic LLM 반환 (lazy init — API 키 없으면 ImportError 방지)"""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.2,
    )


def _invoke_llm(
    system_prompt: str,
    user_content: str,
    agent_type: str = "manager",
) -> str:
    """시스템 프롬프트 + 유저 메시지로 LLM 호출. API 키 없으면 Mock 응답 반환."""
    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        return str(response.content)
    except ImportError:
        return _mock_llm_response(agent_type)
    except Exception as e:
        if "API key" in str(e) or "auth" in str(e).lower():
            return _mock_llm_response(agent_type)
        return f"[LLM invoke error: {e}]"


def _mock_llm_response(agent_type: str) -> str:
    """API 키 없을 때 모의 LLM 응답을 반환 (테스트용)"""
    if agent_type == "hypothesis":
        return json.dumps({
            "bull_scenario": "Mock: 기술적 강세 — RSI 과매도 구간에서 반등 예상, 목표가 $67,000",
            "bear_scenario": "Mock: 거시적 압력 — 금리 인상 우려로 조정 가능, 지지선 $63,000",
            "primary_bias": "BULL",
            "confidence": 0.68,
        })
    elif agent_type == "decision":
        return json.dumps({
            "max_loss_usd": 1500,
            "cvar_estimate": 0.08,
            "atr_stop_loss_pct": 0.025,
            "entry_decision": "ENTER",
            "recommended_weight": 0.6,
            "reasoning": "Mock: 리스크 관리 가능 범위内 — ATR 기반 손절 설정",
        })
    else:  # final
        return json.dumps({
            "action": 0,
            "weight": 0.6,
            "reasoning": "Mock: 강세 시나리오 + 기술적 반등 신호 — BUY 60% 비중 권고",
        })


def _parse_json_output(raw: str) -> Dict[str, Any]:
    """LLM 응답에서 첫 번째 JSON 객체를 추출하여 파싱한다."""
    # Try greedy first, then fallback
    for pattern in [
        r'\{(?:[^{}]|"[^"]*")*\}',
        r'\{[\s\S]*?\}',
    ]:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Hypothesis Agent
# ─────────────────────────────────────────────────────────────────────────────

def node_hypothesis_agent(state: TradingState) -> Dict[str, Any]:
    """
    강세(Bull)와 약세(Bear) 관점의 시나리오를 각각 1개 이상 도출한다.
    TradingAgents의 토론 메커니즘을 차용, 발산적 시나리오 도출에 집중한다.
    """
    # ── CVRF 규칙 주입 ──────────────────────────────────────────────────────
    cvrf_str = build_cvrf_rules_str(state.get("episodic_memory", {}))
    system_prompt = build_agent_system_prompt("hypothesis", cvrf_str)

    # ── Analyst 레포트 취합 ────────────────────────────────────────────────
    reports = state.get("analyst_reports", {})
    analyst_summary = "\n\n".join([
        f"=== {key.upper()} ANALYST ===\n{val}"
        for key, val in reports.items()
    ]) or "(아직 Analyst 레포트가 없습니다.)"

    # ── Base Predictor 결과 ────────────────────────────────────────────────
    base_preds = state.get("base_predictions", {})
    base_summary = f"""
[Base Predictor 결과]
- 신호: {base_preds.get('signal', 'N/A')}
- 예측 방향: {base_preds.get('direction', 'N/A')}
- 예측 평균가: {base_preds.get('pred_mean', 'N/A')}
- 불확실성 (uncertainty): {base_preds.get('uncertainty', 'N/A')}
- 신뢰도: {base_preds.get('confidence', 'N/A')}
"""

    user_content = f"""
아래는 세 명의 애널리스트 보고서입니다. 이를 종합하여 강세(Bull)와 약세(Bear) 시나리오를 도출하세요.

{analyst_summary}

{base_summary}

[뉴스 컨텍스트]
{state.get('news_context', '(없음)')}
"""

    raw = _invoke_llm(system_prompt, user_content, agent_type="hypothesis")
    parsed = _parse_json_output(raw)

    # Fallback
    if not parsed or "bull_scenario" not in parsed:
        parsed = {
            "bull_scenario": "(분석 실패 — 기본값: 유지)",
            "bear_scenario": "(분석 실패 — 기본값: 유지)",
            "primary_bias": "NEUTRAL",
            "confidence": 0.5,
        }

    hypotheses_text = json.dumps(parsed, ensure_ascii=False, indent=2)

    return {
        "hypotheses": hypotheses_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Investment Decision Agent (兼 Risk Manager)
# ─────────────────────────────────────────────────────────────────────────────

def node_investment_decision(state: TradingState) -> Dict[str, Any]:
    """
    CVaR, ATR, Base Predictor 분산을 기준으로 실시간 리스크 제어를 수행한다.
    허용 리스크 초과 시: 매수 기각 → HOLD/SELL 강제 조정.

    Implements: 이중 수준 리스크 제어 — "에피소드 내 제어"
    """
    # ── CVRF 규칙 주입 ──────────────────────────────────────────────────────
    cvrf_str = build_cvrf_rules_str(state.get("episodic_memory", {}))
    system_prompt = build_agent_system_prompt("decision", cvrf_str)

    # ── 리스크 계산 ──────────────────────────────────────────────────────────
    market_data = state.get("market_data", {})
    indicators = market_data.get("indicators", {})
    price_data = market_data.get("price_data", {})

    # ATR 계산 (14-period)
    highs = price_data.get("high_list", [])
    lows = price_data.get("low_list", [])
    closes = price_data.get("close_list", [])
    atr_value = calculate_atr(highs, lows, closes, period=14) if highs else 0.0

    # CVaR 계산 (직전 20 기간 수익률)
    returns_history = market_data.get("recent_returns", [])
    cvar_value = calculate_cvar(returns_history, confidence_level=0.95) if returns_history else 0.0

    # Base Predictor uncertainty
    base_preds = state.get("base_predictions", {})
    uncertainty = base_preds.get("uncertainty", 0.0)
    pred_mean = base_preds.get("pred_mean", price_data.get("close", 0.0) or 0.0)

    # ATR 기반 손절 비율 (진입价的의 ATR × multiplier)
    atr_multiplier = RISK_THRESHOLDS["atr_multiplier"]
    current_price = price_data.get("close", 0.0) or 0.0
    atr_stop_pct = (atr_value / current_price) * atr_multiplier if current_price > 0 else 0.0

    # ── 리스크 임계치 검사 ───────────────────────────────────────────────────
    cvar_max = RISK_THRESHOLDS["cvar_max"]          # 0.15
    weight_raw = state.get("position_weight", 0.5)

    risk_result = assess_overall_risk(
        cvar=cvar_value,
        atr_stop_pct=atr_stop_pct,
        position_weight=weight_raw,
        cvar_max=cvar_max,
    )

    # ── LLM 호출 ────────────────────────────────────────────────────────────
    hypotheses_parsed = {}
    try:
        hypotheses_parsed = json.loads(state.get("hypotheses", "{}"))
    except json.JSONDecodeError:
        pass

    user_content = f"""
아래는 투자결정을 위한 정량 리스크 데이터입니다.

[리스크 지표]
- CVaR (95%): {cvar_value:.4f}  (임계치: {cvar_max})
- ATR(14): {atr_value:.4f}
- ATR 기반 손절 %: {atr_stop_pct:.4f}
- 예측 불확실성: {uncertainty:.4f}
- 현재가가: {current_price}

[Base Predictor]
- 신호: {base_preds.get('signal', 'N/A')}
- 방향: {base_preds.get('direction', 'N/A')}
- 신뢰도: {base_preds.get('confidence', 'N/A')}

[가설 시나리오]
- 강세: {hypotheses_parsed.get('bull_scenario', 'N/A')}
- 약세: {hypotheses_parsed.get('bear_scenario', 'N/A')}
- 편향: {hypotheses_parsed.get('primary_bias', 'N/A')}

[이전 리스크 평가 결과]
- 리스크 수준: {risk_result['risk_level']}
- 권장 행동: {risk_result['recommendation']}
- 조정된 비중: {risk_result['override_weight']:.4f}
"""

    raw = _invoke_llm(system_prompt, user_content, agent_type="decision")
    parsed = _parse_json_output(raw)

    # Fallback: 리스크 초과 시
    if risk_result["risk_level"] == "HIGH":
        entry_decision = "SKIP"
        recommended_weight = risk_result["override_weight"]
        reasoning = f"CVaR 임계치 초과 — 진입 거부, 비중 {recommended_weight:.2%}로 축소"
    elif not parsed or "entry_decision" not in parsed:
        entry_decision = "HOLD"
        recommended_weight = 0.3
        reasoning = "(분석 실패 — 기본값: HOLD, 비중 30%)"
    else:
        entry_decision = parsed.get("entry_decision", "HOLD")
        recommended_weight = float(parsed.get("recommended_weight", 0.5))

    # 최종 비중 리스크 필터 적용
    final_weight = min(
        RISK_THRESHOLDS["weight_max"],
        max(RISK_THRESHOLDS["weight_min"], recommended_weight),
    )

    risk_assessment: Dict[str, Any] = {
        "cvar": round(cvar_value, 6),
        "atr": round(atr_value, 4),
        "atr_stop_pct": round(atr_stop_pct, 4),
        "uncertainty": round(uncertainty, 4),
        "risk_level": risk_result["risk_level"],
        "entry_decision": entry_decision,
        "recommended_weight": round(final_weight, 4),
        "llm_raw_output": raw[:500],   # 디버깅용
    }

    return {
        "risk_assessment": risk_assessment,
        "position_weight": final_weight,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Final Judgment Agent
# ─────────────────────────────────────────────────────────────────────────────

def node_final_judgment(state: TradingState) -> Dict[str, Any]:
    """
    환각/오류 검증을 수행하고, final_decision(0=Buy, 1=Hold, 2=Sell)과
    최종 position_weight를 TradingState에 엄격하게 업데이트한다.

    Pydantic 스키마로 LLM 응답을 강제하여 0, 1, 2 외의 값이 나오지 않도록 한다.
    """
    # ── Typed output validation ─────────────────────────────────────────────
    VALID_ACTIONS: set[int] = {0, 1, 2}

    def _clamp_action(value: Any) -> int:
        try:
            v = int(value)
            return v if v in VALID_ACTIONS else 1  # default = HOLD
        except (TypeError, ValueError):
            return 1

    def _clamp_weight(value: Any) -> float:
        try:
            v = float(value)
            return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return 0.0

    # ── CVRF 규칙 주입 ──────────────────────────────────────────────────────
    cvrf_str = build_cvrf_rules_str(state.get("episodic_memory", {}))
    system_prompt = build_agent_system_prompt("final", cvrf_str)

    # ── 리스크 평가 결과 ────────────────────────────────────────────────────
    risk_assessment = state.get("risk_assessment", {})
    entry_decision = risk_assessment.get("entry_decision", "HOLD")
    risk_level = risk_assessment.get("risk_level", "NORMAL")

    # ── Hypothesis 파싱 ─────────────────────────────────────────────────────
    hypotheses_parsed: Dict[str, Any] = {}
    try:
        hypotheses_parsed = json.loads(state.get("hypotheses", "{}"))
    except json.JSONDecodeError:
        pass

    primary_bias = hypotheses_parsed.get("primary_bias", "NEUTRAL")

    # ── LLM 호출 ────────────────────────────────────────────────────────────
    base_preds = state.get("base_predictions", {})
    analyst_reports = state.get("analyst_reports", {})

    user_content = f"""
아래는 최종 판단을 위한 종합 데이터입니다.
당신은 환각과 근거 없는 확신을 감별하고, 최종 행동을 결정합니다.

[투자결정 결과]
- 리스크 수준: {risk_level}
- 진입 의사: {entry_decision}
- 권장 비중: {risk_assessment.get('recommended_weight', 'N/A')}

[가설 시나리오]
- 편향: {primary_bias}
- 강세: {hypotheses_parsed.get('bull_scenario', 'N/A')}
- 약세: {hypotheses_parsed.get('bear_scenario', 'N/A')}

[Base Predictor]
- 신호: {base_preds.get('signal', 'N/A')}
- 방향: {base_preds.get('direction', 'N/A')}
- 신뢰도: {base_preds.get('confidence', 'N/A')}
- 불확실성: {base_preds.get('uncertainty', 'N/A')}

[Analyst 신호 요약]
- 기술적: {analyst_reports.get('technical', 'N/A')}
- 거시: {analyst_reports.get('macro', 'N/A')}
- 온체인: {analyst_reports.get('onchain', 'N/A')}

[에피소드 메모리]
{state.get('episodic_memory', {})}

출력 형식: 반드시 JSON으로만 응답하라.
{{"action": 0|1|2, "weight": 0.0~1.0, "reasoning": "<100단어 이내>"}}
"""

    raw = _invoke_llm(system_prompt, user_content, agent_type="final")
    parsed = _parse_json_output(raw)

    # ── Strict type enforcement ────────────────────────────────────────────
    action: int = _clamp_action(parsed.get("action"))
    weight: float = _clamp_weight(parsed.get("weight"))
    reasoning: str = str(parsed.get("reasoning", "(추론 실패)"))[:200]

    # 리스크 강제 덮어쓰기: HIGH risk 시 무조건 HOLD(1) 이상
    if risk_level == "HIGH" and action == 0:
        action = 1  # Buy → 강제 HOLD
        reasoning = f"[RISK OVERRIDE] {reasoning}"

    # CVaR 임계치 초과 시 → SELL(2)
    if risk_assessment.get("cvar_exceeded", False):
        action = 2
        reasoning = f"[CVaR EXCEEDED] {reasoning}"

    # 최종 비중 필터
    final_weight: float = max(0.0, min(1.0, weight))

    # Map action to signal string for logging
    action_map = {0: "BUY", 1: "HOLD", 2: "SELL"}

    return {
        "final_decision": action,
        "position_weight": final_weight,
        "risk_assessment": {
            **risk_assessment,
            "final_action": action_map[action],
            "final_reasoning": reasoning,
            "llm_raw": raw[:500],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optional — CVRF Meta-Learning Node (에피소드 종료 시 호출)
# ─────────────────────────────────────────────────────────────────────────────

def node_cvrf_update(state: TradingState) -> Dict[str, Any]:
    """
    에피소드 종료 후 포트폴리오 가치 + 거래 로그를 분석하여
    TradingState["current_prompts"]를 업데이트한다.
    """
    from prompts.prompt_templates import SYSTEM_PROMPT_CVRF

    portfolio_values = state.get("episodic_memory", {}).get("portfolio_values", [])
    trade_log = state.get("episodic_memory", {}).get("trade_log", [])
    episode_id = state.get("episode_id", "unknown")

    user_content = f"""
[에피소드 ID] {episode_id}

[포트폴리오 가치 히스토리] (최근순)
{json.dumps(portfolio_values[-20:], ensure_ascii=False) if portfolio_values else "[]"}

[거래 로그] (최근순)
{json.dumps(trade_log[-20:], ensure_ascii=False) if trade_log else "[]"}

이전 에피소드의 current_prompts:
{json.dumps(state.get("current_prompts", {}), ensure_ascii=False, indent=2)}

역할:
1) 연속 수익/손실 구간을 분석한다.
2) 에이전트들의 의사결정 패턴을 개념화한다.
3) TradingState["current_prompts"]를 업데이트할 규칙을 JSON으로 출력한다.
"""

    raw = _invoke_llm(SYSTEM_PROMPT_CVRF, user_content, agent_type="cvrf")
    parsed = _parse_json_output(raw)

    # Update current_prompts with new rules
    new_rules = parsed.get("new_rules", [])
    updated_prompts = dict(state.get("current_prompts", {}))

    for rule in new_rules:
        target = rule.get("target_agent", "")
        if target and target in updated_prompts:
            # Append rule to existing prompt (simple injection strategy)
            updated_prompts[target] += f"\n[CVRF 경험] {rule.get('rule', '')}"

    return {
        "current_prompts": updated_prompts,
        "episodic_memory": {
            **state.get("episodic_memory", {}),
            "last_cvrf_output": parsed,
            "tau": parsed.get("tau", 0.0),
        },
    }