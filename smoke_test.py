"""Quick smoke test for all modules — no LLM calls"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

# 1) State
from graphs.trading_state import TradingState
print("State keys:", list(TradingState.__annotations__.keys()))

# 2) Prompts
from prompts.prompt_templates import DEFAULT_PROMPTS, build_cvrf_rules_str, RISK_THRESHOLDS
print("Prompts:", list(DEFAULT_PROMPTS.keys()))
print("Risk thresholds:", RISK_THRESHOLDS)
print("CVRF rules empty:", build_cvrf_rules_str({}))

# 3) Risk tools
from tools.risk_tools import calculate_cvar, calculate_atr, assess_overall_risk, should_trigger_on_demand
returns = [0.01, -0.02, 0.005, -0.015, 0.008, -0.01, 0.003, 0.012, -0.005,
           0.007, -0.02, 0.015, -0.008, 0.01, 0.004, -0.012, 0.006, 0.009,
           -0.003, 0.011]
highs = [100 + i*0.5 for i in range(25)]
lows  = [100 + i*0.4 for i in range(25)]
closes = [100 + i*0.45 for i in range(25)]
print(f"CVar: {round(calculate_cvar(returns, 0.95), 6)}")
print(f"ATR(14): {round(calculate_atr(highs, lows, closes, 14), 4)}")
print("Risk assess:", assess_overall_risk(0.05, 0.02, 0.5))
print("Trigger check:", should_trigger_on_demand({"base_predictions": {}}, None))

# 4) Analyst nodes (import only — no execution)
from agents import analyst_nodes
print("Analyst nodes module loaded OK")
print("node_analyst_technical:", analyst_nodes.node_analyst_technical)
print("node_analyst_macro:", analyst_nodes.node_analyst_macro)
print("node_analyst_onchain:", analyst_nodes.node_analyst_onchain)

# 5) Manager nodes
from agents import manager_nodes
print("Manager nodes module loaded OK")
print("node_hypothesis_agent:", manager_nodes.node_hypothesis_agent)
print("node_investment_decision:", manager_nodes.node_investment_decision)
print("node_final_judgment:", manager_nodes.node_final_judgment)

# 6) Build mock state and run analyst nodes (no LLM)
state: TradingState = {
    "timestamp": "2026-05-20T12:00:00",
    "market_data": {
        "price_data": {"close": 65000.0, "open": 64900, "high": 65200, "low": 64700,
                       "close_list": closes[-10:], "high_list": highs[-10:], "low_list": lows[-10:]},
        "indicators": {"rsi": 42.0, "macd": 150.0, "macdh": 20.0, "atr": 180.0,
                       "bb_upper": 66000, "bb_lower": 64000, "cci": -30.0,
                       "sma_10": 64800, "sma_30": 64500, "sma_60": 64000,
                       "ema_12": 64900, "ema_26": 64600, "stoch_k": 45.0, "stoch_d": 42.0},
        "recent_returns": returns[-30:],
        "sentiment": {"fear_greed": 68, "btc_dominance": 53.2},
    },
    "base_predictions": {"signal": "BUY", "direction": 1, "confidence": 0.72,
                         "uncertainty": 0.025, "pred_mean": 65000,
                         "pred_upper": 66000, "pred_lower": 64000},
    "analyst_reports": {},
    "news_context": "① 미국 SEC, 비트코인 현물 ETF 승인 취소 논의 중\n② 테슬라, 비트코인 대규모 추가 매수 발표",
    "macro_context": "Fed 금리 4.25%, CPI 3.2%",
    "episodic_memory": {"profitable_rules": [], "losing_rules": []},
    "episode_id": "test_001",
}

# Run analyst nodes (uses mock LLM — no real API call)
s1 = analyst_nodes.node_analyst_technical(state)
s2 = analyst_nodes.node_analyst_macro(s1)
s3 = analyst_nodes.node_analyst_onchain(s2)
print(f"\nAnalyst reports: {list(s3['analyst_reports'].keys())}")

# Run manager nodes
s4 = manager_nodes.node_hypothesis_agent(s3)
print(f"Hypothesis set: {'hypotheses' in s4}")

s5 = manager_nodes.node_investment_decision(s4)
print(f"Risk assessment keys: {list(s5.get('risk_assessment', {}).keys())}")
print(f"Weight after decision: {s5.get('position_weight')}")

s6 = manager_nodes.node_final_judgment(s5)
print(f"Final decision: {s6.get('final_decision')} (0=BUY,1=HOLD,2=SELL)")
print(f"Final weight: {s6.get('position_weight')}")
print(f"Risk level: {s6.get('risk_assessment', {}).get('risk_level')}")

# 7) Graph builder
from graphs.graph_builder import build_trading_graph, print_graph_diagram
print("\n" + print_graph_diagram())
graph = build_trading_graph()
print(f"Graph compiled: nodes={len(graph.nodes)}, edges={len(graph.edges)}")

print("\n" + "="*60)
print("ALL MODULES PASSED — LangGraph Ready!")
print("="*60)