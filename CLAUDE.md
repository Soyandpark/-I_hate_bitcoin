# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BTC/USDT 1-hour trading system with a dual-layer LangGraph agent architecture:
- **Analyst Layer** (parallel): Technical, Macro, On-chain analysts
- **Manager Layer** (sequential): Hypothesis → Investment Decision → Final Judgment
- **CVRF** (meta-learning): Updates agent prompts after each episode based on trading outcomes

## Common Commands

```bash
# Smoke test (no LLM calls)
python smoke_test.py

# Run the full trading graph
python run_trading_graph.py

# ML experiments (from README)
python experiments/run_model_compare.py   # 6-model comparison
python experiments/run_xgb_optuna.py     # Optuna tuning
python experiments/run_ensemble.py        # Ensemble top 3
python experiments/run_ablation_tta.py     # TTA ablation study
python experiments/run_xgb_visualize.py   # Trading visualization

# Data collection (auto-runs on first experiment)
python data_collector.py
```

## Architecture

### Trading State (`graphs/trading_state.py`)
TypedDict that flows through the entire LangGraph pipeline. Key fields:
- `base_predictions`: LGBM/Chronos prediction results
- `analyst_reports`: {"technical", "macro", "onchain"} reports
- `hypotheses`: Bull/Bear scenarios from Hypothesis Agent
- `risk_assessment`: CVaR, ATR metrics
- `final_decision`: 0=Buy, 1=Hold, 2=Sell
- `episodic_memory`: CVRF learning results (profitable_rules, losing_rules)

### Agent Layer (`agents/`)
- `analyst_nodes.py`: 3 parallel nodes → `node_analyst_technical`, `node_analyst_macro`, `node_analyst_onchain`
- `manager_nodes.py`: 3 sequential nodes → `node_hypothesis_agent`, `node_investment_decision`, `node_final_judgment`, `node_cvrf_update`

All nodes are pure functions: input TradingState → return dict of fields to update.

### Graph Builder (`graphs/graph_builder.py`)
`build_trading_graph()` assembles the full LangGraph pipeline with conditional edges:
- On-demand inference routing (cache vs re-run)
- Risk-level conditional routing (HIGH risk → Final Judgment with override)
- CVRF update on episode end

### Prompts (`prompts/prompt_templates.py`)
- System prompts for all 6 agents (analysts + managers + CVRF)
- `build_cvrf_rules_str()` injects episodic memory into prompts
- `build_agent_system_prompt()` assembles agent prompt with CVRF rules
- `RISK_THRESHOLDS`: cvar_max=0.15, atr_multiplier=2.0

### Risk Tools (`tools/risk_tools.py`)
- `calculate_atr()`, `calculate_cvar()`: core risk metrics
- `assess_overall_risk()`: combines CVaR + ATR into risk_level (HIGH/NORMAL)
- `should_trigger_on_demand()`: decides when to re-run base predictor (volatility break, signal conflict, uncertainty threshold)

### CVRF (`agents/manager_nodes.py` - `node_cvrf_update`)
Meta-learning node that analyzes portfolio values + trade logs after each episode. Outputs:
- `new_rules`: conceptual patterns to inject into agent prompts
- `tau`: learning rate (decision overlap between episodes)
- Updates `TradingState["current_prompts"]` for next episode

### Mock LLM Responses
`agents/manager_nodes.py` and `agents/analyst_nodes.py` use lazy loading for ChatAnthropic. If `ANTHROPIC_API_KEY` is not set, they return mock JSON responses — allowing local testing without API access.

## Key Concepts

| Term | Description |
|------|-------------|
| **3-class** | Buy(0)/Hold(1)/Sell(2) classification |
| **LA (Lookahead)** | Label generation lookahead period (default 6 hours) |
| **DZ (Dead Zone)** | Returns within ±DZ classified as Hold |
| **TTA** | Re-training every ~720 hours (30 days) |
| **CVRF** | Conceptual Verbal Reinforcement — episode-based prompt meta-learning |

## Dependencies

Key packages in `requirements.txt`:
- `ccxt==4.2.29` — Binance data collection
- `lightgbm==4.6.0`, `xgboost==2.0.3`, `catboost==1.2.7` — ML models
- `langgraph` — agent orchestration
- `langchain-anthropic` — LLM calls (lazy import)