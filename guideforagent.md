# guideforagent.md

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



0. . 하부 인프라 (Data & Base Predictor)Base Predictor: models/chronos_model.py와 models/lgbm_model.py를 활용하여 정량적 시계열 예측 수행. (TTA 개념을 적용하여 최근 윈도우 기반 예측 분산 및 확률 산출)Context Builder: LangChain RAG/VectorDB를 사용해 뉴스, 온체인 데이터, 소셜 감성 데이터를 수집하고 각 에이전트의 역할에 맞게 격리된 프롬프트 컨텍스트로 압축하여 전달.

1. /agent : 애널리스트 계층 (Analyst Layer - 병렬 실행)각 애널리스트는 타 에이전트와 소통하지 않고 본인의 데이터만 분석하여 독립적인 보고서를 제출합니다.기술적(Technical) 애널리스트: MACD, RSI, SMA, VWMA 등 config.TECH_INDICATORS 데이터 분석.거시/뉴스(Macro/News) 애널리스트: 금리, 전역 뉴스 등 거시 경제 지표 분석.온체인/감성(On-chain/Sentiment) 애널리스트: 공포탐욕지수, SNS 데이터 기반 군중 심리 분석.2.3. 매니저 계층 (Manager Layer - 3단계 순차 실행)애널리스트들의 보고서는 다음 3단계 에이전트 체인을 거칩니다.가설 에이전트 (Hypothesis Agent): 강세(Bull)와 약세(Bear) 관점을 모두 생성하여 발산적 시나리오 도출 (TradingAgents의 토론 메커니즘 차용).투자결정 에이전트 (Investment Decision Agent & Risk Manager): 가설을 바탕으로 에피소드 내 리스크 제어(Within-Episode Control) 실행. 실시간 CVaR, ATR, Base Predictor의 예측 분산값을 확인하여 허용 리스크 초과 시 매수 기각 또는 비중 축소(Hold/Sell).최종판단 에이전트 (Final Judgment Agent): 과거 일화적 기억(Episodic Memory)을 참조하여 환각/오류를 검증하고 최종 Action(0=Buy, 1=Hold, 2=Sell)과 자본 투입 비중(Weight)을 결정.

2. . State 정의 (TypedDict)그래프를 관통하는 전역 상태(State)는 다음 정보들을 반드시 포함해야 합니다.
Pythonfrom typing import TypedDict, List, Dict, Any
class TradingState(TypedDict):
    timestamp: str                 # 현재 분석 시점
    market_data: Dict[str, Any]    # 가격, 지표 등 정량 데이터
    base_predictions: Dict[str, Any] # LGBM/Chronos 예측 결과 및 분산도
    analyst_reports: Dict[str, str]# 애널리스트들의 개별 보고서
    hypotheses: str                # 가설 에이전트의 시나리오
    risk_assessment: Dict[str, Any]# CVaR 등 리스크 평가 결과
    final_decision: int            # 0(Buy), 1(Hold), 2(Sell)
    position_weight: float         # 0.0 ~ 1.0 자본 투입 비중
    current_prompts: Dict[str, str]# ★핵심: CVRF로 인해 동적으로 변하는 에이전트별 프롬프트
3.  노드(Node) 설계 원칙
순수 함수 지향: 각 노드(에이전트)는 상태(State)를 입력받아 처리 후 필요한 부분만 업데이트하여 반환해야 합니다.
도구(Tool) 바인딩: 데이터 수집 기능(API 호출 등)은 LangChain의 @tool 데코레이터를 사용하여 Base Predictor나 Context Builder 노드에 바인딩합니다

3.3. 온디맨드 추론(On-Demand Inference) 트리거 조건
 에이전트가 `trigger_custom_inference` 도구를 무분별하게 호출하지 않고, 오직 다음의 조건이 충족될 때만 호출하도록 프롬프트/로직을 설계해야 합니다.

* **조건 1 (변동성 돌파):** 기술적 애널리스트가 계산한 현재 ATR 값이 직전 에피소드 대비 급증했거나, 볼린저 밴드 상/하단을 크게 이탈하여 **캐시된 예측값의 유효성이 만료되었다고 판단될 때.**
* **조건 2 (시그널 충돌):** 뉴스/감성 애널리스트의 강한 방향성(예: 강력한 호재 뉴스)과 캐시된 정량 예측값(Base Signal)의 방향성이 완전히 반대일 때, **가설 에이전트가 더 짧거나 긴 윈도우(`context_len`)로 검증이 필요하다고 판단할 때.**
* **조건 3 (불확실성 제어):** 캐시된 예측값의 상단(pred_upper)과 하단(pred_lower)의 편차가 비정상적으로 넓어 신뢰할 수 없을 때.

**구현 제약사항:** 위 조건에 해당하지 않는 평범한 횡보장이나 일반적인 상황에서는 무조건 비용과 속도 절감을 위해 `get_latest_base_signal()` 도구를 사용하여 캐시 데이터를 읽어야 합니다.

4. 설정 관리: config.py의 상수를 적극 활용하세요 (예: TECH_INDICATORS, INITIAL_BALANCE, 액션 기준 ACTION_THRESHOLDS).시계열 예측: models/chronos_model.py의 predict_rolling 및 models/lgbm_model.py를 직접 호출하여 도출된 predictions 배열을 State의 base_predictions로 주입하세요
백테스트 평가: LangGraph 루프가 완료되어 시계열 전체에 대한 actions(0, 1, 2) 리스트가 생성되면, 반드시 backtester.py의 run_3action() 함수를 호출하여 수익률, MDD, Sharpe Ratio를 도출해야 합니다.6. Claude Code를 위한 코딩 규칙 (Coding Standards)모듈화: LangGraph의 Node 함수, State 정의, Tool 함수는 하나의 거대한 파일에 넣지 말고 agents/, tools/, graphs/ 폴더로 나누어 작성하세요.프롬프트 관리: 프롬프트 템플릿은 코드 내에 하드코딩하지 말고, prompts/ 디렉토리 내에 별도 파일로 분리하거나 CVRF가 쉽게 읽고 쓸 수 있는 JSON/DB 구조로 관리하세요.로깅 (Logging): 의사결정의 이유를 추적하는 것이 생명입니다. 각 에이전트가 어떤 근거로 결정을 내렸는지 (특히 CVRF가 프롬프트를 어떻게 업데이트했는지) 파일로 저장하는 로깅 모듈을 포함하세요.Hallucination 방지: 최종 액션 출력 시, LLM이 불필요한 설명을 덧붙이지 못하도록 LangChain의 with_structured_output (Pydantic 모델)을 사용하여 엄격한 JSON 형태로 action, weight, reasoning을 반환받도록 강제하세요.

# Updated Version 1 

1. data_collector_polygon.py (신규)

Pre-fetch 전용 모듈. 기존 data_collector.py의 load_or_fetch 패턴을 그대로 따름.

# 저장 구조

datasets/
├── btc_polygon_1h_2024-01-01_2025-01-01.json   ← Aggregates
└── btc_polygon_news_2024-01-01_2025-01-01.json  ← Financial News

핵심 함수:

- fetch_polygon_aggregates_batch() — 배치로 기간 분할 요청 (5000개 제한 대응)
- fetch_polygon_news_batch() — 뉴스 폴링, cursor 기반 페이지네이션
- save_aggregates_cache() — JSONlines 또는 단일 JSON 파일로 저장
- load_aggregates_cache() — 백테스트 중 Tool이 호출
- save_news_cache() / load_news_cache()
- ensure_polygon_data() — 있으면 로드, 없으면 자동 fetch 후 저장
1. tools/polygon_tools.py (수정)

Tool 함수 내부에 Cache-First 전략 추가:

Tool 호출
└→ local cache 파일 존재? ─Yes→ 로컬 JSON 로드 → 마크다운 파싱 → 반환
└→ No → live API 호출 → 응답 JSON 저장 → 마크다운 파싱 → 반환

구체 로직:
def *get_aggregates_cache_path(from_date, to_date, ticker):
return os.path.join("datasets", f"polygon_agg*{ticker}*{from_date}*{to_date}.json")

def _load_from_cache(path):
if os.path.exists(path):
with open(path) as f: return json.load(f)
return None

USE_LOCAL_CACHE 환경변수(default True)로 live API 폴백 제어.

```
 if os.path.exists(path):
     with open(path) as f: return json.load(f)
 return None
```

USE_LOCAL_CACHE 환경변수(default True)로 live API 폴백 제어.

---

```
 return None
```

USE_LOCAL_CACHE 환경변수(default True)로 live API 폴백 제어.

```
                        └→ No → live API 호출 → 응답 JSON 저장 → 마크다운 파싱 → 반환
```

구체 로직:
def *get_aggregates_cache_path(from_date, to_date, ticker):
return os.path.join("datasets", f"polygon_agg*{ticker}*{from_date}*{to_date}.json")

def _load_from_cache(path):
if os.path.exists(path):
with open(path) as f: return json.load(f)
return None

USE_LOCAL_CACHE 환경변수(default True)로 live API 폴백 제어.

---

Files to Modify/Create

┌───────────────────────────┬───────────────────────────────────────┐
│           파일            │                 작업                  │
├───────────────────────────┼───────────────────────────────────────┤
│ data_collector_polygon.py │ 신규 — Polygon pre-fetch 수집기       │
├───────────────────────────┼───────────────────────────────────────┤
│ tools/polygon_tools.py    │ 수정 — Cache-First 로직 주입          │
├───────────────────────────┼───────────────────────────────────────┤
│ datasets/                 │ 저장 디렉토리 (gitignore에 이미 등록) │
├───────────────────────────┼───────────────────────────────────────┤
│ api_connection.txt        │ Polygon API 엔드포인트 문서 (참고용)  │
└───────────────────────────┴───────────────────────────────────────┘

---

Reuse Existing Patterns

- data_collector.py의 load_or_fetch → CSV 캐시 패턴 참고
- config.py의 DATA_DIR = "datasets" 재활용
- _parse_aggregates_to_md() / _parse_news_to_md() — 기존 파서 그대로 활용 (변경 없음)

---

Verification

# 1) Pre-fetch 실행 (실제 API 키 필요)

python data_collector_polygon.py --start 2024-01-01 --end 2025-01-01

# 2) Tool이 로컬 캐시에서 읽는지 확인 (API 키 없이)

export POLYGON_API_KEY=  # 비우기
python -c "
from tools.polygon_tools import get_polygon_aggregates, get_polygon_news
print(get_polygon_aggregates.invoke({'from_date':'2024-01-01','to_date':'2024-01-03'}))
"

# 3) 백테스트 실행

python run_trading_graph.py --dry-run

Tool 출력 로그에서 🔧 Tool: get_polygon_aggregates 결과 확인 → 로컬 캐시 읽으면 [CACHE HIT] 태그 표시.