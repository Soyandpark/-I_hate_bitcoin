### model_guide.md about llm agents for bitcoin
1. **프로젝트 개요 (System Overview)**본 프로젝트는 암호화폐(BTC/USDT) 트레이딩을 위한 다중 에이전트 LLM 시스템(Multi-Agent System)을 구축하는 것입니다.기존 I_Hate_BitCoin 레포지토리의 시계열 ML 엔진(LightGBM, Chronos)을 'Base Predictor'로 활용하고, FinCon 프레임워크의 핵심 철학(수직적 계층 구조, CVRF, 이중 리스크 제어)과 TradingAgents의 전문 애널리스트 토론 구조를 결합합니다.모든 에이전트 파이프라인은 LangChain 및 LangGraph를 사용하여 구현해야 합니다.
2.**핵심 아키텍처 및 에이전트 역할 (Architecture & Roles)**시스템은 수평적 통신(P2P)을 배제하고, 철저한 수직적 계층 구조(Manager-Analyst Hierarchy)를 따릅니다.
2.1. 하부 인프라 (Data & Base Predictor)Base Predictor: models/chronos_model.py와 models/lgbm_model.py를 활용하여 정량적 시계열 예측 수행. (TTA 개념을 적용하여 최근 윈도우 기반 예측 분산 및 확률 산출)Context Builder: LangChain RAG/VectorDB를 사용해 뉴스, 온체인 데이터, 소셜 감성 데이터를 수집하고 각 에이전트의 역할에 맞게 격리된 프롬프트 컨텍스트로 압축하여 전달.

### 2.1. 하부 인프라 (Data & Base Predictor) - 🚀 쾌속 프로토타이핑 모드 버전
현재는 빠른 아키텍처 검증이 최우선이므로, 복잡한 Vector DB나 외부 유료 뉴스 API(RAG 파이프라인) 구축은 생략하고 가장 가벼운 방식으로 컨텍스트를 구성합니다.

* **Context Builder (뉴스/거시 데이터 수집):** 복잡한 크롤링 대신 다음 두 가지 "빠른 테스트(Quick Test)" 방식 중 하나를 구현하여 에이전트에게 텍스트 컨텍스트를 제공하세요.
  
1. **yfinance API 활용 (권장):** `yfinance.Ticker("BTC-USD").news`를 호출하면 최근 뉴스 기사의 제목(title)과 짧은 요약(publisher, relatedTickers)을 무료로 즉시 가져올 수 있습니다. 이 텍스트 딕셔너리를 그대로 `TradingState`의 뉴스 컨텍스트로 주입하세요.

2. **Mock Data 주입 (최소 시간):** yfinance 연동조차 번거롭다면, 테스트 스크립트 실행 시 인위적으로 극단적인 뉴스 문자열(예: "미국 SEC, 비트코인 현물 ETF 승인 취소 논의", "테슬라, 비트코인 대규모 추가 매수 발표")을 하드코딩하여 State에 밀어넣고, 가설 에이전트와 리스크 매니저가 이 텍스트에 어떻게 반응하는지만 우선 테스트하세요.


2.2. 애널리스트 계층 (Analyst Layer - 병렬 실행)각 애널리스트는 타 에이전트와 소통하지 않고 본인의 데이터만 분석하여 독립적인 보고서를 제출합니다.기술적(Technical) 애널리스트: MACD, RSI, SMA, VWMA 등 config.TECH_INDICATORS 데이터 분석.거시/뉴스(Macro/News) 애널리스트: 금리, 전역 뉴스 등 거시 경제 지표 분석.온체인/감성(On-chain/Sentiment) 애널리스트: 공포탐욕지수, SNS 데이터 기반 군중 심리 분석.2.3. 매니저 계층 (Manager Layer - 3단계 순차 실행)애널리스트들의 보고서는 다음 3단계 에이전트 체인을 거칩니다.가설 에이전트 (Hypothesis Agent): 강세(Bull)와 약세(Bear) 관점을 모두 생성하여 발산적 시나리오 도출 (TradingAgents의 토론 메커니즘 차용).투자결정 에이전트 (Investment Decision Agent & Risk Manager): 가설을 바탕으로 에피소드 내 리스크 제어(Within-Episode Control) 실행. 실시간 CVaR, ATR, Base Predictor의 예측 분산값을 확인하여 허용 리스크 초과 시 매수 기각 또는 비중 축소(Hold/Sell).최종판단 에이전트 (Final Judgment Agent): 과거 일화적 기억(Episodic Memory)을 참조하여 환각/오류를 검증하고 최종 Action(0=Buy, 1=Hold, 2=Sell)과 자본 투입 비중(Weight)을 결정.

**3. LangGraph 구현 지침 (LangGraph Implementation Rules)**
3.1. State 정의 (TypedDict)그래프를 관통하는 전역 상태(State)는 다음 정보들을 반드시 포함해야 합니다.
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
3.2 노드(Node) 설계 원칙
순수 함수 지향: 각 노드(에이전트)는 상태(State)를 입력받아 처리 후 필요한 부분만 업데이트하여 반환해야 합니다.
도구(Tool) 바인딩: 데이터 수집 기능(API 호출 등)은 LangChain의 @tool 데코레이터를 사용하여 Base Predictor나 Context Builder 노드에 바인딩합니다
### 3.3. 온디맨드 추론(On-Demand Inference) 트리거 조건
Claude는 에이전트가 `trigger_custom_inference` 도구를 무분별하게 호출하지 않고, 오직 다음의 조건이 충족될 때만 호출하도록 프롬프트/로직을 설계해야 합니다.

* **조건 1 (변동성 돌파):** 기술적 애널리스트가 계산한 현재 ATR 값이 직전 에피소드 대비 급증했거나, 볼린저 밴드 상/하단을 크게 이탈하여 **캐시된 예측값의 유효성이 만료되었다고 판단될 때.**
* **조건 2 (시그널 충돌):** 뉴스/감성 애널리스트의 강한 방향성(예: 강력한 호재 뉴스)과 캐시된 정량 예측값(Base Signal)의 방향성이 완전히 반대일 때, **가설 에이전트가 더 짧거나 긴 윈도우(`context_len`)로 검증이 필요하다고 판단할 때.**
* **조건 3 (불확실성 제어):** 캐시된 예측값의 상단(pred_upper)과 하단(pred_lower)의 편차가 비정상적으로 넓어 신뢰할 수 없을 때.

**구현 제약사항:** 위 조건에 해당하지 않는 평범한 횡보장이나 일반적인 상황에서는 무조건 비용과 속도 절감을 위해 `get_latest_base_signal()` 도구를 사용하여 캐시 데이터를 읽어야 합니다.


### 3.3. LangGraph 도구(Tool) 설계 지침 추가사항 [뉴스 관련 추가]
* `get_recent_news_yfinance(ticker)`: LangChain `@tool`로 정의하되, 내부 로직은 `yfinance` 라이브러리를 사용해 최근 뉴스 5개의 제목과 요약본을 하나의 문자열(String)로 이어 붙여 반환하는 매우 단순한 함수로 작성하세요. 별도의 데이터베이스 저장은 필요 없습니다.

.**4. 핵심 알고리즘 구현 지침 (Core Algorithms to Implement)**
4.1. 개념적 언어 강화 (CVRF - Conceptual Verbal Reinforcement)이 프로젝트의 가장 중요한 차별점입니다. LLM의 파라미터를 수정하지 않고 프롬프트를 진화시켜야 합니다. 백테스트의 각 에피소드(예: 1주일 또는 1개월 단위)가 끝날 때마다 다음 로직을 실행하는 별도의 함수/노드를 구성하세요.backtester.py의 portfolio_values 및 trade_log를 스캔하여 연속 수익 구간(Profitable Streak)과 연속 손실 구간(Losing Streak)을 추출합니다.LLM(Supervisor)을 호출하여 해당 구간에서 에이전트들이 왜 그런 결정을 내렸는지 분석하고 개념화된 규칙(Conceptual Perspectives)을 텍스트로 생성합니다. (예: "MACD 데드크로스 상황에서 뉴스가 호재여도 진입하면 손실이 컸다.")이전 에피소드와의 결정 중복도(학습률 $\tau$)를 계산합니다.TradingState 내의 current_prompts (매니저 및 관련 애널리스트의 시스템 프롬프트)에 이 "경험 규칙"을 텍스트로 추가/수정하여 다음 에피소드에 전달합니다.4.2. 이중 수준 리스크 제어 (Dual-level Risk Control)에피소드 내 제어 (투자결정 에이전트 담당): 현재 자산의 단기 하락 확률(Base Predictor 결과)과 ATR(변동성, config.py 참조)을 기반으로 실시간 CVaR을 추정합니다. CVaR 임계치를 넘거나 당일 PnL이 마이너스인 상태면 무조건 HOLD 또는 SELL 상태로 강제하는 로직을 LangGraph 조건부 엣지(Conditional Edge)로 구현하세요.
5. **기존 코드베이스 통합 지침 (Integration Guidelines)**
Claude는 코드를 작성할 때 다음 파일들을 반드시 Import하고 연동해야 합니다.설정 관리: config.py의 상수를 적극 활용하세요 (예: TECH_INDICATORS, INITIAL_BALANCE, 액션 기준 ACTION_THRESHOLDS).시계열 예측: models/chronos_model.py의 predict_rolling 및 models/lgbm_model.py를 직접 호출하여 도출된 predictions 배열을 State의 base_predictions로 주입하세요.백테스트 평가: LangGraph 루프가 완료되어 시계열 전체에 대한 actions(0, 1, 2) 리스트가 생성되면, 반드시 backtester.py의 run_3action() 함수를 호출하여 수익률, MDD, Sharpe Ratio를 도출해야 합니다.6. Claude Code를 위한 코딩 규칙 (Coding Standards)모듈화: LangGraph의 Node 함수, State 정의, Tool 함수는 하나의 거대한 파일에 넣지 말고 agents/, tools/, graphs/ 폴더로 나누어 작성하세요.프롬프트 관리: 프롬프트 템플릿은 코드 내에 하드코딩하지 말고, prompts/ 디렉토리 내에 별도 파일로 분리하거나 CVRF가 쉽게 읽고 쓸 수 있는 JSON/DB 구조로 관리하세요.로깅 (Logging): 의사결정의 이유를 추적하는 것이 생명입니다. 각 에이전트가 어떤 근거로 결정을 내렸는지 (특히 CVRF가 프롬프트를 어떻게 업데이트했는지) 파일로 저장하는 로깅 모듈을 포함하세요.Hallucination 방지: 최종 액션 출력 시, LLM이 불필요한 설명을 덧붙이지 못하도록 LangChain의 with_structured_output (Pydantic 모델)을 사용하여 엄격한 JSON 형태로 action, weight, reasoning을 반환받도록 강제하세요.