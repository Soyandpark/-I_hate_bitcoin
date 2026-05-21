"""
LangGraph 그래프 빌더 — trading_graph
전체 에이전트 파이프라인을 하나의CompiledGraph로 조립한다.

그래프 구조 (수직적 계층):
┌──────────────────────────────────────────────────────────────────┐
│  START                                                              │
│    │                                                                  │
│    ▼                                                                  │
│  [Market Data & Base Signal 수집]  ← On-Demand Inference Trigger    │
│    │                                                                  │
│    ▼                                                                  │
│  ┌─────────────┬──────────────┬─────────────┐  ← Analyst Layer (병렬)
│  │Technical    │Macro         │On-chain     │                        │
│  │Analyst      │Analyst       │Analyst      │                        │
│  └──────┬──────┴──────┬───────┴──────┬──────┘                        │
│         └─────────────┴─────────────┘                               │
│                    │                                                 │
│                    ▼                                                 │
│         [Hypothesis Agent]  ── 시나리오 도출                          │
│                    │                                                 │
│                    ▼                                                 │
│    [Investment Decision Agent + Risk Manager]  ← CVaR/ATR 리스크 제어  │
│                    │                                                 │
│         ┌──────────┴──────────┐                                      │
│         │  Conditional Edge   │                                      │
│    (risk_level == "HIGH")     │ (risk_level == "NORMAL")            │
│         ▼                     ▼                                      │
│    [Final Judgment ───────────┼──→ Final Judgment]                  │
│    Risk-Override path          Normal path                           │
│                    │                                                 │
│                    ▼                                                 │
│         [Final Judgment Agent]  ── action(0|1|2), weight 최종 결정  │
│                    │                                                 │
│                    ▼                                                 │
│  END (final_decision + position_weight 출력)                          │
└──────────────────────────────────────────────────────────────────┘
"""
from typing import Literal, Callable
from langgraph.graph import StateGraph, END

from graphs.trading_state import TradingState
from agents.analyst_nodes import (
    node_analyst_technical,
    node_analyst_macro,
    node_analyst_onchain,
)
from agents.manager_nodes import (
    node_hypothesis_agent,
    node_investment_decision,
    node_final_judgment,
    node_cvrf_update,
)
from tools.risk_tools import should_trigger_on_demand


# ─────────────────────────────────────────────────────────────────────────────
# 0) Helper: On-Demand Inference Router
# ─────────────────────────────────────────────────────────────────────────────

def _route_inference(state: TradingState) -> str:
    """
    온디맨드 추론 트리거 조건을 평가하여 다음 노드를 결정한다.
      - trigger_reason == "initial_run" → run_base_predictor
      - 그 외 조건 충족 시          → run_base_predictor
      - 평범한 상황               → use_cache (run_base_predictor 미실행)
    """
    should_trigger, reason = should_trigger_on_demand(state.get("market_data", {}))

    if should_trigger:
        return "run_on_demand"     # Base Predictor 재실행
    return "use_cache"            # 캐시 사용 (비용/속도 절감)


# ─────────────────────────────────────────────────────────────────────────────
# 1) Analyst Branch Router — 병렬 실행 후 취합 확인
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_analysts(state: TradingState) -> str:
    """
    세 Analyst 노드 병렬 실행 후:
      - 모든 레포트가 존재하면 → Hypothesis Agent
      - 하나라도欠損면       → Partial Report로도 Hypothesis Agent 진행
    """
    reports = state.get("analyst_reports", {})
    required_keys = {"technical", "macro", "onchain"}
    present_keys = set(reports.keys())

    missing = required_keys - present_keys
    if missing:
        # 部分적으로도 진행하되, 로그에欠損을 기록 (state에 기록)
        print(f"[WARNING] Missing analyst reports: {missing}")

    return "hypothesis_agent"


# ─────────────────────────────────────────────────────────────────────────────
# 2) Risk Conditional Edge — CVaR 임계치 초과 시 Final Judgment 경유 결정
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_decision(state: TradingState) -> Literal["final_judgment", "cvrf_update"]:
    """
    Investment Decision Agent 이후:
      - risk_assessment["risk_level"] == "HIGH" → Final Judgment (강제 HOLD/SELL)
      - 그 외                              → Final Judgment (정상 경로)
    """
    risk = state.get("risk_assessment", {})
    if risk.get("risk_level") == "HIGH":
        print(f"[RISK OVERRIDE] CVaR/ATR exceeded — proceeding to Final Judgment")
    return "final_judgment"


# ─────────────────────────────────────────────────────────────────────────────
# 3) Final Judgment 이후
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_final(state: TradingState) -> Literal["cvrf_update", END]:
    """
    Final Judgment Agent 이후:
      - 에피소드 종료 플래그(episodic_memory["episode_end"])가 True면 CVRF 업데이트
      - 그 외                              → END (단일 에피소드 실행)
    """
    episodic = state.get("episodic_memory", {})
    if episodic.get("episode_end", False):
        return "cvrf_update"
    return END


# ─────────────────────────────────────────────────────────────────────────────
# 4) Main Graph Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_trading_graph() -> StateGraph:
    """
    전체 trading 파이프라인 LangGraph를 빌드하고 컴파일하여 반환한다.
    사용법:
        graph = build_trading_graph()
        result = graph.invoke(initial_state)
    """
    # ── StateGraph 초기화 ──────────────────────────────────────────────────
    builder = StateGraph(TradingState)

    # ── Node 등록 ───────────────────────────────────────────────────────────
    # Analyst Layer — 병렬 실행을 위해 START → 같은虚拟노드로 취급
    builder.add_node("analyst_technical", node_analyst_technical)
    builder.add_node("analyst_macro",      node_analyst_macro)
    builder.add_node("analyst_onchain",    node_analyst_onchain)

    # Base Signal / Context Nodes (의사노드 — 실제 데이터 주입은 invoke 전에 수행)
    builder.add_node("on_demand_inference", lambda s: s)  # placeholder, 실제 예측은 tools에서 수행
    builder.add_node("use_cache",           lambda s: s)   # placeholder

    # Manager Layer
    builder.add_node("hypothesis_agent",    node_hypothesis_agent)
    builder.add_node("investment_decision", node_investment_decision)
    builder.add_node("final_judgment",       node_final_judgment)

    # CVRF Meta-Learning
    builder.add_node("cvrf_update",         node_cvrf_update)

    # ── Edge 정의 ───────────────────────────────────────────────────────────

    # START → On-Demand Inference Router
    builder.add_edge("__start__", "on_demand_inference")

    # Inference routing (Conditional Edge)
    builder.add_conditional_edges(
        "on_demand_inference",
        _route_inference,
        {
            "run_on_demand": "use_cache",  # 실행 후 캐시로 전달 (실제로는 tools에서 직접 update)
            "use_cache":    "analyst_technical",
        },
    )

    # ★ Analyst Layer — 3명 병렬 실행 ( START → 3개 노드로 분기 )
    # LangGraph에서 병렬은同一个 라우터에서 3개 노드에 edge 부여로 구현
    builder.add_edge("use_cache", "analyst_technical")
    builder.add_edge("use_cache", "analyst_macro")
    builder.add_edge("use_cache", "analyst_onchain")

    # Analyst 취합 라우터
    builder.add_conditional_edges(
        "__analyst_join__",
        _route_after_analysts,
        {"hypothesis_agent": "hypothesis_agent"},
    )

    # Analyst 완료 후 취합 (3개 노드 → Hypothesis Agent)
    # → 직접 edge 연결 (병렬 완료 대기 없이 각 노드 완료 시 호출)
    for analyst_node in ["analyst_technical", "analyst_macro", "analyst_onchain"]:
        builder.add_edge(analyst_node, "__analyst_join__")

    builder.add_node("__analyst_join__", lambda s: s)   # 가상 취합 노드

    # Hypothesis → Investment Decision
    builder.add_edge("hypothesis_agent", "investment_decision")

    # Investment Decision → Conditional Edge (risk level)
    builder.add_conditional_edges(
        "investment_decision",
        _route_after_decision,
        {
            "final_judgment": "final_judgment",
            "cvrf_update":    "cvrf_update",   # (현재는 사용하지 않음, future use)
        },
    )

    # Final Judgment → END or CVRF
    builder.add_conditional_edges(
        "final_judgment",
        _route_after_final,
        {
            "cvrf_update": "cvrf_update",
            END:           END,
        },
    )

    # CVRF → END
    builder.add_edge("cvrf_update", END)

    # ── 그래프 컴파일 ───────────────────────────────────────────────────────
    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 5) Convenience: Compile & Check
# ─────────────────────────────────────────────────────────────────────────────

def get_trading_graph() -> StateGraph:
    """캐시된 컴파일된 그래프를 반환한다 (싱글톤 패턴)."""
    if not hasattr(get_trading_graph, "_graph"):
        get_trading_graph._graph = build_trading_graph()
    return get_trading_graph._graph


# ─────────────────────────────────────────────────────────────────────────────
# 6) Diagram printer (디버깅용)
# ─────────────────────────────────────────────────────────────────────────────

def print_graph_diagram() -> str:
    """그래프 구조를 ASCII 다이어그램으로 출력한다."""
    return """
trading_graph 흐름도:
────────────────────────────────────────────────────────────────────
  [START]
     │
     ▼
  [on_demand_inference] ──conditional──┬─ run_on_demand
     │                                  └─ use_cache
     ▼
  ┌──────────┬───────────┬──────────┐
  │Technical │ Macro     │ On-chain │   ← Analyst Layer (병렬)
  │Analyst   │ Analyst   │ Analyst  │
  └────┬─────┴─────┬─────┴────┬─────┘
       └──────────┴──────────┘
              │
              ▼
      [__analyst_join__] ──conditional──► [hypothesis_agent]
              │                                  │
              │                                  ▼
              │                       [investment_decision]
              │                                  │
              │                       ──conditional──┬─ HIGH risk
              │                                          ▼
              │                               [final_judgment]
              │                                          │
              │                                          ▼
              │                               [final_judgment] ──conditional──┬─ episode_end=True
              │                                                        │             ▼
              │                                                     END      [cvrf_update]
              │                                                               │
              └──────────────────────────────────────────────────────────────▶END
────────────────────────────────────────────────────────────────────
"""


if __name__ == "__main__":
    print(print_graph_diagram())
    graph = build_trading_graph()
    print(f"\n✅ 그래프 빌드 완료 — 노드: {len(graph.nodes)}, 엣지: {len(graph.edges)}")