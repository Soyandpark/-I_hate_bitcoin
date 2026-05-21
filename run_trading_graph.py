"""
run_trading_graph.py — 엔트리 포인트
  1) 데이터 로딩 (실제 없음 → Mock 데이터 생성)
  2) Base Predictor 실행 (LGBM + Chronos)
  3) LangGraph 전체 파이프라인 실행 (invoke)
  4) 백테스트 평가 (backtester.run_3action)

사용법:
  python run_trading_graph.py                # 단일 에피소드
  python run_trading_graph.py --episodes 4  # 4개 에피소드 백테스트
"""
import os, sys, json, random, argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 프로젝트 루트를 PYTHONPATH에 추가
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import INITIAL_BALANCE, TRADING_FEE, TECH_INDICATORS
from backtester import run_3action, print_result
from graphs.trading_state import TradingState
from graphs.graph_builder import build_trading_graph, print_graph_diagram
from prompts.prompt_templates import DEFAULT_PROMPTS, build_cvrf_rules_str
from agents.manager_nodes import node_hypothesis_agent


# ══════════════════════════════════════════════════════════════════════════════
# 0) Mock / Real Data Helpers
# ══════════════════════════════════════════════════════════════════════════════

def generate_mock_ohlcv(n_bars: int = 200, base_price: float = 65_000) -> pd.DataFrame:
    """테스트용 BTC 가격 Mock 데이터 생성"""
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="h")
    np.random.seed(42)
    log_returns = np.random.normal(0.0002, 0.015, n_bars)
    close = base_price * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low  = close * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_ = close * (1 + np.random.normal(0, 0.003, n_bars))
    volume = np.random.uniform(1000, 5000, n_bars) * 1e6

    df = pd.DataFrame({
        "date": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """ta 라이브러리로 기술지표 추가"""
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.momentum import RSIIndicator, MACDIndicator, StochasticOscillator
    from ta.trend import CCIIndicator, ADXIndicator

    df = df.copy()

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=20)
    df["boll_ub"] = bb.bollinger_hband()
    df["boll_lb"] = bb.bollinger_lband()

    # ATR
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr_14"] = atr.average_true_range()

    # RSI
    rsi = RSIIndicator(df["close"], window=14)
    df["rsi_14"] = rsi.rsi()

    # MACD
    macd = MACDIndicator(df["close"])
    df["macd"] = macd.macd()
    df["macds"] = macd.macd_signal()
    df["macdh"] = macd.macd_diff()

    # Stochastic
    stoch = StochasticOscillator(df["high"], df["low"], df["close"])
    df["kdjk"] = stoch.stoch()
    df["kdjd"] = stoch.stoch_signal()

    # CCI
    cci = CCIIndicator(df["high"], df["low"], df["close"])
    df["cci_14"] = cci.cci()

    # ADX / DX
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["dx_14"] = adx.adx_pos()

    # SMAs
    for w in [10, 30, 60]:
        df[f"close_{w}_sma"] = df["close"].rolling(w).mean()
    for w in [10, 30]:
        df[f"close_{w}_ema"] = df["close"].ewm(span=w).mean()

    df.ffill(inplace=True)
    return df


def df_to_market_data(df: pd.DataFrame) -> Dict[str, Any]:
    """DataFrame → TradingState market_data dict 변환"""
    last = df.iloc[-1]
    return {
        "price_data": {
            "close":        float(last["close"]),
            "open":         float(last["open"]),
            "high":         float(last["high"]),
            "low":          float(last["low"]),
            "volume":       float(last["volume"]),
            "close_list":   df["close"].tolist()[-50:],
            "high_list":    df["high"].tolist()[-50:],
            "low_list":     df["low"].tolist()[-50:],
        },
        "indicators": {
            "macd":       float(last["macd"]),
            "macds":      float(last["macds"]),
            "macdh":      float(last["macdh"]),
            "rsi":        float(last["rsi_14"]),
            "atr":        float(last["atr_14"]),
            "bb_upper":   float(last["boll_ub"]),
            "bb_lower":   float(last["boll_lb"]),
            "cci":        float(last["cci_14"]),
            "dx":         float(last["dx_14"]),
            "sma_20":     float(last.get("close_10_sma", np.nan)),
            "sma_50":     float(last.get("close_30_sma", np.nan)),
            "sma_200":    float(last.get("close_60_sma", np.nan)),
            "sma_10":     float(last.get("close_10_sma", np.nan)),
            "sma_30":     float(last.get("close_30_sma", np.nan)),
            "sma_60":     float(last.get("close_60_sma", np.nan)),
            "ema_12":     float(last.get("close_10_ema", np.nan)),
            "ema_26":     float(last.get("close_30_ema", np.nan)),
            "stoch_k":    float(last["kdjk"]),
            "stoch_d":    float(last["kdjd"]),
        },
        "recent_returns": df["close"].pct_change().dropna().tolist()[-30:],
    }


def run_mock_lgbm(df: pd.DataFrame) -> Dict[str, Any]:
    """Mock LGBM 예측 (실제 모델 없이 가상의Buy/Hold/Sell 비율을 반환)"""
    latest_rsi = df["rsi_14"].iloc[-1] if "rsi_14" in df.columns else 50.0
    latest_macd = float(df["macd"].iloc[-1]) if "macd" in df.columns else 0.0

    if latest_rsi < 35 and latest_macd > 0:
        signal = "BUY"; direction = 1; action = 0
    elif latest_rsi > 70 and latest_macd < 0:
        signal = "SELL"; direction = -1; action = 2
    else:
        signal = "HOLD"; direction = 0; action = 1

    return {
        "signal": signal, "direction": direction, "action": action,
        "confidence": round(random.uniform(0.55, 0.85), 3),
        "uncertainty": round(random.uniform(0.01, 0.04), 4),
        "pred_mean": float(df["close"].iloc[-1]),
        "pred_upper": float(df["close"].iloc[-1] * 1.025),
        "pred_lower": float(df["close"].iloc[-1] * 0.975),
    }


def run_mock_chronos(df: pd.DataFrame) -> Dict[str, Any]:
    """Mock Chronos 예측"""
    last_price = float(df["close"].iloc[-1])
    trend = random.choice([1.005, 1.01, 0.995, 1.0])
    pred_mean = last_price * trend
    return {
        "signal": "BUY" if trend > 1.005 else ("SELL" if trend < 0.995 else "HOLD"),
        "direction": 1 if trend > 1.005 else (-1 if trend < 0.995 else 0),
        "pred_mean": pred_mean,
        "pred_upper": pred_mean * 1.03,
        "pred_lower": pred_mean * 0.97,
        "uncertainty": round(random.uniform(0.01, 0.04), 4),
        "confidence": round(random.uniform(0.5, 0.8), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1) Build Initial State
# ══════════════════════════════════════════════════════════════════════════════

def build_initial_state(
    df: pd.DataFrame,
    use_mock: bool = True,
    episode_id: str = "ep_001",
    run_reason: str = "mock_test",
) -> TradingState:
    """초기 TradingState 구성"""

    market_data = df_to_market_data(df)

    if use_mock:
        lgbm_result = run_mock_lgbm(df)
        chronos_result = run_mock_chronos(df)
        base_predictions = {
            **lgbm_result,
            "chronos_signal": chronos_result["signal"],
            "chronos_direction": chronos_result["direction"],
            "chronos_uncertainty": chronos_result["uncertainty"],
        }
    else:
        # 실전: models/lgbm_model.py, models/chronos_model.py 호출
        base_predictions = {
            "signal": "HOLD", "direction": 0, "confidence": 0.5,
            "uncertainty": 0.05, "pred_mean": float(df["close"].iloc[-1]),
            "pred_upper": 0, "pred_lower": 0,
        }

    state: TradingState = {
        "timestamp": datetime.now().isoformat(),
        "market_data": market_data,
        "base_predictions": base_predictions,
        "analyst_reports": {},
        "hypotheses": "",
        "risk_assessment": {},
        "final_decision": 1,      # default HOLD
        "position_weight": 0.5,  # default 50%
        "current_prompts": dict(DEFAULT_PROMPTS),
        "trigger_custom_inference": False,
        "trigger_reason": None,
        "episodic_memory": {
            "profitable_rules": [],
            "losing_rules": [],
            "portfolio_values": [],
            "trade_log": [],
            "episode_end": False,
        },
        "episode_id": episode_id,
        "news_context": (
            "① 미국 SEC, 비트코인 현물 ETF 승인 취소 논의 중\n"
            "② 테슬라, 비트코인 대규모 추가 매수 발표\n"
            "③ 중국 정부, 디지털자산 합법화 검토 중"
        ),
        "macro_context": (
            "[거시경제 데이터]\n"
            "- 미국 Fed 금리: 4.25%\n"
            "- BTC Dominance: 53.2%\n"
            "- Fear & Greed Index: 68 (Greed)\n"
            "- DXY: 104.5"
        ),
        "run_reason": run_reason,
    }
    return state


# ══════════════════════════════════════════════════════════════════════════════
# 2) LangGraph 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_single_episode(
    df: pd.DataFrame,
    use_mock: bool = True,
    episode_id: str = "ep_001",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """단일 에피소드를 실행하고 결과를 반환한다."""
    print(f"\n{'═'*60}")
    print(f"▶ Episode [{episode_id}] 시작 — use_mock={use_mock}")
    print(f"{'═'*60}")

    state = build_initial_state(df, use_mock=use_mock, episode_id=episode_id)

    if dry_run:
        # LangGraph 없이 노드만 순차 호출 (LLM 미사용 빠른 테스트)
        print("[DRY RUN] Analyst + Manager 노드 순차 실행 (LLM 미호출)")
        from agents.analyst_nodes import (
            node_analyst_technical,
            node_analyst_macro,
            node_analyst_onchain,
        )
        from agents.manager_nodes import (
            node_hypothesis_agent,
            node_investment_decision,
            node_final_judgment,
        )

        state = node_analyst_technical(state)
        state = node_analyst_macro(state)
        state = node_analyst_onchain(state)
        state = node_hypothesis_agent(state)
        state = node_investment_decision(state)
        state = node_final_judgment(state)
    else:
        # LangGraph 실행
        graph = build_trading_graph()
        state = graph.invoke(state)

    # 결과 출력
    decision_map = {0: "BUY 🟢", 1: "HOLD 🟡", 2: "SELL 🔴"}
    action = state.get("final_decision", 1)
    weight = state.get("position_weight", 0.0)
    risk = state.get("risk_assessment", {})

    print(f"\n📊 Final Decision:  {decision_map[action]}")
    print(f"   Position Weight: {weight:.2%}")
    print(f"   Risk Level:      {risk.get('risk_level', 'N/A')}")
    print(f"   CVaR:            {risk.get('cvar', 'N/A')}")
    print(f"   ATR Stop %:      {risk.get('atr_stop_pct', 'N/A')}")

    # Analyst 레포트 요약
    reports = state.get("analyst_reports", {})
    for key, val in reports.items():
        try:
            parsed = json.loads(val)
            sig = parsed.get("signal", "N/A")
            conf = parsed.get("confidence", "N/A")
            print(f"   [{key.upper():10s}] signal={sig}, confidence={conf}")
        except json.JSONDecodeError:
            print(f"   [{key.upper():10s}] (파싱 실패 또는 미실행)")

    return {
        "episode_id": episode_id,
        "final_decision": action,
        "position_weight": weight,
        "risk_assessment": risk,
        "analyst_reports": reports,
        "hypotheses": state.get("hypotheses", ""),
        "state": state,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3) Multi-Episode 백테스트
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    n_episodes: int = 4,
    bars_per_episode: int = 200,
    use_mock: bool = True,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Multi-Episode 백테스트 실행.
    각 에피소드는 시간적으로 연결되지 않으며, 독립적으로 실행 후
    모든 action 리스트를 취합하여 backtester.run_3action으로 평가한다.
    """
    print(f"\n{'#'*70}")
    print(f"  📈 Multi-Episode 백테스트 — {n_episodes}개 에피소드")
    print(f"{'#'*70}")

    all_actions: List[int] = []
    all_results: List[Dict] = []

    for ep in range(1, n_episodes + 1):
        np.random.seed(ep * 111)  # 매 에피소드 다른 가격 시나리오
        df = generate_mock_ohlcv(n_bars=bars_per_episode, base_price=65_000 + ep * 1000)
        df = add_technical_indicators(df)

        result = run_single_episode(
            df=df,
            use_mock=use_mock,
            episode_id=f"ep_{ep:03d}",
            dry_run=dry_run,
        )
        all_results.append(result)
        all_actions.append(result["final_decision"])

    # ── 3-Action Backtest 평가 ─────────────────────────────────────────────
    # Mock price list for backtest evaluation
    prices_for_bt = generate_mock_ohlcv(n_bars=n_episodes * bars_per_episode)
    prices_for_bt = add_technical_indicators(prices_for_bt)
    bt_df = prices_for_bt[["date", "open", "high", "low", "close", "volume"]].copy()
    bt_df["target_3class"] = 1  # unknown — accuracy는 None

    # 에피소드 수만큼 action 할당 (반복하여 timeseries 구성)
    actions_for_bt = all_actions * (len(bt_df) // n_episodes + 1)
    actions_for_bt = actions_for_bt[: len(bt_df)]

    bt_result = run_3action(
        actions=actions_for_bt,
        test_df=bt_df,
        initial_balance=INITIAL_BALANCE,
        trading_fee=TRADING_FEE,
    )

    print(f"\n{'═'*60}")
    print("📈 Multi-Episode 백테스트 결과")
    print_result(bt_result)
    print(f"  에피소드별 행동: {all_actions}")

    return {
        "episode_results": all_results,
        "backtest_result": bt_result,
        "all_actions": all_actions,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4) Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LangGraph Trading Agent 실행")
    parser.add_argument("--episodes", type=int, default=1,
                        help="에피소드 수 (기본: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="LLM 미호출 — 노드만 순차 실행 (빠른 검증)")
    parser.add_argument("--real", action="store_true",
                        help="실제 모델 사용 (LGBM + Chronos 호출)")
    parser.add_argument("--bars", type=int, default=200,
                        help="에피소드당 캔들 수 (기본: 200)")
    args = parser.parse_args()

    print(print_graph_diagram())

    use_mock = not args.real

    if args.episodes == 1:
        # ── 단일 에피소드 ────────────────────────────────────────────────────
        df = generate_mock_ohlcv(n_bars=args.bars)
        df = add_technical_indicators(df)
        result = run_single_episode(
            df=df,
            use_mock=use_mock,
            episode_id="ep_single",
            dry_run=args.dry_run,
        )
        print(f"\n✅ 단일 에피소드 완료")
    else:
        # ── Multi-Episode 백테스트 ──────────────────────────────────────────
        run_backtest(
            n_episodes=args.episodes,
            bars_per_episode=args.bars,
            use_mock=use_mock,
            dry_run=args.dry_run,
        )
        print(f"\n✅ Multi-Episode 백테스트 완료 — {args.episodes}개 에피소드")


if __name__ == "__main__":
    main()