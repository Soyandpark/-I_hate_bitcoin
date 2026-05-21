"""
Trading Agent 전용 로거 — Rich 라이브러리 기반 매매 근거 시각화
각 노드 전환 시 reasoning + 데이터 요약을 컬러 터미널에 출력한다.
"""
import os, sys
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.markdown import Markdown
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None  # fallback to print


# ─────────────────────────────────────────────────────────────────────────────
# 색상 팔레트
# ─────────────────────────────────────────────────────────────────────────────
class _Palette:
    TITLE  = "bold cyan"
    AGENT  = "bold magenta"
    DATA   = "white"
    BULL   = "bold green"
    BEAR   = "bold red"
    NEUTRAL= "bold yellow"
    WARN   = "bold red"
    SUCCESS= "bold green"
    INFO   = "dim"
    REASON = "italic white"


def _rich_available() -> bool:
    """RICH_AVAILABLE은 import 시점에 결정되므로 함수로 래핑"""
    return RICH_AVAILABLE and console is not None


# ─────────────────────────────────────────────────────────────────────────────
# Core: 노드 전환 헤더
# ─────────────────────────────────────────────────────────────────────────────

def log_node_transition(node_name: str, agent_role: str, step: str) -> None:
    """노드 시작 시thick 구분선 + 에이전트 이름 출력"""
    sep = "━" * 70
    if _rich_available():
        console.print(f"\n[cyan]{sep}[/cyan]")
        console.print(f"[bold magenta]▶ [{step}] {agent_role}[/bold magenta]  →  [cyan]{node_name}[/cyan]")
        console.print(f"[cyan]{sep}[/cyan]")
    else:
        print(f"\n{sep}\n▶ [{step}] {agent_role}  →  {node_name}\n{sep}")


def log_tool_call(tool_name: str, result_preview: str, max_len: int = 200) -> None:
    """Tool 호출 결과(마크다운) 표시"""
    preview = result_preview[:max_len] + ("..." if len(result_preview) > max_len else "")
    if _rich_available():
        console.print(f"[dim]🔧 Tool: {tool_name}[/dim]")
        console.print(Panel(
            Text(preview, style="white"),
            title=f"[bold]✅ {tool_name} 결과[/bold]",
            border_style="dim",
            expand=False,
        ))
    else:
        print(f"  🔧 {tool_name}: {preview}")


def log_analyst_report(report_key: str, raw_output: str, signal: str = "N/A", confidence: float = 0.0) -> None:
    """Analyst 노드 결과 출력 (signal + confidence 요약)"""
    # signal 색상 매핑
    color_map = {"BULL": "green", "BEAR": "red", "NEUTRAL": "yellow", "N/A": "dim"}
    sig_color = color_map.get(signal.upper(), "dim")

    conf_str = f"{confidence:.0%}" if isinstance(confidence, float) else str(confidence)

    if _rich_available():
        console.print(f"[dim]  └─ report key: {report_key}[/dim]")
        console.print(f"      signal: [{sig_color}]{signal}[/{sig_color}]  confidence: {conf_str}")
        # Markdown preview (first 300 chars)
        preview = raw_output[:300] if raw_output else "(empty)"
        console.print(Panel(
            Text(preview, style="white"),
            title=f"[bold]Analyst Reasoning[/bold]",
            border_style=sig_color,
            expand=False,
            width=80,
        ))
    else:
        print(f"  └─ [{report_key}] signal={signal}, conf={conf_str}")


def log_hypothesis(hypotheses_json: str) -> None:
    """Hypothesis Agent 결과 — Bull/Bear 시나리오 출력"""
    import json as _json
    try:
        data = _json.loads(hypotheses_json) if hypotheses_json else {}
    except Exception:
        data = {}

    bull    = data.get("bull_scenario", "N/A")
    bear    = data.get("bear_scenario", "N/A")
    bias    = data.get("primary_bias", "NEUTRAL")
    conf    = data.get("confidence", 0.0)

    bias_color = {"BULL": "green", "BEAR": "red", "NEUTRAL": "yellow"}.get(bias.upper(), "dim")

    if _rich_available():
        t = Table(title=f"[bold]🔮 Hypothesis — Bias: [{bias_color}]{bias}[/{bias_color}]  conf={conf:.0%}", expand=True)
        t.add_column("시나리오", style="bold", width=30)
        t.add_column("내용", style="white", width=50)
        t.add_row("🟢 강세 (Bull)", bull[:120] + ("..." if len(bull) > 120 else ""))
        t.add_row("🔴 약세 (Bear)", bear[:120] + ("..." if len(bear) > 120 else ""))
        console.print(t)
    else:
        print(f"  Bull: {bull[:120]}")
        print(f"  Bear: {bear[:120]}")


def log_risk_assessment(risk: Dict[str, Any]) -> None:
    """Investment Decision Agent — CVaR/ATR/weight 출력"""
    cvar       = risk.get("cvar", 0)
    atr        = risk.get("atr", 0)
    atr_stop   = risk.get("atr_stop_pct", 0)
    risk_level = risk.get("risk_level", "NORMAL")
    entry      = risk.get("entry_decision", "HOLD")
    weight     = risk.get("recommended_weight", 0)

    level_color = {"HIGH": "red", "NORMAL": "green"}.get(risk_level.upper(), "yellow")

    if _rich_available():
        t = Table(title=f"[bold]⚖️ Risk Assessment — Level: [{level_color}]{risk_level}[/{level_color}][/bold]", expand=True)
        t.add_column("지표", style="dim", width=20)
        t.add_column("값", style="bold", width=25)
        t.add_column("판정", style="white", width=25)
        t.add_row("CVaR (95%)", f"{cvar:.4f}", "⚠️ HIGH" if risk_level == "HIGH" else "✅ OK")
        t.add_row("ATR Stop %", f"{atr_stop:.4f}", "⚠️ HIGH" if atr_stop > 0.05 else "✅ OK")
        t.add_row("진입 의사", entry, "⛔ REJECTED" if entry == "SKIP" else "✅ ALLOWED")
        t.add_row("권장 비중", f"{weight:.2%}", "")
        console.print(t)
    else:
        print(f"  Risk Level: {risk_level}, CVaR={cvar:.4f}, ATR%={atr_stop:.4f}, Entry={entry}, Weight={weight:.2%}")


def log_final_decision(action: int, weight: float, reasoning: str, risk: Dict[str, Any]) -> None:
    """Final Judgment — 최종 행동 + 매매근거 출력"""
    action_map = {0: "🟢 BUY", 1: "🟡 HOLD", 2: "🔴 SELL"}
    action_str = action_map.get(action, "❓")
    action_color = {0: "green", 1: "yellow", 2: "red"}.get(action, "white")

    if _rich_available():
        # Decision banner
        console.print(f"\n[bold {action_color}]{'═'*70}[/bold {action_color}]")
        console.print(f"[bold {action_color}]  FINAL DECISION: {action_str}  |  Weight: {weight:.0%}  |  Reasoning: {reasoning[:80]}...[/bold {action_color}]")
        console.print(f"[bold {action_color}]{'═'*70}[/bold {action_color}]\n")

        # Reasoning panel
        console.print(Panel(
            Text(reasoning, style="italic white"),
            title="[bold]💡 Final Reasoning (환각 검증 완료)[/bold]",
            border_style=action_color,
            expand=False,
        ))
    else:
        print(f"\n{'='*60}")
        print(f"  FINAL DECISION: {action_str} | Weight: {weight:.0%}")
        print(f"  Reasoning: {reasoning[:80]}")
        print(f"{'='*60}\n")


def log_episode_summary(
    episode_id: str,
    action: int,
    weight: float,
    risk_level: str,
    reports_keys: list,
    duration_sec: Optional[float] = None,
) -> None:
    """에피소드 종료 후 전체 요약 테이블"""
    action_map = {0: "🟢 BUY", 1: "🟡 HOLD", 2: "🔴 SELL"}
    action_str = action_map.get(action, "❓")

    dur_str = f"{duration_sec:.1f}s" if duration_sec else "N/A"

    if _rich_available():
        t = Table(title=f"[bold cyan]📋 Episode Summary: {episode_id}[/bold cyan]", expand=True)
        t.add_column("항목", style="dim", width=20)
        t.add_column("값", style="bold", width=50)
        t.add_row("결정", action_str)
        t.add_row("비중", f"{weight:.0%}")
        t.add_row("리스크", risk_level)
        t.add_row("Analysts 실행됨", ", ".join(reports_keys) if reports_keys else "없음")
        t.add_row("수행 시간", dur_str)
        console.print(t)
        console.print("[cyan]" + "━" * 70 + "[/cyan]")
    else:
        print(f"  Episode: {episode_id}, Action={action_str}, Weight={weight:.0%}, Risk={risk_level}, Duration={dur_str}")


def log_backtest_summary(results: Dict[str, Any]) -> None:
    """전체 백테스트 종료 후 종합 결과"""
    bt = results.get("backtest_result", {})
    actions = results.get("all_actions", [])

    if _rich_available():
        t = Table(title="[bold cyan]📈 Multi-Episode 백테스트 종합[/bold cyan]", expand=True)
        t.add_column("지표", style="dim", width=20)
        t.add_column("값", style="bold", width=30)
        t.add_row("최종 자산", f"${bt.get('final_asset', 0):,.0f}")
        t.add_row("모델 수익률", f"{bt.get('model_return_pct', 0):+.2f}%")
        t.add_row("Buy&Hold 대비", f"{bt.get('buyhold_return_pct', 0):+.2f}%")
        t.add_row("MDD", f"{bt.get('mdd_pct', 0):.2f}%")
        t.add_row("Sharpe Ratio", f"{bt.get('sharpe_ratio', 0):.2f}")
        t.add_row("총 거래 횟수", str(bt.get('total_trades', 0)))
        t.add_row("에피소드별 행동", str(actions))
        console.print(t)
    else:
        print(f"  Final: ${bt.get('final_asset', 0):,.0f}, Return: {bt.get('model_return_pct', 0):+.2f}%, MDD: {bt.get('mdd_pct', 0):.2f}%")