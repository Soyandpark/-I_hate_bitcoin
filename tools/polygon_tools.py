"""
Polygon.io REST API Tools -- Aggregates + Financial News
LangChain @tool로 제공하며, Raw JSON -> Pydantic -> 마크다운 리포트 전처리를 내부에서 수행.

Cache-First 전략:
  Tool 호출 -> 로컬 캐시 파일 존재? -> Yes: 로컬 JSON 로드 -> 마크다운 파싱 -> 반환
                                   -> No:  live API 호출 -> 응답 JSON 저장 -> 마크다운 파싱 -> 반환
  USE_LOCAL_CACHE=0 설정 시 live API만 사용 (기본: 1, 캐시 우선)
"""
import os
from dotenv import load_dotenv
load_dotenv()
import json
from datetime import datetime
from typing import Optional, List

import requests

from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────────────────────
# 0) Config
# ─────────────────────────────────────────────────────────────────────────────
POLYGON_API_KEY  = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE_V2  = "https://api.polygon.io/v2"
POLYGON_BASE_V1  = "https://api.polygon.io/v1"
TICKER           = "C:BTCUSD"
USE_LOCAL_CACHE  = os.getenv("USE_LOCAL_CACHE", "1") == "1"
DATA_DIR         = os.path.join(os.path.dirname(__file__), "..", "datasets")


# ─────────────────────────────────────────────────────────────────────────────
# 1) Pydantic Schemas -- Raw JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

class AggBar(BaseModel):
    t:  int    = Field(description="Unix timestamp ms")
    o:  float  = Field(description="Open")
    h:  float  = Field(description="High")
    l:  float  = Field(description="Low")
    c:  float  = Field(description="Close")
    v:  int    = Field(description="Volume")
    vw: float  = Field(default=0.0, description="VWAP")


class AggregatesResponse(BaseModel):
    ticker: str
    queryCount: int
    results: List[AggBar]


class NewsItem(BaseModel):
    title:        str = Field(description="제목")
    published_at: str = Field(description="발행 시각")
    publisher:    str = Field(description="발행 매체")
    author:     Optional[str] = Field(default=None, description="작성자")
    description:  str = Field(description="본문 요약")
    url:          str = Field(description="원문 링크")
    tickers:      List[str] = Field(default_factory=list, description="关联 티커")


class FinancialNewsResponse(BaseModel):
    results: List[NewsItem]


# ─────────────────────────────────────────────────────────────────────────────
# 2) Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path_agg(from_date: str, to_date: str) -> str:
    fname = f"polygon_agg_{TICKER.replace(':', '_')}_{from_date}_{to_date}.json"
    return os.path.join(DATA_DIR, fname)


def _cache_path_news(from_date: str, to_date: str) -> str:
    fname = f"polygon_news_{TICKER.replace(':', '_')}_{from_date}_{to_date}.json"
    return os.path.join(DATA_DIR, fname)


def _ts_to_date(ts_ms: int) -> str:
    if not ts_ms: return "1970-01-01"
    return datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")


def _load_cache(path: str) -> Optional[dict]:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _filter_cache_by_range(cache: dict, from_date: str, to_date: str) -> dict:
    """Wide-range cache -> filter to requested date window"""
    results = cache.get("results", [])
    if not results:
        return cache

    # Aggregates: t(ts_ms) 필터
    if isinstance(results[0], dict) and "t" in results[0]:
        filtered = [b for b in results if from_date <= _ts_to_date(b.get("t", 0)) <= to_date]
    # News: published_utc 필터
    else:
        filtered = [
            b for b in results
            if from_date <= (b.get("published_utc") or b.get("published_at", ""))[:10] <= to_date
        ]

    return {**cache, "results": filtered, "count": len(filtered)}


# ─────────────────────────────────────────────────────────────────────────────
# 3) Internal Parsers — Raw JSON → Markdown
# ─────────────────────────────────────────────────────────────────────────────

def _unix_to_iso(ms: int) -> str:
    return datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


def _parse_aggregates_to_md(data: dict) -> str:
    ticker  = data.get("ticker", "N/A")
    results = data.get("results", [])
    if not results:
        return "**[Polygon Aggregates]** 데이터 없음\n"

    recent = results[-24:]
    closes  = [r["c"] for r in recent]
    highs   = [r["h"] for r in recent]
    lows    = [r["l"] for r in recent]
    volumes = [r["v"] for r in recent]
    latest  = closes[-1]
    high24  = max(highs);  low24 = min(lows)
    vol_avg = sum(volumes) / len(volumes) if volumes else 0
    change  = (closes[-1] / closes[0] - 1) * 100 if closes[0] else 0
    sma20   = sum(closes[-20:]) / min(20, len(closes)) if len(closes) >= 20 else sum(closes) / len(closes)
    sma5    = sum(closes[-5:])  / min(5,  len(closes)) if len(closes) >= 5  else sum(closes) / len(closes)
    trend   = "▲ 상승" if change > 0.5 else ("▼ 하락" if change < -0.5 else "→ 보합")

    lines = [
        f"## [Polygon] Aggregates ({ticker} -- last 24h)",
        f"| Item         | Value          |",
        f"|--------------|----------------|",
        f"| Current Price| ${latest:,.2f}       |",
        f"| 24h High     | ${high24:,.2f}       |",
        f"| 24h Low      | ${low24:,.2f}       |",
        f"| 24h Change   | {change:+.2f}%       |",
        f"| 24h SMA20   | ${sma20:,.2f}       |",
        f"| 24h SMA5    | ${sma5:,.2f}       |",
        f"| Avg Volume  | {vol_avg/1e6:,.1f}M    |",
        f"| Trend        | {trend}       |",
        "",
        "**Last 6 Bars**",
        "| Time               | O        | H        | L        | C        | Vol(M)  |",
        "|--------------------|----------|----------|----------|----------|---------|",
    ]
    for bar in results[-6:]:
        ts  = _unix_to_iso(bar["t"])
        vol_m = bar["v"] / 1e6
        lines.append(
            f"| {ts} | `{bar['o']:>8,.0f}` | `{bar['h']:>8,.0f}` | "
            f"`{bar['l']:>8,.0f}` | `{bar['c']:>8,.0f}` | `{vol_m:>6.1f}` |"
        )
    return "\n".join(lines)


def _parse_news_to_md(data: dict, limit: int = 10) -> str:
    results  = data.get("results", [])
    articles = results[:limit]
    if not articles:
        return "**[Polygon News]** 데이터 없음\n"

    lines = [f"## [Polygon] Financial News ({ticker} -- latest {len(articles)} articles)", ""]
    for i, art in enumerate(articles, 1):
        title  = art.get("title", "N/A")
        pub_at = art.get("published_utc", art.get("published_at", "N/A"))[:10]
        pub_by = (art.get("publisher", {}) or {}).get("name", art.get("publisher", "N/A"))
        desc   = art.get("description", "")[:120]
        url    = art.get("article_url", art.get("url", ""))
        combined = (title + desc).lower()
        tag = "[NEUTRAL]"
        if any(w in combined for w in ["etf", "approval", "bullish", "surge", "all-time high", "record"]):
            tag = "[BULL]"
        elif any(w in combined for w in ["regulation", "ban", "sell-off", "crash", "bearish"]):
            tag = "[BEAR]"
        lines.append(
            f"### {i}. {tag} {title}\n"
            f"- Source: {pub_by}  |  Date: {pub_at}\n"
            f"- {desc}...\n"
            f"- Link: {url}\n"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4) LangChain @tool 들 — Cache-First
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_polygon_aggregates(
    ticker: str = "C:BTCUSD",
    from_date: str = "2024-01-01",
    to_date: str = "2025-01-01",
    multiplier: int = 1,
    timespan: str = "hour",
) -> str:
    """
    Polygon.io Aggregates API — 1시간봉(OHLCV) 조회 + 마크다운 변환.

    Cache-First: USE_LOCAL_CACHE=1(기본) → 로컬 캐시 datasets/ 우선.
    캐시 없으면 live API 호출 후 결과 저장.
    """
    cache_path = _cache_path_agg(from_date, to_date)

    # ── 1) Cache-First ────────────────────────────────────────────────────────
    if USE_LOCAL_CACHE:
        cache = _load_cache(cache_path)
        if cache:
            cache = _filter_cache_by_range(cache, from_date, to_date)
            if cache.get("results"):
                return f"[CACHE HIT]\n{_parse_aggregates_to_md(cache)}"

        # 동일 기간 캐시 없음 → 상위 폴더의 넓은 범위 캐시 탐색
        for p in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else []:
            if not p.startswith("polygon_agg_") or p.endswith(os.path.basename(cache_path)):
                continue
            try:
                wider = _load_cache(os.path.join(DATA_DIR, p))
                if wider and wider.get("results"):
                    filtered = _filter_cache_by_range(wider, from_date, to_date)
                    if filtered.get("results"):
                        return f"[CACHE HIT — filtered from {p}]\n{_parse_aggregates_to_md(filtered)}"
            except Exception:
                continue

    # ── 2) Live API fallback ────────────────────────────────────────────────
    if not POLYGON_API_KEY:
        return "[Polygon Aggregates] API key missing — set POLYGON_API_KEY env or run pre-fetch"

    url = (
        f"{POLYGON_BASE_V2}/aggs/ticker/{ticker}/range/"
        f"{multiplier}/{timespan}/{from_date}/{to_date}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        try:
            validated = AggregatesResponse.model_validate(raw)
            data = validated.model_dump()
        except Exception:
            data = raw

        # ── 3) Cache 저장 ───────────────────────────────────────────────────
        if USE_LOCAL_CACHE and data.get("results"):
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass  # 저장 실패해도 마크다운은 반환

        return _parse_aggregates_to_md(data)

    except requests.RequestException as e:
        return f"[Polygon Aggregates] API 오류: {e}"


@tool
def get_polygon_news(
    ticker: str = "C:BTCUSD",
    from_date: str = "2024-01-01",
    to_date: str = "2025-01-01",
    limit: int = 10,
    order: str = "desc",
) -> str:
    """
    Polygon.io Financial News API — BTC/USD 관련 뉴스 + 마크다운 변환.

    Cache-First: USE_LOCAL_CACHE=1(기본) → 로컬 캐시 datasets/ 우선.
    """
    cache_path = _cache_path_news(from_date, to_date)

    # ── 1) Cache-First ────────────────────────────────────────────────────────
    if USE_LOCAL_CACHE:
        cache = _load_cache(cache_path)
        if cache:
            cache = _filter_cache_by_range(cache, from_date, to_date)
            if cache.get("results"):
                return f"[CACHE HIT]\n{_parse_news_to_md(cache, limit=limit)}"

        for p in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else []:
            if not p.startswith("polygon_news_"):
                continue
            try:
                wider = _load_cache(os.path.join(DATA_DIR, p))
                if wider and wider.get("results"):
                    filtered = _filter_cache_by_range(wider, from_date, to_date)
                    if filtered.get("results"):
                        return f"[CACHE HIT — filtered from {p}]\n{_parse_news_to_md(filtered, limit=limit)}"
            except Exception:
                continue

    # ── 2) Live API fallback ────────────────────────────────────────────────
    if not POLYGON_API_KEY:
        return "[Polygon News] API key missing — set POLYGON_API_KEY env or run pre-fetch"

    params = {
        "ticker": ticker, "limit": min(limit, 50),
        "order": order, "sort": "published_utc", "apiKey": POLYGON_API_KEY,
    }
    try:
        resp = requests.get(f"{POLYGON_BASE_V1}/news", params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        try:
            validated = FinancialNewsResponse.model_validate(raw)
            data = validated.model_dump()
        except Exception:
            data = raw

        if USE_LOCAL_CACHE and data.get("results"):
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return _parse_news_to_md(data, limit=limit)

    except requests.RequestException as e:
        return f"[Polygon News] API 오류: {e}"