"""
Polygon.io Pre-fetch Collector
백테스트 전 Polygon REST API에서 Aggregates + Financial News를 수집하여
datasets/ 폴더에 JSON으로 저장한다.

Usage:
  # 전체 기간 수집 (실제 API 키 필요)
  python data_collector_polygon.py --start 2024-01-01 --end 2025-01-01

  # Aggregates만 수집
  python data_collector_polygon.py --start 2024-01-01 --end 2025-01-01 --type agg

  # News만 수집
  python data_collector_polygon.py --start 2024-01-01 --end 2025-01-01 --type news

  # 자동 생성 (datasets/에서 가장 오래된 ~ 가장 최신 파일 자동 감지)
  python data_collector_polygon.py --auto
"""
import os, json, sys, time, argparse
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "datasets"
DATA_DIR.mkdir(exist_ok=True)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE_V2 = "https://api.polygon.io/v2"
POLYGON_BASE_V1 = "https://api.polygon.io/v1"
TICKER = "C:BTCUSD"

# ─────────────────────────────────────────────────────────────────────────────
# 1) Raw API Fetchers
# ─────────────────────────────────────────────────────────────────────────────

def _paginate_aggregates(
    ticker: str,
    from_date: str,
    to_date: str,
    multiplier: int = 1,
    timespan: str = "hour",
    batch_days: int = 30,
) -> List[Dict]:
    """
    Polygon Aggregates를 batch_days 단위로 쪼개서 Fetch.
    5000 limit을 우회하기 위해 기간을 분할한다.
    """
    if not POLYGON_API_KEY:
        print("[ERROR] POLYGON_API_KEY not set"); return []

    results_all: List[Dict] = []
    current = datetime.strptime(from_date, "%Y-%m-%d")
    end     = datetime.strptime(to_date, "%Y-%m-%d")

    while current < end:
        batch_end = min(current + timedelta(days=batch_days), end)
        url = (
            f"{POLYGON_BASE_V2}/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{timespan}/{current.strftime('%Y-%m-%d')}/{batch_end.strftime('%Y-%m-%d')}"
            f"?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            bars = data.get("results", [])
            results_all.extend(bars)
            print(f"  {current.strftime('%Y-%m-%d')} ~ {batch_end.strftime('%Y-%m-%d')}: "
                  f"{len(bars)} bars (total: {len(results_all)})")
            time.sleep(0.2)  # rate limit 방지
        except Exception as e:
            print(f"  [WARN] Batch fetch 실패: {e}")
        current = batch_end + timedelta(seconds=1)

    return results_all


def _paginate_news(
    ticker: str,
    from_date: str,
    to_date: str,
    limit_per_page: int = 50,
) -> List[Dict]:
    """
    Polygon News를 cursor 기반 페이지네이션으로 전체 수집.
    """
    if not POLYGON_API_KEY:
        print("[ERROR] POLYGON_API_KEY not set"); return []

    results_all: List[Dict] = []
    cursor = None
    max_pages = 20
    page = 0

    while page < max_pages:
        params = {
            "ticker": ticker,
            "limit": limit_per_page,
            "order": "asc",
            "sort": "published_utc",
            "apiKey": POLYGON_API_KEY,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(f"{POLYGON_BASE_V1}/news", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("results", [])
            if not items:
                break

            # 기간 필터 (from_date / to_date)
            filtered = [
                item for item in items
                if from_date <= (item.get("published_utc") or "")[:10] <= to_date
            ]
            results_all.extend(filtered)

            # 다음 페이지
            next_url = data.get("next_url") or data.get("next_cursor")
            if not next_url:
                break
            # cursor 추출: next_url에서 cursor 파라미터 추출
            import urllib.parse
            parsed = urllib.parse.urlparse(next_url)
            qs = urllib.parse.parse_qs(parsed.query)
            cursor = qs.get("cursor", [None])[0]
            if not cursor:
                break

            print(f"  page {page+1}: {len(items)} items, {len(filtered)} in range (total: {len(results_all)})")
            time.sleep(0.2)
            page += 1

        except Exception as e:
            print(f"  [WARN] News page {page+1} 실패: {e}")
            break

    return results_all


# ─────────────────────────────────────────────────────────────────────────────
# 2) Save / Load Cache
# ─────────────────────────────────────────────────────────────────────────────

def save_aggregates_cache(data: List[Dict], from_date: str, to_date: str) -> Path:
    """Aggregates JSON 캐시 저장"""
    fname = f"polygon_agg_{TICKER.replace(':', '_')}_{from_date}_{to_date}.json"
    path = DATA_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ticker": TICKER, "from_date": from_date, "to_date": to_date,
                   "count": len(data), "results": data}, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {path} ({len(data)} bars)")
    return path


def load_aggregates_cache(from_date: str, to_date: str) -> Optional[Dict]:
    """Aggregates 캐시 로드 (기간 매칭)"""
    fname = f"polygon_agg_{TICKER.replace(':', '_')}_{from_date}_{to_date}.json"
    path = DATA_DIR / fname
    if not path.exists():
        # 파싱 범위 축소: 더 넓은 범위 캐시에서 필터
        for p in DATA_DIR.glob("polygon_agg_*.json"):
            try:
                with open(p) as f:
                    cache = json.load(f)
                results = cache.get("results", [])
                filtered = [
                    b for b in results
                    if from_date <= _ts_to_date(b.get("t", 0)) <= to_date
                ]
                if filtered:
                    print(f"[Cache filter] {p.name}: {len(filtered)} bars in range")
                    cache["results"] = filtered
                    return cache
            except Exception:
                continue
        return None
    with open(path) as f:
        return json.load(f)


def save_news_cache(data: List[Dict], from_date: str, to_date: str) -> Path:
    """News JSON 캐시 저장"""
    fname = f"polygon_news_{TICKER.replace(':', '_')}_{from_date}_{to_date}.json"
    path = DATA_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ticker": TICKER, "from_date": from_date, "to_date": to_date,
                   "count": len(data), "results": data}, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {path} ({len(data)} articles)")
    return path


def load_news_cache(from_date: str, to_date: str) -> Optional[Dict]:
    """News 캐시 로드 (기간 매칭)"""
    fname = f"polygon_news_{TICKER.replace(':', '_')}_{from_date}_{to_date}.json"
    path = DATA_DIR / fname
    if not path.exists():
        for p in DATA_DIR.glob("polygon_news_*.json"):
            try:
                with open(p) as f:
                    cache = json.load(f)
                results = cache.get("results", [])
                filtered = [
                    item for item in results
                    if from_date <= (item.get("published_utc") or "")[:10] <= to_date
                ]
                if filtered:
                    print(f"[Cache filter] {p.name}: {len(filtered)} articles in range")
                    cache["results"] = filtered
                    return cache
            except Exception:
                continue
        return None
    with open(path) as f:
        return json.load(f)


def _ts_to_date(ts_ms: int) -> str:
    if not ts_ms: return "1970-01-01"
    return datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# 3) CLI Entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Polygon.io 데이터 Pre-fetch")
    parser.add_argument("--start", default="2024-01-01", help="시작일 YYYY-MM-DD")
    parser.add_argument("--end",   default="2025-01-01", help="종료일 YYYY-MM-DD")
    parser.add_argument("--type",  default="all",
                        choices=["all", "agg", "news"],
                        help="수집 타입: all / agg / news")
    parser.add_argument("--ticker", default=TICKER, help=f"Polygon 티커 (기본: {TICKER})")
    parser.add_argument("--days-per-batch", type=int, default=30,
                        help="Agg 배치 기간 (일, 기본 30)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Polygon Pre-fetch: {args.start} ~ {args.end}")
    print(f"  Ticker: {args.ticker}  Type: {args.type}")
    print(f"{'='*60}\n")

    if args.type in ("all", "agg"):
        print("[1/2] Fetching Aggregates...")
        bars = _paginate_aggregates(args.ticker, args.start, args.end, batch_days=args.days_per_batch)
        if bars:
            save_aggregates_cache(bars, args.start, args.end)
        else:
            print("[AGG] 데이터 없음 — datasets/에 이미 캐시가 있을 수 있음")

    if args.type in ("all", "news"):
        print("\n[2/2] Fetching News...")
        news_items = _paginate_news(args.ticker, args.start, args.end)
        if news_items:
            save_news_cache(news_items, args.start, args.end)
        else:
            print("[NEWS] 데이터 없음 — datasets/에 이미 캐시가 있을 수 있음")

    print(f"\n[Done] Pre-fetch 완료 — datasets/ 폴더 확인")
    print("  Tool은 자동으로 로컬 캐시를 우선으로 읽습니다.")


if __name__ == "__main__":
    main()