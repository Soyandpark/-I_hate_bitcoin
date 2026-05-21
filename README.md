# I_Hate_BitCoin

BTC/USDT 1시간봉 기반 3-class 분류 트레이딩 시스템 변경 사항

     agents/analyst_nodes.py의 Technical Analyst와 Macro Analyst가 각각 get_polygon_aggregates, get_polygon_news를 호출한다. 현재는 매 호출마다 live Polygon REST API를 호출하므로:
     - 백테스트 중 반복 API 호출 → 속도 느림 + API rate limit 위험
     - API 키 없을 때 실패 → 백테스트 불가

     해결: Pre-fetch로 datasets/에 과거 데이터를 저장하고, Tool은 로컬 파일优先으로 읽도록 변경.

     ---
     Implementation Plan

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

     2. tools/polygon_tools.py (수정)

     Tool 함수 내부에 Cache-First 전략 추가:

     Tool 호출
       └→ local cache 파일 존재? ─Yes→ 로컬 JSON 로드 → 마크다운 파싱 → 반환
                                └→ No → live API 호출 → 응답 JSON 저장 → 마크다운 파싱 → 반환

     구체 로직:
     def _get_aggregates_cache_path(from_date, to_date, ticker):
         return os.path.join("datasets", f"polygon_agg_{ticker}_{from_date}_{to_date}.json")

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

  변경 요약

  모든 에이전트/툴 모듈에 load_dotenv() 추가:

  agent_nodes, manager_nodes, polygon_tools, data_collector_polygon, run_trading_graph
    └→ from dotenv import load_dotenv → load_dotenv()
        └→ .env 읽어서 os.environ["POLYGON_API_KEY"], ["OPENAI_API_KEY"] 자동 주입

  테스트 결과 (실제 API 키로 live 호출):

  [AGG] [Polygon Aggregates] API 오류: 403 Client Error: Forbidden for url
  [NEWS] [Polygon News] API 오류: 404 Client Error: Not Found for url

  403은 Polygon API 키가 Free Tier라 Aggregates 접근 제한 → rate limit / plan 확인 필요.
  404는 /v1/news 엔드포인트가 실제 Polygon 문서와 다를 수 있음.

  ---
  Pre-fetch + Cache-First 흐름

  data_collector_polygon.py --start 2024-01-01 --end 2025-01-01
      └→ Polygon REST API 호출 → datasets/polygon_agg_*.json 저장

  백테스트 실행 시
      └→ get_polygon_aggregates.invoke()
          ├→ datasets/ 캐시 존재? → [CACHE HIT] → 마크다운 반환 (API 호출 0)
          └→ 캐시 없음 → live API 호출 → 결과 저장 → 마크다운 반환
