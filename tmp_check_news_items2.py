from dotenv import load_dotenv
import os, json
from data_collector_polygon import _paginate_news

load_dotenv()
news = _paginate_news('X:BTCUSD', '2026-05-24', '2026-05-24', limit_per_page=50)
output = {
    'loaded_key': bool(os.getenv('POLYGON_API_KEY')),
    'count': len(news),
    'titles': [item.get('title') for item in news[:5]],
    'dates': [(item.get('published_utc') or '')[:10] for item in news[:5]],
}
with open('tmp_check_news_items_result.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print('wrote result')
