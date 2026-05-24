from dotenv import load_dotenv
import os
from data_collector_polygon import _paginate_news

load_dotenv()
print('POLYGON_API_KEY loaded:', bool(os.getenv('POLYGON_API_KEY')))
news = _paginate_news('X:BTCUSD', '2026-05-24', '2026-05-24', limit_per_page=50)
print('news count =', len(news))
print('sample titles =', [item.get('title') for item in news[:3]])
