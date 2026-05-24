import os
from dotenv import load_dotenv
import requests

load_dotenv()
key = os.getenv('POLYGON_API_KEY')
print('POLYGON_API_KEY loaded:', bool(key))

for params in [
    {'ticker': 'X:BTCUSD', 'limit': 5, 'apiKey': key},
    {'ticker': 'X:BTCUSD', 'limit': 50, 'apiKey': key},
    {'limit': 5, 'apiKey': key},
]:
    print('\nREQUEST PARAMS:', params)
    resp = requests.get('https://api.polygon.io/v2/reference/news', params=params, timeout=30)
    print('STATUS CODE:', resp.status_code)
    try:
        data = resp.json()
        print('RESPONSE TYPE:', type(data).__name__)
        if isinstance(data, dict):
            print('KEYS:', list(data.keys()))
            print('RESULTS LENGTH:', len(data.get('results', [])))
            if data.get('results'):
                print('FIRST PUBLISHED:', data['results'][0].get('published_utc'))
                print('FIRST TITLE:', data['results'][0].get('title'))
    except Exception as e:
        print('JSON PARSE ERROR:', e)
        print('RAW TEXT:', resp.text[:400])
