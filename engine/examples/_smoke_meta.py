import json
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

url = 'http://127.0.0.1:8765/api/candidates/search'
data = json.dumps({
    'dem_path': r'D:\Xunlong\data\langzhong_cop30.tif',
    'water_path': r'D:\Xunlong\data\langzhong_rivers_osm.geojson',
    'top_k': 1, 'min_score': 0,
}).encode()
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
r = urllib.request.urlopen(req, timeout=240)
obj = json.loads(r.read().decode('utf-8'))
c0 = obj['candidates'][0]
print('=== scores ===')
print(json.dumps(c0['scores'], ensure_ascii=False, indent=2))
print('=== meta ===')
print(json.dumps(c0.get('meta', {}), ensure_ascii=False, indent=2))
