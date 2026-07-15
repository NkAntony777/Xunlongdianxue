"""验证 P2 字段（xuankong/phenology）在 API 中的输出。"""
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
print('=== geography (xuankong + phenology fields) ===')
geo = c0['geography']
xuan = {k: v for k, v in geo.items() if 'xuankong' in k or 'phenology' in k}
print(json.dumps(xuan, ensure_ascii=False, indent=2))
print('=== messages (xuankong + phenology) ===')
msgs = c0['messages']
print(json.dumps({k: msgs[k] for k in ('xuankong', 'phenology') if k in msgs}, ensure_ascii=False, indent=2))
