"""冒烟测试：调用 /api/candidates/search 验证 P0-1~P0-4 + P1-1/P1-2 集成。"""
import json
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

url = 'http://127.0.0.1:8765/api/candidates/search'
data = json.dumps({
    'dem_path': r'D:\Xunlong\data\langzhong_cop30.tif',
    'water_path': r'D:\Xunlong\data\langzhong_rivers_osm.geojson',
    'top_k': 3, 'min_score': 0,
}).encode()
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
r = urllib.request.urlopen(req, timeout=240)
obj = json.loads(r.read().decode('utf-8'))
cands = obj['candidates']
print('count:', len(cands))
for i, c in enumerate(cands):
    overall = c['overall_score']
    print('--- candidate', i + 1, 'overall=', overall, '---')
    print('scores:', json.dumps(c['scores'], ensure_ascii=False))
    geo = c['geography']
    print('compass_shan:', geo.get('compass_shan'),
          'dev_deg:', geo.get('compass_dev_deg'),
          'chu:', geo.get('compass_chu_gua'),
          'jian:', geo.get('compass_jian_xiang'))
    print('mouth_lock:', geo.get('water_mouth_lock_ratio'))
    msgs = c['messages']
    print('msg compass:', msgs.get('compass', ''))
    print('msg mouth  :', msgs.get('mouth', ''))
