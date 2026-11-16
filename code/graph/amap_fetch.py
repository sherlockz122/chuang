# 环节一/04: 高德多边形搜索抓取 POI（属性补充源）
# 用法: 设置环境变量 AMAP_KEY 后运行; 断点续跑（每格一缓存文件，重跑自动跳过已完成格子）
# 输入:  ../data/graph_out/clean_poi.jsonl（用于匹配合并，可后置）
# 输出:  ../data/amap_poi/{category}/{grid}.json（GCJ-02 原始） + all_wgs84.jsonl（转换合并后）
import os, sys, json, time, math
sys.stdout.reconfigure(encoding='utf-8')
import requests
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
CFG = os.path.join(ROOT, 'config')
sys.path.insert(0, os.path.join(ROOT, 'graph'))
from coord_convert import gcj02_to_wgs84

city = yaml.safe_load(open(os.path.join(CFG, 'city.yaml'), encoding='utf-8'))
cats = yaml.safe_load(open(os.path.join(CFG, 'categories.yaml'), encoding='utf-8'))
KEY = os.environ.get('AMAP_KEY', '')
GRID = city['amap_grid_deg']
OUT = os.path.join(DATA, 'amap_poi')

API = 'https://restapi.amap.com/v3/place/polygon'


def grid_cells(bbox):
    minx, miny, maxx, maxy = bbox
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cells.append((round(x, 6), round(y, 6),
                          round(min(x + GRID, maxx), 6), round(min(y + GRID, maxy), 6)))
            y += GRID
        x += GRID
    return cells


def fetch_grid(cat_id, types, x1, y1, x2, y2):
    """一格的完整分页抓取，返回 (pois, truncated)"""
    pois, page, truncated = [], 1, False
    polygon = f'{x1},{y1}|{x2},{y2}'
    while page <= 40:  # 单格上限 40 页×25 条=1000 条
        resp = requests.get(API, params={
            'key': KEY, 'polygon': polygon, 'types': types,
            'offset': city['amap_page_size'], 'page': page, 'extensions': 'base',
        }, timeout=15).json()
        if resp.get('status') != '1':
            if resp.get('infocode') == '10009':
                print(f'  [配额超限] {cat_id} grid({x1},{y1}) 已抓 {len(pois)} 条，停止')
                truncated = True
                break
            if resp.get('infocode') == '10003':
                time.sleep(1.5)  # 并发超限重试
                continue
            raise RuntimeError(f'高德错误 {resp.get("infocode")}: {resp.get("info")} for {cat_id}')
        pois.extend(resp.get('pois', []))
        if len(resp.get('pois', [])) < city['amap_page_size'] or page >= 40:
            if page >= 40 and len(resp.get('pois', [])) == city['amap_page_size']:
                truncated = True  # 触顶截断，格子内 POI 密度超过 1000
            break
        page += 1
        time.sleep(0.2)  # 控制频率
    return pois, truncated


def main():
    if not KEY:
        print('请先设置环境变量 AMAP_KEY（高德 Web 服务 key，个人开发者即可）')
        return
    # 可选 AMAP_BBOX="minx,miny,maxx,maxy" 缩小抓取范围（如五环内），默认全域
    bbox = [float(v) for v in os.environ.get('AMAP_BBOX', '').split(',') if v] \
        or city['bbox_whole']
    cells = grid_cells(bbox)
    print(f'网格数: {len(cells)}（{GRID}° × {GRID}°，bbox={bbox}）')
    total, truncated_grids = 0, []
    for g in cats['groups']:
        for c in g['categories']:
            if not c.get('amap_code'):
                continue
            d = os.path.join(OUT, c['id'])
            os.makedirs(d, exist_ok=True)
            for (x1, y1, x2, y2) in cells:
                fp = os.path.join(d, f'{x1}_{y1}.json')
                if os.path.exists(fp):
                    continue
                pois, trunc = fetch_grid(c['id'], c['amap_code'], x1, y1, x2, y2)
                json.dump(pois, open(fp, 'w', encoding='utf-8'), ensure_ascii=False)
                total += len(pois)
                if trunc:
                    truncated_grids.append(f'{c["id"]}@{x1},{y1}')
            print(f'{c["name"]}({c["id"]}): 累计 {total} 条')
    print(f'完成。截断格子: {truncated_grids if truncated_grids else "无"}')

    # 合并 + GCJ-02→WGS-84
    out_rows = []
    for cid in os.listdir(OUT):
        d = os.path.join(OUT, cid)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            for p in json.load(open(os.path.join(d, fn), encoding='utf-8')):
                if not p.get('location'):
                    continue
                gl, gt = map(float, p['location'].split(','))
                wl, wt = gcj02_to_wgs84(gl, gt)
                out_rows.append({
                    'amap_id': p.get('id', ''), 'name': p.get('name', ''),
                    'category': cid, 'lng': round(wl, 6), 'lat': round(wt, 6),
                    'adcode': p.get('adcode', ''), 'address': p.get('address', ''),
                    'type_code': p.get('type', ''),
                })
    with open(os.path.join(OUT, 'all_wgs84.jsonl'), 'w', encoding='utf-8') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'WGS-84 合并输出: {len(out_rows)} 条 -> {os.path.join(OUT, "all_wgs84.jsonl")}')


if __name__ == '__main__':
    main()
