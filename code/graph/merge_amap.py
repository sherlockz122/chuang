# 环节一/06: 高德属性合并——amap POI 与 OSM POI 按名称+距离对齐（需先跑 amap_fetch.py）
# 输入:  ../data/amap_poi/all_wgs84.jsonl + ../data/graph_out/clean_poi.jsonl
# 输出:  ../data/aligned/poi_amap_attrs.json（osm_id -> 高德属性）+ merge_stats.json
# 口径: 同名（标准化后）且 ≤50m 才对齐；OSM 为主键，高德仅补充地址/编码等属性
import sys, os, json, re
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'aligned')
os.makedirs(OUT, exist_ok=True)


def norm_name(name):
    n = re.sub(r'[（(].*?[)）]', '', name or '')
    return re.sub(r'\s+', '', n).lower()


def main():
    src = os.path.join(DATA, 'amap_poi', 'all_wgs84.jsonl')
    if not os.path.exists(src):
        print('未找到高德数据，请先设置 AMAP_KEY 并运行 amap_fetch.py')
        return
    amap = [json.loads(l) for l in open(src, encoding='utf-8')]
    # 高德格子抓取会在格边界重复，按 amap_id 去重
    seen = set()
    uniq = []
    for a in amap:
        if a['amap_id'] in seen:
            continue
        seen.add(a['amap_id'])
        uniq.append(a)
    amap = uniq
    osm = [json.loads(l) for l in open(os.path.join(DATA, 'graph_out', 'clean_poi.jsonl'),
                                       encoding='utf-8')]
    amap_coords = np.array([[a['lng'], a['lat']] for a in amap])
    tree = cKDTree(amap_coords)
    attrs, n_name, n_fallback = {}, 0, 0

    def assign(pid, j, match_type):
        attrs[pid] = {
            'amap_id': amap[j]['amap_id'], 'amap_name': amap[j]['name'],
            'address': amap[j].get('address', ''), 'type_code': amap[j].get('type_code', ''),
            'adcode': amap[j].get('adcode', ''), 'match_type': match_type,
        }

    # 一档: 同类别 + 标准化名相等或相互包含（≥2字） + ≤100m
    for p in osm:
        nn = tree.query_ball_point([p['lng'], p['lat']], 100.0 / 111320.0)
        pn = norm_name(p['name'])
        if not pn:
            continue
        for j in nn:
            if amap[j]['category'] != p['category']:
                continue
            an = norm_name(amap[j]['name'])
            if not (an == pn or (len(pn) >= 2 and len(an) >= 2 and (pn in an or an in pn))):
                continue
            assign(p['osm_id'], j, 'name')
            n_name += 1
            break
    # 二档: OSM 无名 → 同类 ≤50m 最近（属性补充口径：地址/类型编码同楼级成立；有名未配上的不兜底，避免身份错配）
    for p in osm:
        if p['osm_id'] in attrs or p['name']:
            continue
        nn = tree.query_ball_point([p['lng'], p['lat']], 50.0 / 111320.0)
        cands = [j for j in nn if amap[j]['category'] == p['category']]
        if not cands:
            continue
        j = min(cands, key=lambda j: (amap_coords[j][0] - p['lng']) ** 2
                + (amap_coords[j][1] - p['lat']) ** 2)
        assign(p['osm_id'], j, 'category_fallback')
        n_fallback += 1

    json.dump(attrs, open(os.path.join(OUT, 'poi_amap_attrs.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump({'osm_poi': len(osm), 'amap_poi': len(amap),
               'matched_name': n_name, 'matched_fallback': n_fallback,
               'matched_total': n_name + n_fallback},
              open(os.path.join(OUT, 'merge_stats.json'), 'w', encoding='utf-8'), indent=1)
    # 人工核验样例（每类 3 条）
    samples = {}
    for pid, a in attrs.items():
        p = next(x for x in osm if x['osm_id'] == pid)
        cat = p['category']
        samples.setdefault(cat, []).append((p['name'], a['amap_name'], a['match_type']))
    samples = {k: v[:3] for k, v in samples.items()}
    json.dump(samples, open(os.path.join(OUT, 'merge_samples.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'OSM POI {len(osm)} / 高德 {len(amap)}，对齐 {n_name + n_fallback} 条'
          f'（同名 {n_name} + 同类兜底 {n_fallback}）-> poi_amap_attrs.json')


if __name__ == '__main__':
    main()
