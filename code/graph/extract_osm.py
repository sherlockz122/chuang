# 环节一/01: 解析 BBBike 北京 pbf（pyrosm，保留完整 OSM tags），产出原始 POI 与道路记录（WGS-84）
# 说明: BBBike 的 shp 包是简化 schema（仅 osm_id/name/type），缺 amenity/shop 等细粒度标签，
#       改用 pbf 全量 tags 与 categories.yaml 的 key=value 模式匹配。
# 输入:  ../data/osm_beijing/Beijing.osm.pbf
# 输出:  ../data/graph_out/raw_poi.jsonl, raw_roads.jsonl, regions.json
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import yaml
import numpy as np
import geopandas as gpd
from pyrosm import OSM
import shapely.geometry as sg
from shapely.geometry import Point, MultiPoint, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
DATA = os.path.join(os.path.dirname(ROOT), 'data')
CFG = os.path.join(ROOT, 'config')
OUT = os.path.join(DATA, 'graph_out')
os.makedirs(OUT, exist_ok=True)

city = yaml.safe_load(open(os.path.join(CFG, 'city.yaml'), encoding='utf-8'))
cats = yaml.safe_load(open(os.path.join(CFG, 'categories.yaml'), encoding='utf-8'))
minx, miny, maxx, maxy = city['bbox_v1']


def match_category(tags: dict):
    """返回 (category_id, group_id)，无匹配返回 None"""
    for g in cats['groups']:
        for c in g['categories']:
            for pat in c['osm']:
                if '=' not in pat:
                    continue
                k, v = pat.split('=', 1)
                if v == '*' and k in tags:
                    return c['id'], g['id']
                if tags.get(k) == v:
                    return c['id'], g['id']
    return None


def build_criterion():
    """从 categories.yaml 的 osm 模式构造 pyrosm 过滤条件 + 展开 key 列表。
    name 只加入列展开（tags_as_columns），不加入过滤条件（否则无名 POI 会被滤掉）"""
    crit, keys = {}, set()
    for g in cats['groups']:
        for c in g['categories']:
            for pat in c['osm']:
                k, v = pat.split('=', 1)
                keys.add(k)
                if v == '*':
                    crit[k] = True
                else:
                    crit.setdefault(k, []).append(v)
    keys.add('name')
    return crit, sorted(keys)


def main():
    src = os.path.join(DATA, 'osm_beijing', 'Beijing.osm.pbf')
    osm = OSM(src)

    # ---------- 道路（可驾驶网络，LineString） ----------
    keep_classes = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
                    'residential', 'unclassified', 'motorway_link', 'trunk_link',
                    'primary_link', 'secondary_link', 'tertiary_link'}
    roads = osm.get_network(network_type='driving')
    roads = roads[roads['highway'].isin(keep_classes)].copy()
    roads = roads.cx[minx:maxx, miny:maxy]
    n_road = 0
    with open(os.path.join(OUT, 'raw_roads.jsonl'), 'w', encoding='utf-8') as f:
        for _, r in roads.iterrows():
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue
            mid = geom.interpolate(0.5, normalized=True)
            p0, p1 = geom.coords[0], geom.coords[-1]
            nm = r.get('name')
            rec = {
                'way_id': str(int(r['id'])),
                'seg': 0,
                'name': nm if isinstance(nm, str) else '',
                'highway': r['highway'],
                'length_m': round(geom.length * 111320, 1),  # 粗略换算，clean 阶段 geodesic 重算
                'lng': round(mid.x, 6), 'lat': round(mid.y, 6),
                'x1': round(p0[0], 6), 'y1': round(p0[1], 6),
                'x2': round(p1[0], 6), 'y2': round(p1[1], 6),
                'oneway': ('yes' if r.get('oneway') is True else ''),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_road += 1
    print(f'道路记录: {n_road}')

    # ---------- POI（按 categories.yaml 的 tag 条件抓取，key 展开为列后匹配） ----------
    crit, keys = build_criterion()
    pois = osm.get_data_by_custom_criteria(crit, filter_type='keep', tags_as_columns=keys)
    pois = pois.cx[minx:maxx, miny:maxy]
    n_poi = 0
    stat = {}
    with open(os.path.join(OUT, 'raw_poi.jsonl'), 'w', encoding='utf-8') as f:
        for _, p in pois.iterrows():
            tags = {}
            for k in keys:
                v = p.get(k)
                if isinstance(v, str):  # NaN/None 跳过
                    tags[k] = v
            m = match_category(tags)
            if m is None:
                continue
            cid, gid = m
            geom = p.geometry
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, (Polygon, sg.MultiPolygon)):  # 面状 POI（商场/景区等）取质心
                g = geom.centroid
            elif isinstance(geom, sg.LineString):  # 线状（如地铁站入口沿线）取中点
                g = geom.interpolate(0.5, normalized=True)
            else:
                g = geom
            if isinstance(g, MultiPoint):
                g = g.geoms[0]
            if not isinstance(g, Point):
                continue
            nm = p.get('name')
            rec = {
                'osm_id': str(int(p['id'])),
                'name': nm if isinstance(nm, str) else '',
                'category': cid, 'group': gid,
                'lng': round(g.x, 6), 'lat': round(g.y, 6),
                'tags': tags,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_poi += 1
            stat[cid] = stat.get(cid, 0) + 1
    print(f'POI 记录: {n_poi}')
    print('分类统计:', json.dumps(stat, ensure_ascii=False))

    # ---------- 区域（北京行政区划，REG 节点） ----------
    admin = json.load(open(os.path.join(DATA, 'osm_beijing', 'beijing_admin.json'), encoding='utf-8'))
    bbox_poly = sg.box(minx, miny, maxx, maxy)
    regions = []
    for feat in admin['features']:
        props = feat['properties']
        geom = sg.shape(feat['geometry'])
        if geom.intersects(bbox_poly):
            regions.append({
                'adcode': str(props['adcode']),
                'name': props['name'],
                'center': props['centroid'],
                'geometry': json.loads(json.dumps(sg.mapping(geom))),
            })
    json.dump(regions, open(os.path.join(OUT, 'regions.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'区域节点: {len(regions)} -> {[r["name"] for r in regions]}')


if __name__ == '__main__':
    main()
