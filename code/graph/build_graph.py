# 环节一/03: 构图——节点(POI/道路/区域) + 五类边
# 输入:  ../data/graph_out/clean_poi.jsonl, clean_roads.jsonl, regions.json
# 输出:  ../data/graph_out/nodes.json, edges.json, graph.pkl + 统计
import sys, os, json, pickle
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import shapely.geometry as sg
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'graph_out')
CFG = os.path.join(ROOT, 'config')
city = yaml.safe_load(open(os.path.join(CFG, 'city.yaml'), encoding='utf-8'))
cats = yaml.safe_load(open(os.path.join(CFG, 'categories.yaml'), encoding='utf-8'))
POI_ROAD_M = city['poi_road_dist_m'] / 111320.0
POI_POI_R = city['poi_poi_radius_m'] / 111320.0
POI_POI_K = city['poi_poi_k']
DEG = 111320.0  # 每度经度约 111.32km（北京纬度附近近似，构图邻接判定够用）

pair_to_strength = {}
for a, b in cats['complement_pairs']:
    pair_to_strength[(a, b)] = 1.0
    pair_to_strength[(b, a)] = 1.0


def main():
    pois = [json.loads(l) for l in open(os.path.join(OUT, 'clean_poi.jsonl'), encoding='utf-8')]
    roads = [json.loads(l) for l in open(os.path.join(OUT, 'clean_roads.jsonl'), encoding='utf-8')]
    regions = json.load(open(os.path.join(OUT, 'regions.json'), encoding='utf-8'))

    G = nx.MultiGraph()
    nodes_out, edges_out = [], []
    eid = 0

    def add_edge(src, dst, etype, props):
        nonlocal eid
        G.add_edge(src, dst, key=eid, etype=etype, **props)
        edges_out.append({'edge_id': f'e_{eid:07d}', 'src': src, 'dst': dst,
                          'etype': etype, 'props': props})
        eid += 1

    # ---------- 节点 ----------
    poi_coords = np.array([[p['lng'], p['lat']] for p in pois])
    for p in pois:
        nid = f"OSM_N_{p['osm_id']}"
        G.add_node(nid, ntype='poi', name=p['name'], lng=p['lng'], lat=p['lat'],
                   category=p['category'], group=p['group'], tags=p.get('tags', {}))
        nodes_out.append({'node_id': nid, 'type': 'poi', 'name': p['name'],
                          'lng': p['lng'], 'lat': p['lat'], 'category': p['category'],
                          'group': p['group']})

    road_ids = []  # (node_id, x1,y1,x2,y2, lng,lat)
    for r in roads:
        nid = f"OSM_W_{r['way_id']}_{r['seg']}"
        G.add_node(nid, ntype='road', name=r['name'], highway=r['highway'],
                   lng=r['lng'], lat=r['lat'], length_m=r['length_m'], oneway=r['oneway'])
        nodes_out.append({'node_id': nid, 'type': 'road', 'name': r['name'],
                          'lng': r['lng'], 'lat': r['lat'], 'road_class': r['highway']})
        road_ids.append((nid, r['x1'], r['y1'], r['x2'], r['y2'], r['lng'], r['lat']))

    reg_polys = []
    for reg in regions:
        nid = f"REG_{reg['adcode']}"
        geom = sg.shape(reg['geometry'])
        G.add_node(nid, ntype='region', name=reg['name'],
                   lng=reg['center'][0], lat=reg['center'][1])
        nodes_out.append({'node_id': nid, 'type': 'region', 'name': reg['name'],
                          'lng': reg['center'][0], 'lat': reg['center'][1],
                          'adcode': reg['adcode']})
        reg_polys.append((nid, geom))

    print(f'节点: POI {len(pois)} / 道路段 {len(road_ids)} / 区域 {len(reg_polys)}')

    # ---------- 边1: poi-road 空间邻接（点到线段精确距离 ≤50m） ----------
    # 先按道路段中点 KDTree 粗筛 150m 候选，再算点到线段的 shapely 距离（避免中点近似漏掉长路段）
    road_mid = np.array([[r[5], r[6]] for r in road_ids])
    tree_road = cKDTree(road_mid)
    seg_lines = [sg.LineString([(x1, y1), (x2, y2)]) for (_, x1, y1, x2, y2, _, _) in road_ids]
    n1 = 0
    for i, p in enumerate(pois):
        pt = sg.Point(poi_coords[i][0], poi_coords[i][1])
        for j in tree_road.query_ball_point(poi_coords[i], 150.0 / DEG):
            d = pt.distance(seg_lines[j]) * DEG
            if d <= 50:
                add_edge(f"OSM_N_{p['osm_id']}", road_ids[j][0], 'SPATIAL_NEAR',
                         {'subtype': 'poi-road', 'distance_m': round(float(d), 1)})
                n1 += 1
    print(f'边 poi-road: {n1}')

    # ---------- 边2: poi-poi 空间邻接（r=500m, k≤20） ----------
    tree_poi = cKDTree(poi_coords)
    n2 = 0
    for i in range(len(pois)):
        dists, idxs = tree_poi.query(poi_coords[i], k=min(POI_POI_K + 1, len(pois)))
        if np.isscalar(idxs):
            continue
        for d, j in zip(dists, idxs):
            if j == i:
                continue
            if d > POI_POI_R:
                continue
            a, b = pois[i]['osm_id'], pois[j]['osm_id']
            if a >= b:
                continue  # 无向，只存一次
            add_edge(f'OSM_N_{a}', f'OSM_N_{b}', 'SPATIAL_NEAR',
                     {'subtype': 'poi-poi', 'distance_m': round(d * DEG, 1)})
            n2 += 1
    print(f'边 poi-poi: {n2}')

    # ---------- 边3: poi-region 点在区内 ----------
    n3 = 0
    for p in pois:
        pt = sg.Point(p['lng'], p['lat'])
        for rid, poly in reg_polys:
            if poly.contains(pt):
                add_edge(f"OSM_N_{p['osm_id']}", rid, 'IN_REGION', {})
                n3 += 1
                break
    print(f'边 poi-region: {n3}')

    # ---------- 边4: road-road 共享交叉口（端点 10m 内视为同一交叉口） ----------
    ends = []
    for k, (nid, x1, y1, x2, y2, lng, lat) in enumerate(road_ids):
        ends.append((nid, 's', x1, y1))
        ends.append((nid, 'e', x2, y2))
    end_coords = np.array([[e[2], e[3]] for e in ends])
    tree_end = cKDTree(end_coords)
    seen, n4 = set(), 0
    for i in range(len(ends)):
        near = tree_end.query_ball_point(end_coords[i], 10 / DEG)
        for j in near:
            if j <= i:
                continue
            a, b = ends[i][0], ends[j][0]
            if a == b:
                continue
            if (a, b) in seen or (b, a) in seen:
                continue
            seen.add((a, b))
            add_edge(a, b, 'ROAD_CONN', {})
            n4 += 1
    print(f'边 road-road: {n4}')

    # ---------- 边5: FUNC_COMP 功能互补（先验表，同类对 r=500m） ----------
    by_cat = {}
    for i, p in enumerate(pois):
        by_cat.setdefault(p['category'], []).append(i)
    n5 = 0
    for (ca, cb), strength in pair_to_strength.items():
        if ca not in by_cat or cb not in by_cat:
            continue
        for i in by_cat[ca]:
            dists, idxs = tree_poi.query(poi_coords[i], k=min(POI_POI_K + 1, len(pois)))
            if np.isscalar(idxs):
                continue
            for d, j in zip(dists, idxs):
                if j == i or pois[j]['category'] != cb or d > POI_POI_R:
                    continue
                a, b = pois[i]['osm_id'], pois[j]['osm_id']
                if a >= b:
                    continue
                add_edge(f'OSM_N_{a}', f'OSM_N_{b}', 'FUNC_COMP',
                         {'pair': f'{ca}-{cb}', 'strength': strength})
                n5 += 1
    print(f'边 FUNC_COMP: {n5}')

    # ---------- 保存 ----------
    json.dump(nodes_out, open(os.path.join(OUT, 'nodes.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    with open(os.path.join(OUT, 'edges.json'), 'w', encoding='utf-8') as f:
        for e in edges_out:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    with open(os.path.join(OUT, 'graph.pkl'), 'wb') as f:
        pickle.dump(G, f)  # networkx 3.4 已移除 gpickle，用标准 pickle 存内部图对象
    stats = {'nodes': len(nodes_out), 'edges': len(edges_out),
             'by_etype': {}, 'poi_by_cat': {}}
    for e in edges_out:
        stats['by_etype'][e['etype']] = stats['by_etype'].get(e['etype'], 0) + 1
    for n in nodes_out:
        if n['type'] == 'poi':
            stats['poi_by_cat'][n['category']] = stats['poi_by_cat'].get(n['category'], 0) + 1
    json.dump(stats, open(os.path.join(OUT, 'build_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('统计:', json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()
