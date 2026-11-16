# 环节五/04: 提取完整车行路网（供回放轨迹贴路）
# 原 clean_roads.jsonl 只是"POI 附近道路子集"（8675km），碎片化严重，贴路走不通。
# 本脚本从 BBBike Beijing pbf 提取全部车行道路，按 OSM 共享节点建立精确段邻接
# （路口坐标=共享节点坐标，无需端点匹配）。
# 输出: data/graph_out/roads_all.jsonl（段几何）+ roads_all_conn.json（{seg_id: [[邻居, x, y], ...]}）
import sys, os, json, math

import pyrosm
from shapely.geometry import LineString, MultiLineString

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
PBF = os.path.join(DATA, 'osm_beijing', 'Beijing.osm.pbf')
OUT_SEG = os.path.join(DATA, 'graph_out', 'roads_all.jsonl')
OUT_CONN = os.path.join(DATA, 'graph_out', 'roads_all_conn.json')

CAR_HIGHWAYS = ['motorway', 'motorway_link', 'trunk', 'trunk_link',
                'primary', 'primary_link', 'secondary', 'secondary_link',
                'tertiary', 'tertiary_link', 'residential', 'living_street',
                'unclassified', 'service']


def main():
    osm = pyrosm.OSM(PBF)
    gdf = osm.get_data_by_custom_criteria(
        custom_filter={'highway': CAR_HIGHWAYS}, filter_type='keep',
        tags_as_columns=['name', 'oneway', 'maxspeed'])
    print(f'ways: {len(gdf)}', flush=True)
    node_segs = {}  # (x, y) -> [seg_id, ...]
    n_seg = 0
    with open(OUT_SEG, 'w', encoding='utf-8') as fo:
        for row in gdf.itertuples():
            way_id = int(row.id)
            geom = row.geometry
            if isinstance(geom, LineString):
                lines = [geom]
            elif isinstance(geom, MultiLineString):
                lines = list(geom.geoms)
            else:
                continue  # 环形面（环岛/广场）跳过
            hw = row.highway
            name = getattr(row, 'name', '') or ''
            oneway = str(getattr(row, 'oneway', '') or '')
            for line in lines:
                coords = list(line.coords)
                for k in range(len(coords) - 1):
                    x1, y1 = coords[k]
                    x2, y2 = coords[k + 1]
                    if x1 == x2 and y1 == y2:
                        continue
                    seg_id = f'OSM_W_{way_id}_{k}'
                    length_m = math.hypot((x2 - x1) * math.cos(math.radians((y1 + y2) / 2)),
                                          y2 - y1) * 111320
                    rec = {'seg_id': seg_id, 'way_id': way_id, 'seg': k, 'highway': hw,
                           'name': name, 'oneway': oneway,
                           'x1': round(x1, 7), 'y1': round(y1, 7),
                           'x2': round(x2, 7), 'y2': round(y2, 7),
                           'length_m': round(length_m, 1),
                           'lng': round((x1 + x2) / 2, 7), 'lat': round((y1 + y2) / 2, 7)}
                    fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    k1 = (round(x1, 7), round(y1, 7))
                    k2 = (round(x2, 7), round(y2, 7))
                    node_segs.setdefault(k1, []).append(seg_id)
                    node_segs.setdefault(k2, []).append(seg_id)
                    n_seg += 1
    print(f'segments: {n_seg}', flush=True)
    # 邻接：共享节点的段两两相邻，路口坐标=共享节点
    conn = {}
    n_conn = 0
    for (x, y), segs in node_segs.items():
        if len(segs) < 2:
            continue
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                if a == b:
                    continue
                conn.setdefault(a, []).append([b, x, y])
                conn.setdefault(b, []).append([a, x, y])
                n_conn += 1
    json.dump(conn, open(OUT_CONN, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'conn pairs: {n_conn}, segs with conn: {len(conn)}', flush=True)
    # 连通分量统计
    seen = set()
    comps = []
    for s in conn:
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        c = 0
        while stack:
            cur = stack.pop()
            c += 1
            for n in conn[cur]:
                if n[0] not in seen:
                    seen.add(n[0])
                    stack.append(n[0])
        comps.append(c)
    comps.sort(reverse=True)
    print(f'components top5: {comps[:5]}, total: {len(comps)}')


if __name__ == '__main__':
    main()
