# 环节三/01: 停留事件 → 语义图节点 ID 强制对齐（地图匹配口径 + 区域归属）
# 输入:  ../data/tdrive/stay_events.jsonl + ../data/graph_out/nodes.json + regions.json
# 输出:  ../data/aligned/stay_node_pairs.jsonl（每个停留事件的最近 POI/道路/区域节点 + stay_type）
# 指标:  对齐覆盖率 = 有 POI 节点（≤阈值）的停留 / (五环内停留 - 等待类停留)
#        等待类(waiting) = 停在道路≤50m 的停留（等客/堵车，停留提取假阳性，不期待 POI 对齐）
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import yaml
from scipy.spatial import cKDTree
import shapely.geometry as sg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'aligned')
os.makedirs(OUT, exist_ok=True)
CFG = os.path.join(ROOT, 'config')
traj = yaml.safe_load(open(os.path.join(CFG, 'traj.yaml'), encoding='utf-8'))
city = yaml.safe_load(open(os.path.join(CFG, 'city.yaml'), encoding='utf-8'))

POI_ATTACH_M = float(traj['align']['poi_attach_m'])
ROAD_ATTACH_M = float(traj['align']['road_attach_m'])
BBOX_V1 = tuple(city['bbox_v1'])  # POI 对齐只在五环内做
DEG = 111320.0
LATF = 0.813


def main():
    nodes = json.load(open(os.path.join(DATA, 'graph_out', 'nodes.json'), encoding='utf-8'))
    regions = json.load(open(os.path.join(DATA, 'graph_out', 'regions.json'), encoding='utf-8'))
    stays = [json.loads(l) for l in open(os.path.join(DATA, 'tdrive', 'stay_events.jsonl'),
                                          encoding='utf-8')]

    poi_nodes = [(n['node_id'], n['lng'], n['lat']) for n in nodes if n['type'] == 'poi']
    poi_tree = cKDTree(np.array([[n[1], n[2]] for n in poi_nodes]))
    reg_polys = [(r['adcode'], sg.shape(r['geometry'])) for r in regions]
    # 道路: 用 clean_roads 的线段做"点到线段"精确距离（与语义图 poi-road 边同口径）
    roads = [json.loads(l) for l in open(os.path.join(DATA, 'graph_out', 'clean_roads.jsonl'),
                                          encoding='utf-8')]
    road_ids = [f"OSM_W_{r['way_id']}_{r['seg']}" for r in roads]
    seg_lines = [sg.LineString([(r['x1'], r['y1']), (r['x2'], r['y2'])]) for r in roads]
    road_tree = cKDTree(np.array([[r['lng'], r['lat']] for r in roads]))

    n_matched, out = 0, []
    coords = np.array([[s['lng'], s['lat']] for s in stays])

    # 最近 POI（含距离）
    pd, pi = poi_tree.query(coords, k=1)

    for i, s in enumerate(stays):
        poi_d_m = round(float(pd[i]) * DEG, 1)
        # 最近道路段（中点粗筛 150m → 点到线段精确距离）
        road_d_m, road_id = float('inf'), None
        pt = sg.Point(s['lng'], s['lat'])
        for j in road_tree.query_ball_point(coords[i], 150.0 / DEG):
            d = pt.distance(seg_lines[j]) * DEG
            if d < road_d_m:
                road_d_m, road_id = d, road_ids[j]
        road_d_m = round(road_d_m, 1)
        in_v1 = (BBOX_V1[0] <= s['lng'] <= BBOX_V1[2]
                 and BBOX_V1[1] <= s['lat'] <= BBOX_V1[3])
        poi_id = poi_nodes[pi[i]][0] if (in_v1 and poi_d_m <= POI_ATTACH_M) else None
        if road_d_m > ROAD_ATTACH_M:
            road_id = None
        region_id = None
        for adcode, poly in reg_polys:
            if poly.contains(pt):
                region_id = f'REG_{adcode}'
                break
        if poi_id:
            n_matched += 1
        if in_v1 and not poi_id and road_id is not None:
            stay_type = 'waiting'  # 停在道路上：等客/堵车，非目的地停留
        elif in_v1 and not poi_id:
            stay_type = 'unmatched'  # 五环内真缺口
        elif not in_v1:
            stay_type = 'out_v1'
        else:
            stay_type = 'poi'
        out.append({
            'taxi_id': s['taxi_id'], 't_start': s['t_start'], 't_end': s['t_end'],
            'dur_min': s['dur_min'], 'lng': s['lng'], 'lat': s['lat'],
            'poi_node': poi_id, 'poi_dist_m': poi_d_m,
            'road_node': road_id, 'road_dist_m': road_d_m,
            'region_node': region_id, 'stay_type': stay_type,
        })

    with open(os.path.join(OUT, 'stay_node_pairs.jsonl'), 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    n_in_v1 = sum(1 for r in out if r['stay_type'] != 'out_v1')
    n_waiting = sum(1 for r in out if r['stay_type'] == 'waiting')
    denom = n_in_v1 - n_waiting  # 修正口径：剔除等待类停留（不期待 POI 对齐）
    cov = n_matched / denom * 100 if denom else 0
    stats = {'total_stays': len(stays), 'stays_in_v1': n_in_v1,
             'waiting_on_road': n_waiting, 'poi_matched': n_matched,
             'align_denominator': denom, 'align_coverage_pct': round(cov, 2)}
    json.dump(stats, open(os.path.join(OUT, 'align_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'停留事件: {len(stays)}（五环内 {n_in_v1}，其中等待类 {n_waiting}），'
          f'POI 对齐: {n_matched}，对齐覆盖率(修正口径): {cov:.2f}%')
    print('输出: stay_node_pairs.jsonl + align_stats.json')


if __name__ == '__main__':
    main()
