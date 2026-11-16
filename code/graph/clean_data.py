# 环节一/02: 清洗——去重、坐标核查、道路长度 geodesic 重算
# 输入:  ../data/graph_out/raw_poi.jsonl, raw_roads.jsonl
# 输出:  ../data/graph_out/clean_poi.jsonl, clean_roads.jsonl + 去重报告
import sys, os, re, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.spatial import cKDTree
from pyproj import Geod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(ROOT), 'data', 'graph_out')
geod = Geod(ellps='WGS84')


def norm_name(name: str) -> str:
    """名称标准化：去空白、去括号内注释（分店名），用于同名判重"""
    if not name:
        return ''
    n = re.sub(r'[（(].*?[)）]', '', name)
    n = re.sub(r'\s+', '', n)
    return n.lower()


def dedup(records, dist_m=50):
    """同名（标准化后）且距离<dist_m 的合并为一条（保留先出现者，即 OSM 主键）"""
    coords = np.array([[r['lng'], r['lat']] for r in records])
    # 球面坐标直接 KDTree（近似欧氏），50m 内误差可忽略
    tree = cKDTree(coords)
    groups = {}  # norm_name -> idx 列表
    for i, r in enumerate(records):
        groups.setdefault(norm_name(r['name']), []).append(i)
    keep, merged = [], 0
    merged_ids = []
    for name, idxs in groups.items():
        if len(idxs) == 1:
            keep.append(records[idxs[0]])
            continue
        remaining = set(idxs)
        while remaining:
            i0 = min(remaining)
            remaining.discard(i0)
            near = tree.query_ball_point(coords[i0], dist_m / 111320.0)
            dup = remaining.intersection(near)
            for j in dup:
                merged += 1
                merged_ids.append((records[i0]['osm_id'], records[j]['osm_id']))
                remaining.discard(j)
            keep.append(records[i0])
    return keep, merged, merged_ids


def main():
    pois = [json.loads(l) for l in open(os.path.join(OUT, 'raw_poi.jsonl'), encoding='utf-8')]
    roads = [json.loads(l) for l in open(os.path.join(OUT, 'raw_roads.jsonl'), encoding='utf-8')]
    print(f'原始: POI {len(pois)} / 道路段 {len(roads)}')

    keep_poi, n_merged, merged_ids = dedup(pois)
    with open(os.path.join(OUT, 'clean_poi.jsonl'), 'w', encoding='utf-8') as f:
        for r in keep_poi:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'POI 去重: 合并 {n_merged} 条 -> 保留 {len(keep_poi)}')

    # 道路段长度 geodesic 重算
    for r in roads:
        try:
            _, _, dist = geod.inv(r['x1'], r['y1'], r['x2'], r['y2'])
            r['length_m'] = round(dist, 1)
        except Exception:
            pass
    with open(os.path.join(OUT, 'clean_roads.jsonl'), 'w', encoding='utf-8') as f:
        for r in roads:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'道路段保留: {len(roads)}')

    report = {'poi_raw': len(pois), 'poi_kept': len(keep_poi), 'poi_merged': n_merged,
              'roads': len(roads), 'merge_examples': merged_ids[:5]}
    json.dump(report, open(os.path.join(OUT, 'clean_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('去重示例:', merged_ids[:5])


if __name__ == '__main__':
    main()
