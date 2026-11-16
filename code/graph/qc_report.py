# 环节一/05: 语义图质检——统计报告 + folium 分层着色图
# 输入:  ../data/graph_out/nodes.json, edges.json, build_stats.json, graph.pkl
# 输出:  ../data/graph_out/qc_report.md, graph_map.html
import sys, os, json, pickle
sys.stdout.reconfigure(encoding='utf-8')
import folium
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(ROOT), 'data', 'graph_out')

# 分组着色（group 口径：instant/destination/anchor）
GROUP_COLORS = {'instant': '#E4572E', 'destination': '#4472C4', 'anchor': '#2A9D8F'}


def main():
    nodes = json.load(open(os.path.join(OUT, 'nodes.json'), encoding='utf-8'))
    edges = [json.loads(l) for l in open(os.path.join(OUT, 'edges.json'), encoding='utf-8')]
    G = pickle.load(open(os.path.join(OUT, 'graph.pkl'), 'rb'))

    # ---------- 统计 ----------
    poi_nodes = [n for n in nodes if n['type'] == 'poi']
    road_nodes = [n for n in nodes if n['type'] == 'road']
    reg_nodes = [n for n in nodes if n['type'] == 'region']
    etype_counts = Counter(e['etype'] for e in edges)
    cat_counts = Counter(n['category'] for n in poi_nodes)
    poi_ids = [n['node_id'] for n in poi_nodes]
    degs = [d for _, d in G.degree(poi_ids)]
    mean_deg = sum(degs) / len(degs) if degs else 0
    isolated = sum(1 for d in degs if d == 0)

    lines = [
        '# POI 语义图质检报告',
        '',
        f'- 节点: POI {len(poi_nodes)} / 道路段 {len(road_nodes)} / 区域 {len(reg_nodes)}，共 {len(nodes)}',
        f'- 边: 共 {len(edges)} 条，{dict(etype_counts)}',
        f'- POI 平均度数: {mean_deg:.1f}，孤立 POI（度=0）: {isolated}',
        '',
        '## POI 分类分布',
        '',
    ]
    for cid, cnt in cat_counts.most_common():
        grp = next((n['group'] for n in poi_nodes if n['category'] == cid), '?')
        lines.append(f'- {cid}({grp}): {cnt}')

    # 度=0 的孤立 POI 列前 10 个（质检重点：没有接任何边的 POI 无法被推送）
    isolated_names = [n['name'] for n, d in zip(poi_nodes, degs) if d == 0][:10]
    lines += ['', '## 孤立 POI 示例（前 10）', '', f'- {isolated_names if isolated_names else "无"}']
    open(os.path.join(OUT, 'qc_report.md'), 'w', encoding='utf-8').write('\n'.join(lines))

    # ---------- folium 分层着色图 ----------
    m = folium.Map(location=[39.93, 116.40], zoom_start=11, tiles='CartoDB positron')
    for n in road_nodes:
        folium.CircleMarker([n['lat'], n['lng']], radius=0.6, color='#8D99AE',
                            fill=True, fill_opacity=0.4, weight=0).add_to(m)
    for n in poi_nodes:
        color = GROUP_COLORS.get(n['group'], '#596B90')
        folium.CircleMarker([n['lat'], n['lng']], radius=2.0, color=color,
                            fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"{n['name']}({n['category']})").add_to(m)
    m.save(os.path.join(OUT, 'graph_map.html'))
    print('\n'.join(lines))
    print(f'\n交互图: {os.path.join(OUT, "graph_map.html")}')


if __name__ == '__main__':
    main()
