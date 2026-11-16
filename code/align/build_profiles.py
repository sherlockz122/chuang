# 环节三/02: 每用户语义 JSON（"语义化"落地：轨迹 → 图节点 ID 序列 + 时间语义）
# 输入:  ../data/aligned/stay_node_pairs.jsonl
# 输出:  ../data/aligned/user_profiles/{taxi_id}.json（每车一个，模型输入/演示数据源）
# 口径: 节点 ID 与 nodes.json 强制一致（OSM_N_*/OSM_W_*_*/REG_*），JSON 与模型解耦——
#       模型吃的是这些 ID 与统计特征，JSON 是给 LLM 通道的"句子"。
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'aligned', 'user_profiles')
os.makedirs(OUT, exist_ok=True)


def hour_bin(t):
    return int(t[11:13])


def main():
    pairs = [json.loads(l) for l in open(os.path.join(DATA, 'aligned', 'stay_node_pairs.jsonl'),
                                          encoding='utf-8')]
    by_taxi = defaultdict(list)
    for p in pairs:
        by_taxi[p['taxi_id']].append(p)

    n_users = 0
    for tid, evs in by_taxi.items():
        evs.sort(key=lambda e: e['t_start'])
        seq = []
        for e in evs:
            seq.append({
                'node': e['poi_node'] or e['road_node'] or e['region_node'],
                'node_type': 'poi' if e['poi_node'] else ('road' if e['road_node'] else 'region'),
                'arrive': e['t_start'], 'leave': e['t_end'],
                'dur_min': e['dur_min'], 'hour': hour_bin(e['t_start']),
            })
        profile = {
            'user_id': f'U_{tid}',
            'n_stays': len(seq),
            'days': sorted({s['arrive'][:10] for s in seq}),
            'seq': seq,
        }
        with open(os.path.join(OUT, f'{tid}.json'), 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False)
        n_users += 1
    print(f'用户语义 JSON: {n_users} 个 -> {OUT}/')


if __name__ == '__main__':
    main()
