# 环节四/01: 特征工程——用户语义 JSON + 图 → MLP 输入特征（"说数字"通道的编码层）
# 输入:  ../data/aligned/user_profiles/*.json + ../data/aligned/stay_node_pairs.jsonl
#       ../data/graph_out/nodes.json
# 输出:  ../data/model/features.npz（X 特征矩阵, y 类别标签, meta 元信息）
# 口径:  样本 = 一次真实停留（按时间序）；特征只用"该停留之前"的历史（在线口径，防泄漏）
#        y = 该停留 POI 的类别（19 类之一）→ 感知准确率 = top-k 命中
import sys, os, json, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import yaml
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'model')
os.makedirs(OUT, exist_ok=True)

CATS = ['fuel', 'charging', 'parking', 'toilet', 'hospital', 'pharmacy', 'convenience',
        'car_repair', 'services', 'restaurant', 'shopping', 'scenic', 'hotel', 'park',
        'sports', 'cinema', 'transport', 'office', 'school']
CAT2ID = {c: i for i, c in enumerate(CATS)}
N_REGION = 11


def user_history_features(hist_cats, hist_durs, hist_days, t_str, region_rid):
    """构造单样本 64 维特征（训练与推流共用同一口径）
    hist_cats: 历史类别 Counter（不含当前样本）
    hist_durs: 历史停留时长列表（分钟）
    hist_days: 历史活跃天数集合
    t_str: 'YYYY-MM-DD HH:MM:SS'; region_rid: 区域 adcode 字符串（可为 ''）
    """
    import datetime
    hist_total = sum(hist_cats.values()) or 1
    cat_freq = np.array([hist_cats[c] / hist_total for c in CATS], dtype=np.float32)
    scalars = np.array([
        len(hist_durs) / 20.0,
        np.mean(hist_durs) / 120.0 if hist_durs else 0.0,
        len(hist_days) / 7.0,
    ], dtype=np.float32)
    hour = np.zeros(24, dtype=np.float32)
    hour[int(t_str[11:13])] = 1.0
    weekday = np.zeros(7, dtype=np.float32)
    weekday[datetime.datetime.strptime(t_str[:10], '%Y-%m-%d').weekday()] = 1.0
    region = np.zeros(N_REGION, dtype=np.float32)
    if region_rid:
        region[int(region_rid) % N_REGION] = 1.0  # 简化：按 adcode 散列到 11 维（区域维度可换 one-hot 表）
    return np.concatenate([cat_freq, scalars, hour, weekday, region])


def main():
    nodes = {n['node_id']: n for n in json.load(
        open(os.path.join(DATA, 'graph_out', 'nodes.json'), encoding='utf-8'))}
    pairs = [json.loads(l) for l in open(os.path.join(DATA, 'aligned', 'stay_node_pairs.jsonl'),
                                          encoding='utf-8')]
    by_taxi = {}
    for p in pairs:
        by_taxi.setdefault(p['taxi_id'], []).append(p)

    X, y, meta = [], [], []
    n_dropped = 0
    for tid, evs in by_taxi.items():
        evs.sort(key=lambda e: (e['t_start']))
        hist_cat = Counter()      # 历史类别计数（在线口径：不含当前样本）
        hist_dur = []             # 历史停留时长
        hist_days = set()         # 历史活跃天数
        for e in evs:
            n = nodes.get(e['poi_node'] or '')
            if n is None or n['type'] != 'poi':
                n_dropped += 1
                continue  # 无 POI 对齐的停留不构成类别监督样本
            cat = n.get('category')
            if cat not in CAT2ID:
                n_dropped += 1
                continue
            x = user_history_features(hist_cat, hist_dur, hist_days, e['t_start'],
                                      (e['region_node'] or '').replace('REG_', ''))
            X.append(x)
            y.append(CAT2ID[cat])
            meta.append({'taxi_id': tid, 't': e['t_start'], 'cat': cat,
                         'poi_node': e['poi_node']})
            # 更新历史
            hist_cat[cat] += 1
            hist_dur.append(e['dur_min'])
            hist_days.add(e['t_start'][:10])

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    np.savez(os.path.join(OUT, 'features.npz'), X=X, y=y)
    with open(os.path.join(OUT, 'meta.jsonl'), 'w', encoding='utf-8') as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    print(f'样本: {len(y)}（丢弃无 POI 对齐 {n_dropped}），特征维度: {X.shape[1]}')
    print('类别分布:', {CATS[i]: int((y == i).sum()) for i in range(len(CATS)) if (y == i).any()})
    print(f'输出: {OUT}/features.npz + meta.jsonl')


if __name__ == '__main__':
    main()
