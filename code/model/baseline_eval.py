# 环节五/04: 对照基线评测——与回放评测同一口径，换预测器对比说服力
# 决策点/判卷答案与 replay_eval.py 完全一致（每20模拟分钟一点；下一停留3小时内）
# 基线1 频次基线: 恒推该用户历史最常去的 top3 类别（不需要任何模型）
# 基线2 随机基线: 恒随机猜 3 个类别（理论值 = 3/19 ≈ 15.8%）
# 输出: data/model/baseline_eval.json
import sys, os, json
from collections import Counter
import datetime as dt
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
sys.path.insert(0, os.path.join(ROOT, 'model'))
from features import CATS, CAT2ID  # noqa: E402

NODE_CAT = {}
def CAT2ID_CAT(node_id):
    if not NODE_CAT:
        nodes = json.load(open(os.path.join(DATA, 'graph_out', 'nodes.json'), encoding='utf-8'))
        NODE_CAT.update({n['node_id']: n.get('category', '') for n in nodes})
    return NODE_CAT.get(node_id, '')

def parse_t(s):
    return dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


meta = [json.loads(l) for l in open(os.path.join(DATA, 'model', 'meta.jsonl'), encoding='utf-8')]
uniq = np.unique([m['taxi_id'] for m in meta])
np.random.seed(42)
np.random.shuffle(uniq)
val_users = set(uniq[: int(len(uniq) * 0.3)])
random.seed(42)


def next_stay_after(seq, t, hours=3):
    for s in seq:
        if s['node_type'] != 'poi':
            continue
        a = parse_t(s['arrive'])
        if a > t and (a - t).total_seconds() <= hours * 3600:
            return s
    return None


agg = {'all': Counter(), 'val': Counter()}
for uid in sorted(uniq):
    prof = json.load(open(os.path.join(DATA, 'aligned', 'user_profiles', f'{uid}.json'),
                          encoding='utf-8'))
    fp = os.path.join(DATA, 'tdrive', 'clean', f'{uid}.jsonl')
    if not os.path.exists(fp):
        continue
    days = {}
    for line in open(fp, encoding='utf-8'):
        o = json.loads(line)
        days.setdefault(o['t'][:10], []).append({'t': o['t'], 'lng': o['lng'], 'lat': o['lat']})
    for day, pts in sorted(days.items()):
        dec = [pts[0]]
        for p in pts[1:]:
            if (parse_t(p['t']) - parse_t(dec[-1]['t'])).total_seconds() >= 20 * 60:
                dec.append(p)
        if len(dec) < 3:
            continue
        for p in dec:
            ns = next_stay_after(prof['seq'], parse_t(p['t']))
            if not ns:
                continue
            # 历史 = 该决策时刻之前的停留（与 replay_eval 同口径）
            hist_c = Counter()
            for s in prof['seq']:
                if s['arrive'] >= p['t']:
                    break
                if s['node_type'] == 'poi':
                    hist_c[CAT2ID_CAT(s['node'])] += 1
            freq_top3 = {c for c, _ in hist_c.most_common(3)}
            rand_top3 = set(random.sample(CATS, 3))
            true = CAT2ID_CAT(ns['node'])
            which = 'all' if uid not in val_users else 'val'
            agg[which]['g'] += 1
            agg['all']['g'] += 1
            if true in freq_top3:
                agg[which]['hf'] += 1
                agg['all']['hf'] += 1
            if true in rand_top3:
                agg[which]['hr'] += 1
                agg['all']['hr'] += 1

out = {
    '口径': '与 replay_eval 同一决策点/判卷; 频次基线=历史最常去top3; 随机基线=随机3类',
    'all': {'graded': agg['all']['g'],
            'freq': {'hit': agg['all']['hf'], 'rate': round(agg['all']['hf'] / max(agg['all']['g'], 1), 4)},
            'rand': {'hit': agg['all']['hr'], 'rate': round(agg['all']['hr'] / max(agg['all']['g'], 1), 4)}},
    'val': {'graded': agg['val']['g'],
            'freq': {'hit': agg['val']['hf'], 'rate': round(agg['val']['hf'] / max(agg['val']['g'], 1), 4)},
            'rand': {'hit': agg['val']['hr'], 'rate': round(agg['val']['hr'] / max(agg['val']['g'], 1), 4)}},
}
json.dump(out, open(os.path.join(DATA, 'model', 'baseline_eval.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('全体: 频次基线', out['all']['freq'], ' 随机基线', out['all']['rand'])
print('验证集: 频次基线', out['val']['freq'], ' 随机基线', out['val']['rand'])
