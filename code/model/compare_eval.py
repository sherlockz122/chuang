# 环节五/05: MLP vs 频次基线 分片对比——回答"MLP 到底赢在哪"
# 与 replay_eval/baseline_eval 同一决策点，同一判卷答案；
# 每个决策点同时算 频次top3 / MLP top3 是否命中，再按 时段 与 用户历史停留数 分片
# 输出: data/model/compare_eval.json
import sys, os, json
from collections import Counter, defaultdict
import datetime as dt

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
sys.path.insert(0, os.path.join(ROOT, 'model'))
from push_decide import load_mlp, user_history_before, CAT2ID_CAT  # noqa: E402
from features import CATS, user_history_features  # noqa: E402

def parse_t(s):
    return dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

meta = [json.loads(l) for l in open(os.path.join(DATA, 'model', 'meta.jsonl'), encoding='utf-8')]
uniq = np.unique([m['taxi_id'] for m in meta])
np.random.seed(42)
np.random.shuffle(uniq)
val_users = set(uniq[: int(len(uniq) * 0.3)])

mlp = load_mlp()

def next_stay_after(seq, t, hours=3):
    for s in seq:
        if s['node_type'] != 'poi':
            continue
        a = parse_t(s['arrive'])
        if a > t and (a - t).total_seconds() <= hours * 3600:
            return s
    return None

# 分片计数: [mlp_graded, mlp_hit, freq_hit]
hour = defaultdict(lambda: [0, 0, 0])
tier = defaultdict(lambda: [0, 0, 0])
tot = [0, 0, 0]
val_tot = [0, 0, 0]

for uid in sorted(uniq):
    prof = json.load(open(os.path.join(DATA, 'aligned', 'user_profiles', f'{uid}.json'),
                          encoding='utf-8'))
    fp = os.path.join(DATA, 'tdrive', 'clean', f'{uid}.jsonl')
    if not os.path.exists(fp):
        continue
    n_stays = sum(1 for s in prof['seq'] if s['node_type'] == 'poi')
    # 用户停留数 1~39（中位 10），按四分位分三档
    tier_name = '冷启动(停留<8)' if n_stays < 8 else ('常客(停留8-14)' if n_stays < 15 else '重度(停留≥15)')
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
            hist_c, hist_d, hist_days = user_history_before(prof, p['t'])
            feats = user_history_features(hist_c, hist_d, hist_days, p['t'], '')
            with torch.no_grad():
                logits = mlp(torch.tensor(feats).unsqueeze(0))[0]
            probs = torch.softmax(logits, dim=-1).numpy()
            mlp_top3 = {CATS[i] for i in probs.argsort()[::-1][:3]}
            freq_top3 = {c for c, _ in hist_c.most_common(3)}
            true = CAT2ID_CAT(ns['node'])
            h = parse_t(p['t']).hour
            hour[h][0] += 1
            hour[h][1] += 1 if true in mlp_top3 else 0
            hour[h][2] += 1 if true in freq_top3 else 0
            tier[tier_name][0] += 1
            tier[tier_name][1] += 1 if true in mlp_top3 else 0
            tier[tier_name][2] += 1 if true in freq_top3 else 0
            tot[0] += 1
            tot[1] += 1 if true in mlp_top3 else 0
            tot[2] += 1 if true in freq_top3 else 0
            if uid in val_users:
                val_tot[0] += 1
                val_tot[1] += 1 if true in mlp_top3 else 0
                val_tot[2] += 1 if true in freq_top3 else 0

def fmt(t):
    return {'graded': t[0], 'mlp': round(t[1] / max(t[0], 1), 4),
            'freq': round(t[2] / max(t[0], 1), 4),
            'diff': round((t[1] - t[2]) / max(t[0], 1), 4)}

out = {
    'all': fmt(tot), 'val': fmt(val_tot),
    'by_hour': {str(h): fmt(v) for h, v in sorted(hour.items()) if v[0] >= 100},
    'by_tier': {k: fmt(v) for k, v in tier.items()},
}
json.dump(out, open(os.path.join(DATA, 'model', 'compare_eval.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('全体:', out['all'])
print('验证集:', out['val'])
print('\n分时段(判卷>=100, diff=MLP-频次):')
for h, v in out['by_hour'].items():
    print(f'  {h}点 graded={v["graded"]} MLP={v["mlp"]:.3f} 频次={v["freq"]:.3f} diff={v["diff"]:+.3f}')
print('\n分活跃度:')
for k, v in out['by_tier'].items():
    print(f'  {k}: graded={v["graded"]} MLP={v["mlp"]:.3f} 频次={v["freq"]:.3f} diff={v["diff"]:+.3f}')
