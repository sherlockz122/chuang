# 环节五/06: 情景题库探针——扫描 用户×天×时刻 的 MLP top3，
# 为金标准情景题库寻找"前置条件匹配"的 (用户, 时刻) 组合（MLP 重训后可重跑刷新）
import sys, os, json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
sys.path.insert(0, os.path.join(ROOT, 'model'))
from push_decide import load_mlp, user_history_before  # noqa: E402
from features import CATS, user_history_features  # noqa: E402

mlp = load_mlp()


def top3_at(prof, t_str):
    hist_c, hist_d, hist_days = user_history_before(prof, t_str)
    feats = user_history_features(hist_c, hist_d, hist_days, t_str, '')
    with torch.no_grad():
        logits = mlp(torch.tensor(feats).unsqueeze(0))[0]
    probs = torch.softmax(logits, dim=-1).numpy()
    idx = probs.argsort()[::-1]
    return [CATS[i] for i in idx if i < len(CATS)][:3], round(float(probs.max()), 3)


meta = [json.loads(l) for l in open(os.path.join(DATA, 'model', 'meta.jsonl'), encoding='utf-8')]
uniq = np.unique([m['taxi_id'] for m in meta])

want = {
    'night_scenic': [],    # 0-5点 top3 含 scenic
    'day_scenic': [],      # 12-15点 top3 含 scenic
    'lunch_restaurant': [],  # 11-13点 top3 含 restaurant
    'day_fuel': [],        # 10-18点 top3 含 fuel
    'cold_midday': [],     # 停留<8 的用户中午时刻
}
N = 3  # 每种条件保留几个候选

for uid in sorted(uniq):
    fp = os.path.join(DATA, 'aligned', 'user_profiles', f'{uid}.json')
    if not os.path.exists(fp):
        continue
    prof = json.load(open(fp, encoding='utf-8'))
    days = prof.get('days') or []
    if not days:
        continue
    last = days[-1]
    n_stays = prof.get('n_stays', 0)
    for hour in (1, 12, 20):
        t = f'{last} {hour:02d}:00:00'
        top3, maxp = top3_at(prof, t)
        row = {'user': f'U_{uid}', 'time': t, 'top3': top3, 'max_p': maxp, 'n_stays': n_stays}
        if hour == 1 and 'scenic' in top3 and len(want['night_scenic']) < N:
            want['night_scenic'].append(row)
        if 12 <= hour <= 20 and 'scenic' in top3 and len(want['day_scenic']) < N:
            want['day_scenic'].append(row)
        if 11 <= hour <= 13 and 'restaurant' in top3 and len(want['lunch_restaurant']) < N:
            want['lunch_restaurant'].append(row)
        if 10 <= hour <= 18 and 'fuel' in top3 and len(want['day_fuel']) < N:
            want['day_fuel'].append(row)
        if n_stays < 8 and hour == 12 and len(want['cold_midday']) < N:
            want['cold_midday'].append(row)
    if all(len(v) >= N for v in want.values()):
        break

print(json.dumps(want, ensure_ascii=False, indent=1))
json.dump(want, open(os.path.join(DATA, 'model', 'scenario_probe.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
