# 环节五/06: 情景题库探针——扫描 用户×天×时刻 的 MLP top3，
# 为金标准情景题库寻找"前置条件匹配"的 (用户, 时刻) 组合（MLP 重训后可重跑刷新）
import sys, os, json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
sys.path.insert(0, os.path.join(ROOT, 'model'))
from push_decide import load_mlp, user_history_before  # noqa: E402
from features import CATS, user_history_features, region_adcode  # noqa: E402

mlp = load_mlp()

_NODES = None


def poi_pos(cat):
    """与 scenario_bank 同一口径: 取第一个带名字的目标类 POI 坐标"""
    global _NODES
    if _NODES is None:
        _NODES = json.load(open(os.path.join(DATA, 'graph_out', 'nodes.json'), encoding='utf-8'))
    for n in _NODES:
        if n.get('type') == 'poi' and n.get('category') == cat and n.get('name'):
            return n['lng'], n['lat']
    return None


def top3_at(prof, t_str, pos_cat):
    """按情景位置口径算 top3（区域特征与 decide() 推理一致）"""
    lng, lat = poi_pos(pos_cat)
    rid = region_adcode(lng, lat)
    hist_c, hist_d, hist_days = user_history_before(prof, t_str)
    feats = user_history_features(hist_c, hist_d, hist_days, t_str, rid)
    with torch.no_grad():
        logits = mlp(torch.tensor(feats).unsqueeze(0))[0]
    probs = torch.softmax(logits, dim=-1).numpy()
    idx = probs.argsort()[::-1]
    return [CATS[i] for i in idx if i < len(CATS)][:3], round(float(probs.max()), 3)


def main():
    meta = [json.loads(l) for l in open(os.path.join(DATA, 'model', 'meta.jsonl'), encoding='utf-8')]
    uniq = np.unique([m['taxi_id'] for m in meta])

    # 情景前置条件（与 scenario_bank 判卷口径一致）：
    # 景区从不进任何用户 top3，景区情景靠互补对 [scenic,parking] 进候选 → 条件=top3 含 parking
    want = {
        'night_scenic': [],    # 0-5点 top3 含 parking（位置=景区 POI）
        'day_scenic': [],      # 12-20点 top3 含 parking
        'lunch_restaurant': [],  # 11-13点 top3 含 restaurant
        'day_fuel': [],        # 10-18点 top3 含 fuel
        'cold_midday': [],     # 停留<8 的用户中午时刻
    }
    POS_CAT = {'night_scenic': 'scenic', 'day_scenic': 'scenic',
               'lunch_restaurant': 'restaurant', 'day_fuel': 'fuel',
               'cold_midday': 'restaurant'}
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
            for key in want:
                if len(want[key]) >= N:
                    continue
                top3, maxp = top3_at(prof, t, POS_CAT[key])
                row = {'user': f'U_{uid}', 'time': t, 'top3': top3, 'max_p': maxp, 'n_stays': n_stays}
                if key == 'night_scenic' and hour == 1 and 'parking' in top3:
                    want[key].append(row)
                elif key == 'day_scenic' and 12 <= hour <= 20 and 'parking' in top3:
                    want[key].append(row)
                elif key == 'lunch_restaurant' and 11 <= hour <= 13 and 'restaurant' in top3:
                    want[key].append(row)
                elif key == 'day_fuel' and 10 <= hour <= 18 and 'fuel' in top3:
                    want[key].append(row)
                elif key == 'cold_midday' and n_stays < 8 and hour == 12:
                    want[key].append(row)
        if all(len(v) >= N for v in want.values()):
            break

    print(json.dumps(want, ensure_ascii=False, indent=1))
    json.dump(want, open(os.path.join(DATA, 'model', 'scenario_probe.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
