# 环节五/03: 离线回放评测——300 用户在真实轨迹上按回放口径逐点判卷
# 口径: 每 20 模拟分钟一个决策点；判卷答案 = 决策后 3 小时内司机真实到达的下一停留
# 指标: MLP top3 是否命中下一站类别（与训练任务一致），分全体/验证集两组
# 输出: data/model/replay_eval.json（含演示推荐天：判卷多且命中多的用户-日期）
import sys, os, json
from collections import Counter
import datetime as dt

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
sys.path.insert(0, os.path.join(ROOT, 'model'))
from push_decide import load_mlp, user_history_before, CAT2ID_CAT  # noqa: E402
from features import CATS, user_history_features, region_adcode  # noqa: E402


def parse_t(s):
    return dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def next_stay_after(seq, t, hours=3):
    for s in seq:
        if s['node_type'] != 'poi':
            continue
        a = parse_t(s['arrive'])
        if a > t and (a - t).total_seconds() <= hours * 3600:
            return s
    return None


def main():
    # 验证集用户（与 train_mlp.py 一致: np.random.seed(42) shuffle unique users 取前 30%）
    meta = [json.loads(l) for l in open(os.path.join(DATA, 'model', 'meta.jsonl'), encoding='utf-8')]
    uniq = np.unique([m['taxi_id'] for m in meta])
    np.random.seed(42)
    np.random.shuffle(uniq)
    val_users = set(uniq[: int(len(uniq) * 0.3)])
    print(f'总用户 {len(uniq)} / 验证集 {len(val_users)}')

    mlp = load_mlp()

    agg = {'all': Counter(), 'val': Counter()}
    day_rows = []
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
            # 决策点: 每 20 模拟分钟一个（与演示回放一致）
            dec = [pts[0]]
            for p in pts[1:]:
                if (parse_t(p['t']) - parse_t(dec[-1]['t'])).total_seconds() >= 20 * 60:
                    dec.append(p)
            if len(dec) < 3:
                continue
            g = h = 0
            for p in dec:
                ns = next_stay_after(prof['seq'], parse_t(p['t']))
                if not ns:
                    continue
                hist_c, hist_d, hist_days = user_history_before(prof, p['t'])
                feats = user_history_features(hist_c, hist_d, hist_days, p['t'],
                                              region_adcode(p['lng'], p['lat']))
                with torch.no_grad():
                    logits = mlp(torch.tensor(feats).unsqueeze(0))[0]
                probs = torch.softmax(logits, dim=-1).numpy()
                top3 = {CATS[i] for i in probs.argsort()[::-1][:3]}
                g += 1
                if CAT2ID_CAT(ns['node']) in top3:
                    h += 1
            agg['all']['g'] += g
            agg['all']['h'] += h
            if uid in val_users:
                agg['val']['g'] += g
                agg['val']['h'] += h
            if g >= 4:
                day_rows.append({'user': f'U_{uid}', 'day': day, 'graded': g, 'hit': h,
                                 'rate': round(h / g, 3), 'val': uid in val_users})

    out = {
        '口径': '每20模拟分钟一个决策点; 判卷=决策后3小时内司机真实下一停留; 命中=MLP top3含该停留类别',
        'all': {'graded': agg['all']['g'], 'hit': agg['all']['h'],
                'rate': round(agg['all']['h'] / max(agg['all']['g'], 1), 4)},
        'val': {'graded': agg['val']['g'], 'hit': agg['val']['h'],
                'rate': round(agg['val']['h'] / max(agg['val']['g'], 1), 4)},
        'top_days': sorted(day_rows, key=lambda r: (-r['hit'], -r['graded']))[:12],
    }
    json.dump(out, open(os.path.join(DATA, 'model', 'replay_eval.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('全体:', out['all'], '验证集:', out['val'])
    print('演示推荐天(top):')
    for r in out['top_days'][:6]:
        print('  ', r)


if __name__ == '__main__':
    main()
