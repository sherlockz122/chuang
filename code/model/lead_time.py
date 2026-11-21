# 环节五/08: 预测提前量统计——决策时刻与司机真实到达停留点的时间差分布
# 回答"固定间隔预测能否保证在到达前给出结果"：判卷口径上预测永远先于答案
# (答案=决策后3小时内第一停留)，本脚本量化提前量(分钟)的分布
import sys, os, json
import datetime as dt

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')

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


def stats(a):
    a = np.array(a)
    return {'n': int(len(a)), 'median_min': round(float(np.median(a)), 1),
            'p25': round(float(np.percentile(a, 25)), 1), 'p75': round(float(np.percentile(a, 75)), 1),
            'p90': round(float(np.percentile(a, 90)), 1), 'max_min': round(float(a.max()), 1),
            'ge5_pct': round(float((a >= 5).mean() * 100), 1),
            'ge15_pct': round(float((a >= 15).mean() * 100), 1),
            'ge30_pct': round(float((a >= 30).mean() * 100), 1),
            'ge60_pct': round(float((a >= 60).mean() * 100), 1)}


def main():
    meta = [json.loads(l) for l in open(os.path.join(DATA, 'model', 'meta.jsonl'), encoding='utf-8')]
    uniq = np.unique([m['taxi_id'] for m in meta])
    np.random.seed(42)
    np.random.shuffle(uniq)
    val_users = set(uniq[: int(len(uniq) * 0.3)])

    leads, val_leads = [], []
    n_dec = n_graded = 0
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
                n_dec += 1
                ns = next_stay_after(prof['seq'], parse_t(p['t']))
                if not ns:
                    continue
                n_graded += 1
                lead = (parse_t(ns['arrive']) - parse_t(p['t'])).total_seconds() / 60
                leads.append(lead)
                if uid in val_users:
                    val_leads.append(lead)

    def summ(a, name):
        a = np.array(a)
        print(f'{name}: n={len(a)} 中位 {np.median(a):.0f}min | p25 {np.percentile(a,25):.0f} | '
              f'p75 {np.percentile(a,75):.0f} | p90 {np.percentile(a,90):.0f} | 最大 {a.max():.0f}')
        print(f'  提前≥5min: {(a>=5).mean()*100:.1f}% | ≥15min: {(a>=15).mean()*100:.1f}% | '
              f'≥30min: {(a>=30).mean()*100:.1f}% | ≥60min: {(a>=60).mean()*100:.1f}%')

    print(f'决策点 {n_dec} / 有答案(3h内有停留) {n_graded} = {n_graded/max(n_dec,1)*100:.1f}%')
    summ(leads, '全体提前量')
    summ(val_leads, '验证集提前量')

    out = {'口径': '决策点=每20模拟分钟; 提前量=决策时刻→3h内司机真实到达的第一停留(分钟)',
           'decision_points': n_dec, 'graded': n_graded,
           'graded_rate': round(n_graded / max(n_dec, 1), 4),
           'all': stats(leads), 'val': stats(val_leads)}
    json.dump(out, open(os.path.join(DATA, 'model', 'lead_time.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('已存 data/model/lead_time.json')


if __name__ == '__main__':
    main()
