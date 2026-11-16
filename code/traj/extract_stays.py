# 环节二/02: 停留事件提取（Zheng 2008 经典停留点算法：时间-距离双阈值）
# 输入:  ../data/tdrive/clean/{taxi_id}.jsonl（子集选择见 config/traj.yaml）
# 输出:  ../data/tdrive/stay_events.jsonl + stay_stats.json
# 用途: 真实停留事件 = 后续对齐验证与推送决策的天然 ground truth
import sys, os, json, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import yaml
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
CFG = os.path.join(ROOT, 'config')
traj = yaml.safe_load(open(os.path.join(CFG, 'traj.yaml'), encoding='utf-8'))

DIST_M = float(traj['stay_dist_m'])
TIME_MIN = float(traj['stay_time_min'])
N_TAXIS = int(traj['n_taxis'])
MIN_POINTS = int(traj['min_points'])
DEG = 111320.0
LATF = 0.813


def haversine(lng1, lat1, lng2, lat2):
    """两点米距（近似，北京范围内足够）"""
    dx = (lng1 - lng2) * DEG
    dy = (lat1 - lat2) * DEG * LATF
    return np.hypot(dx, dy)


def extract(pts):
    """pts: [(ts_datetime, lng, lat)] 时间有序；返回停留事件列表"""
    events = []
    i = 0
    n = len(pts)
    while i < n:
        j = i + 1
        anchor = pts[i]
        maxd = 0.0
        while j < n:
            d = haversine(anchor[1], anchor[2], pts[j][1], pts[j][2])
            maxd = max(maxd, d)
            if maxd > DIST_M:
                break  # 移出停留圈
            j += 1
        # [i, j) 是当前最长的停留圈内连续段
        if j < n and (pts[j][0] - anchor[0]).total_seconds() / 60.0 < TIME_MIN:
            i = j
            continue
        # 尾段处理：时间够长且圈内
        if (pts[j - 1][0] - anchor[0]).total_seconds() / 60.0 >= TIME_MIN:
            seg = pts[i:j]
            lng = float(np.mean([p[1] for p in seg]))
            lat = float(np.mean([p[2] for p in seg]))
            events.append({
                't_start': seg[0][0].strftime('%Y-%m-%d %H:%M:%S'),
                't_end': seg[-1][0].strftime('%Y-%m-%d %H:%M:%S'),
                'dur_min': round((seg[-1][0] - seg[0][0]).total_seconds() / 60.0, 1),
                'lng': round(lng, 6), 'lat': round(lat, 6),
                'n_points': len(seg),
            })
            i = j
        else:
            i += 1
    return events


def main():
    src = os.path.join(DATA, 'tdrive', 'clean')
    out_fp = os.path.join(DATA, 'tdrive', 'stay_events.jsonl')

    # 子集选择（确定性）: 质量筛选（点数≥MIN_POINTS）→ 按 taxi_id 升序取前 N
    summary = json.load(open(os.path.join(DATA, 'tdrive', 'summary.json'), encoding='utf-8'))
    eligible = sorted([t for t in summary['taxis'] if t['n_points'] >= MIN_POINTS],
                      key=lambda t: t['taxi_id'])
    subset = {t['taxi_id'] for t in eligible[:N_TAXIS]}
    print(f'质量筛选后 {len(eligible)} 辆，取前 {N_TAXIS} 辆（taxi_id 升序）')

    total_events, per_day = 0, {}
    n_taxis = 0
    with open(out_fp, 'w', encoding='utf-8') as out:
        for fp in glob.glob(os.path.join(src, '*.jsonl')):
            tid = int(os.path.basename(fp)[:-6])
            if tid not in subset:
                continue
            pts = []
            with open(fp, encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    pts.append((datetime.strptime(r['t'], '%Y-%m-%d %H:%M:%S'),
                                r['lng'], r['lat']))
            for ev in extract(pts):
                ev['taxi_id'] = tid
                out.write(json.dumps(ev, ensure_ascii=False) + '\n')
                total_events += 1
                day = ev['t_start'][:10]
                per_day[day] = per_day.get(day, 0) + 1
            n_taxis += 1
    json.dump({'n_taxis': n_taxis, 'total_events': total_events,
               'per_day': per_day, 'params': {'dist_m': DIST_M, 'time_min': TIME_MIN},
               'subset': {'n_taxis': N_TAXIS, 'min_points': MIN_POINTS,
                          'eligible': len(eligible)}},
              open(os.path.join(DATA, 'tdrive', 'stay_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'停留事件: {total_events}（{n_taxis} 辆车）')
    print('每日分布:', json.dumps(per_day, ensure_ascii=False))


if __name__ == '__main__':
    main()
