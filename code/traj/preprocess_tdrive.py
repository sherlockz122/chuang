# 环节二/01: T-Drive 轨迹预处理
# 输入:  ../data/tdrive/extracted/release/taxi_log_2008_by_id/*.txt（格式: id,ts,lng,lat）
# 输出:  ../data/tdrive/clean/{taxi_id}.jsonl（按时间排序去重后的点）+ summary.json
import sys, os, json, glob
sys.stdout.reconfigure(encoding='utf-8')
import yaml
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
DATA = os.path.join(os.path.dirname(ROOT), 'data')
CFG = os.path.join(ROOT, 'config')
city = yaml.safe_load(open(os.path.join(CFG, 'city.yaml'), encoding='utf-8'))
minx, miny, maxx, maxy = city['bbox_whole']

SRC = os.path.join(DATA, 'tdrive', 'extracted', 'release', 'taxi_log_2008_by_id')
OUT = os.path.join(DATA, 'tdrive', 'clean')
os.makedirs(OUT, exist_ok=True)


def parse_line(line):
    parts = line.strip().split(',')
    if len(parts) != 4:
        return None
    try:
        tid = int(parts[0])
        ts = datetime.strptime(parts[1], '%Y-%m-%d %H:%M:%S')
        lng, lat = float(parts[2]), float(parts[3])
    except ValueError:
        return None
    if not (minx <= lng <= maxx and miny <= lat <= maxy):
        return None  # 北京 bbox 外（含明显异常点）
    return tid, ts, lng, lat


def main():
    files = sorted(glob.glob(os.path.join(SRC, '*.txt')),
                   key=lambda p: int(os.path.basename(p)[:-4]))
    summary, total_pts, skipped_lines = [], 0, 0
    for fp in files:
        tid = os.path.basename(fp)[:-4]
        pts = []
        with open(fp, encoding='utf-8') as f:
            for line in f:
                r = parse_line(line)
                if r is None:
                    skipped_lines += 1
                    continue
                pts.append(r)
        pts.sort(key=lambda r: r[1])
        # 去重：连续时间戳+坐标完全相同的点
        dedup = []
        for r in pts:
            if dedup and dedup[-1][1] == r[1] and dedup[-1][2] == r[2] and dedup[-1][3] == r[3]:
                continue
            dedup.append(r)
        if not dedup:
            continue
        with open(os.path.join(OUT, f'{tid}.jsonl'), 'w', encoding='utf-8') as f:
            for _, ts, lng, lat in dedup:
                f.write(json.dumps({'t': ts.strftime('%Y-%m-%d %H:%M:%S'),
                                    'lng': lng, 'lat': lat}) + '\n')
        total_pts += len(dedup)
        summary.append({'taxi_id': int(tid), 'n_points': len(dedup),
                        't_min': dedup[0][1].strftime('%Y-%m-%d %H:%M:%S'),
                        't_max': dedup[-1][1].strftime('%Y-%m-%d %H:%M:%S')})
    json.dump({'n_taxis': len(summary), 'total_points': total_pts,
               'skipped_lines': skipped_lines, 'taxis': summary},
              open(os.path.join(DATA, 'tdrive', 'summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'车辆数: {len(summary)}, 总点数: {total_pts}, 跳过行: {skipped_lines}')
    print(f'输出: {OUT}/  + summary.json')


if __name__ == '__main__':
    main()
