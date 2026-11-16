# 环节五/04b: 外部测试——学生CNN+PERCLOS vs 教师EAR+PERCLOS 在 NTHU(4受试者) 视频级准确率
# 协议(受试者隔离): 每段视频均匀抽 100 帧 → 逐帧眼态(学生p_open / 教师EAR) → 视频级PERCLOS
#                   → 阈值判定困/清醒；阈值在 001/002/005 上搜索，006 上验证
# 环境: D:/mp-env/Scripts/python.exe（mediapipe 无 3.13 wheel）
# 输出:  ext_test_nthu.json（学生/教师双通道视频级 acc + 分受试者 + 阈值 + 两类PERCLOS分布）
import sys, os, json, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from collections import defaultdict
from fatigue_infer import load_image, frame_eye_states

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'model')
IMG_DIR = os.path.join(DATA, 'fatigue', 'nthu-ddd', 'NTHU DDD')
N_FRAMES_PER_VIDEO = 100
EAR_CLOSED = 0.20  # 教师口径（与伪标签一致）


def video_of(p):
    """视频ID=前3字段(受试者_眼镜_组合)，如 001_glasses_sleepyCombination"""
    base = os.path.basename(p)
    return '_'.join(base.split('_')[:3])


def label_of(vid):
    return 1 if 'sleepy' in vid else 0  # sleepyCombination=困; nonsleepy=清醒


def main():
    frames = sorted(glob.glob(os.path.join(IMG_DIR, 'drowsy', '*.jpg')) +
                    glob.glob(os.path.join(IMG_DIR, 'notdrowsy', '*.jpg')))
    by_video = defaultdict(list)
    for p in frames:
        by_video[video_of(p)].append(p)
    videos = sorted(by_video)
    print(f'视频 {len(videos)} 段，帧 {len(frames)}')

    # 逐视频抽帧 → 学生PERCLOS + 教师EAR-PERCLOS
    perclos, ear_pc, labels, subjs, seen = {}, {}, {}, {}, {}
    for vid in videos:
        ps = by_video[vid]
        step = max(1, len(ps) // N_FRAMES_PER_VIDEO)
        sample = ps[::step][:N_FRAMES_PER_VIDEO]
        closed = ear_closed_n = n_seen = ear_seen = 0
        for p in sample:
            img = load_image(p)
            if img is None:
                continue
            states = frame_eye_states(img)
            if states is None:
                continue
            n_seen += 1
            if any(p_o < 0.5 for p_o, _ in states):
                closed += 1
            if any(e is not None for _, e in states):
                ear_seen += 1
                if any(e is not None and e <= EAR_CLOSED for _, e in states):
                    ear_closed_n += 1
        if n_seen == 0:
            continue
        perclos[vid] = closed / n_seen
        ear_pc[vid] = (ear_closed_n / ear_seen) if ear_seen else None
        labels[vid] = label_of(vid)
        subjs[vid] = vid.split('_')[0]
        seen[vid] = n_seen
    print(f'有效视频 {len(perclos)}')

    tune = [v for v in videos if v in perclos and subjs[v] != '006']
    held = [v for v in videos if v in perclos and subjs[v] == '006']

    def evaluate(metric):
        def acc_at(th, vids):
            vals = [v for v in vids if metric[v] is not None]
            pred = [int(metric[v] >= th) for v in vals]
            lab = [labels[v] for v in vals]
            return sum(a == b for a, b in zip(pred, lab)) / len(vals), len(vals)
        best_th, best_acc = 0.4, 0.0
        for th in np.arange(0.1, 0.81, 0.05):
            a, _ = acc_at(th, tune)
            if a > best_acc:
                best_acc, best_th = a, th
        held_acc, n_held = acc_at(best_th, held)
        per_subj = {}
        for s in sorted(set(subjs.values())):
            vs = [v for v in metric if subjs[v] == s and metric[v] is not None]
            if vs:
                per_subj[s] = round(acc_at(best_th, vs)[0], 4)
        return {'threshold': round(best_th, 2), 'tune_video_acc': round(best_acc, 4),
                'heldout_006_video_acc': round(held_acc, 4), 'n_heldout': n_held,
                'per_subject_acc': per_subj}

    stu = evaluate(perclos)
    tea = evaluate(ear_pc)
    per_class = {}
    for lab_name, lab_val in [('alert', 0), ('drowsy', 1)]:
        vs = [perclos[v] for v in perclos if labels[v] == lab_val]
        per_class[lab_name] = {'n': len(vs),
                               'perclos_mean': round(float(np.mean(vs)), 3),
                               'perclos_std': round(float(np.std(vs)), 3)}

    report = {'protocol': '每视频100帧, 双通道PERCLOS, 阈值001/002/005搜索→006验证',
              'n_videos': len(perclos),
              'student': stu, 'teacher_ear': tea, 'per_class_perclos': per_class,
              'mean_frame_seen_frac': round(float(np.mean(list(seen.values()))) / N_FRAMES_PER_VIDEO, 3),
              'heldout_videos': {v: {'perclos': round(perclos[v], 3), 'label': labels[v],
                                     'correct': int(perclos[v] >= stu['threshold']) == labels[v]}
                                 for v in held}}
    json.dump(report, open(os.path.join(OUT, 'ext_test_nthu.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('报告:', json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
