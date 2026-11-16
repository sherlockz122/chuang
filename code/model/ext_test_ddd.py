# 环节五/04c: 外部测试(正式)——学生CNN+PERCLOS vs 教师EAR+PERCLOS 在 DDD 留来源视频级准确率
# 协议(留来源): 训练=其余来源+NTHU-001/002/005；留出 6困+6醒 来源视频(见 ddd_split.py)
#              每段视频均匀抽 200 帧 → 逐帧眼态(学生/教师) → 视频级PERCLOS → 阈值判困/醒
#              阈值在训练来源视频上搜索，留出来源上验证
# 帧级口径: 教师=双眼EAR≤0.18为闭; 学生=双眼预测EAR≤0.20为闭（严格口径，降单眼眨眼噪声）
# 环境: D:/mp-env/Scripts/python.exe
# 输出:  ext_test_ddd.json
import sys, os, json, glob, re
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from collections import defaultdict
from fatigue_infer import load_image, frame_eye_states
from ddd_split import heldout_sources

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'model')
DDD_DIR = os.path.join(DATA, 'fatigue', 'ddd_raw', 'Driver Drowsiness Dataset (DDD)')
N_FRAMES_PER_VIDEO = 200
EAR_CLOSED = 0.18   # 教师口径（严格: 双眼同闭）
SCORE_CLOSED = 0.20  # 学生口径: 预测EAR≤0.20（与教师闭眼界一致）


def source_of(p):
    m = re.match(r'([A-Za-z]{1,2})\d+', os.path.basename(p))
    return m.group(1)


def main():
    drowsy = sorted(glob.glob(os.path.join(DDD_DIR, 'Drowsy', '*.png')))
    alert = sorted(glob.glob(os.path.join(DDD_DIR, 'Non Drowsy', '*.png')))
    drowsy_src = {source_of(p) for p in drowsy}
    held = heldout_sources()
    print(f'来源: 困{len(drowsy_src)}段 醒{len({source_of(p) for p in alert})}段；留出 {sorted(held)}')

    by_vid = defaultdict(list)
    for p in drowsy + alert:
        by_vid[source_of(p)].append(p)
    videos = sorted(by_vid)

    stu_pc, tea_pc, labels = {}, {}, {}
    for vid in videos:
        ps = by_vid[vid]
        step = max(1, len(ps) // N_FRAMES_PER_VIDEO)
        sample = ps[::step][:N_FRAMES_PER_VIDEO]
        s_closed = t_closed = n_seen = 0
        for p in sample:
            img = load_image(p)
            if img is None:
                continue
            states = frame_eye_states(img)
            if states is None:
                continue
            n_seen += 1
            pops = [p_o for p_o, _ in states]
            ears = [e for _, e in states if e is not None]
            if len(pops) >= 2 and all(p_o <= SCORE_CLOSED for p_o in pops):
                s_closed += 1
            if len(ears) >= 2 and all(e <= EAR_CLOSED for e in ears):
                t_closed += 1
        if n_seen == 0:
            continue
        stu_pc[vid] = s_closed / n_seen
        tea_pc[vid] = t_closed / n_seen
        labels[vid] = 1 if vid in drowsy_src else 0
    print(f'有效来源 {len(stu_pc)}')

    tune = [v for v in videos if v in stu_pc and v not in held]
    heldv = [v for v in videos if v in stu_pc and v in held]

    def evaluate(metric):
        def acc_at(th, vids):
            pred = [int(metric[v] >= th) for v in vids]
            lab = [labels[v] for v in vids]
            return sum(a == b for a, b in zip(pred, lab)) / len(vids)
        best_th, best_acc = 0.1, 0.0
        for th in np.arange(0.02, 0.51, 0.02):
            a = acc_at(th, tune)
            if a > best_acc:
                best_acc, best_th = a, th
        return {'threshold': round(best_th, 3), 'tune_video_acc': round(best_acc, 4),
                'heldout_video_acc': round(acc_at(best_th, heldv), 4),
                'heldout_n': len(heldv),
                'heldout_detail': {v: {'perclos': round(metric[v], 3), 'label': labels[v],
                                       'correct': int(metric[v] >= best_th) == labels[v]}
                                   for v in heldv}}

    stu = evaluate(stu_pc)
    tea = evaluate(tea_pc)
    # 学生-教师视频级一致性（蒸馏保真度）
    vs = [v for v in stu_pc if v in tea_pc]
    corr = float(np.corrcoef([stu_pc[v] for v in vs], [tea_pc[v] for v in vs])[0, 1])
    report = {'protocol': 'DDD留来源(6困+6醒), 每视频200帧, 双通道PERCLOS, 阈值训练来源搜索→留出验证',
              'n_videos': len(stu_pc), 'heldout_sources': sorted(held),
              'student': stu, 'teacher_ear': tea,
              'student_teacher_corr': round(corr, 4),
              'per_class_perclos': {
                  'alert': {'perclos_mean': round(float(np.mean([stu_pc[v] for v in stu_pc if labels[v] == 0])), 3)},
                  'drowsy': {'perclos_mean': round(float(np.mean([stu_pc[v] for v in stu_pc if labels[v] == 1])), 3)}}}
    json.dump(report, open(os.path.join(OUT, 'ext_test_ddd.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('报告:', json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
