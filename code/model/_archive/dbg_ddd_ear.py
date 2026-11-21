# 诊断: 教师EAR-PERCLOS 在 DDD 视频级是否区分 drowsy/alert（决定外部验证换 DDD 留来源协议）
import sys, os, glob, random
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DDD_DIR = r'D:\大创项目实存\new_route\data\fatigue\ddd_raw\Driver Drowsiness Dataset (DDD)'
MODEL = r'D:\mp-env\face_landmarker.task'
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_CLOSED = 0.20

opts = vision.FaceLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=MODEL),
                                    num_faces=1, running_mode=vision.RunningMode.IMAGE)
lmk = vision.FaceLandmarker.create_from_options(opts)


def ear_of(pts):
    p1, p2, p3, p4, p5, p6 = pts
    def d(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    return (d(p2, p6) + d(p3, p5)) / (2.0 * d(p1, p4) + 1e-6)


def video_of(p):
    """视频ID=字母前缀(A-Z/ZA-ZC/a-z/za-zc)，如 A1234.png → A"""
    import re
    m = re.match(r'([A-Za-z]{1,2})\d+', os.path.basename(p))
    return m.group(1)


def main():
    random.seed(42)
    drowsy = sorted(glob.glob(os.path.join(DDD_DIR, 'Drowsy', '*.png')))
    alert = sorted(glob.glob(os.path.join(DDD_DIR, 'Non Drowsy', '*.png')))
    by_vid = {}
    for p in drowsy:
        by_vid.setdefault(video_of(p), []).append(p)
    for p in alert:
        by_vid.setdefault(video_of(p), []).append(p)
    drowsy_vids = sorted({video_of(p) for p in drowsy})
    alert_vids = sorted({video_of(p) for p in alert})
    print(f'视频: 困{len(drowsy_vids)}段 清醒{len(alert_vids)}段')

    res_rows = []
    for vid in sorted(by_vid):
        ps = by_vid[vid]
        step = max(1, len(ps) // 200)
        closed = seen = 0
        for p in ps[::step][:200]:
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if not res.face_landmarks:
                continue
            lm = res.face_landmarks[0]
            H, W = img.shape[:2]
            ears = []
            for idxs in (LEFT_EYE, RIGHT_EYE):
                pts = [(lm[i].x * W, lm[i].y * H) for i in idxs]
                ears.append(ear_of(pts))
            seen += 1
            if max(ears) <= 0.18:  # 双眼同闭(更严)，降单眼眨眼噪声
                closed += 1
        lab = 1 if vid in drowsy_vids else 0
        pc = closed / max(1, seen)
        res_rows.append((vid, lab, pc))
        print(f'{vid:>3} ({"困" if lab else "醒"}): PERCLOS={pc:.3f} ({closed}/{seen})')

    drowsy_pc = sorted(r[2] for r in res_rows if r[1] == 1)
    alert_pc = sorted(r[2] for r in res_rows if r[1] == 0)
    print(f'\n困: {[round(x,3) for x in drowsy_pc]}')
    print(f'醒: {[round(x,3) for x in alert_pc]}')
    # 阈值扫描: 视频级acc(全部)
    for th in np.arange(0.05, 0.41, 0.05):
        acc = sum(1 for _, l, pc in res_rows if int(pc >= th) == l) / len(res_rows)
        print(f'th={th:.2f} 视频级acc={acc:.3f}')


if __name__ == '__main__':
    main()
