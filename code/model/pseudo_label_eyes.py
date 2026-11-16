# 环节五/03f: EAR 教师伪标签 v2——MediaPipe FaceLandmarker(Tasks API) 眼区关键点 → 眼纵横比
# 输出连续 EAR 软标签（含模糊带），学生做回归蒸馏而不是硬二分类
# 为什么: 硬二分类让学生把眯眼/小眼纹理过拟合到闭眼类(DDD留来源外部验证清醒视频幻觉闭眼)；
#         EAR 连续值让学生学到眼睁开度的平滑函数，决策面与教师一致
# 环境: 需 mediapipe 1.x（无 3.13 wheel），用 D:/mp-env（Python 3.11）运行
# 输出:  data/fatigue/eye_rgb/*.jpg  +  eye_rgb_labels.jsonl (path,ear,src)
# 训练集: DDD 全部来源 + NTHU 受试者 001/002/005；006 保留做外部验证
import sys, os, json, glob, random, re
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
DDD_DIR = os.path.join(DATA, 'fatigue', 'ddd_raw', 'Driver Drowsiness Dataset (DDD)')
NTHU_DIR = os.path.join(DATA, 'fatigue', 'nthu-ddd', 'NTHU DDD')
OUT = os.path.join(DATA, 'fatigue', 'eye_rgb')
LABELS = os.path.join(DATA, 'fatigue', 'eye_rgb_labels.jsonl')
MODEL = r'D:\mp-env\face_landmarker.task'

EAR_OPEN = 0.26    # EAR ≥ 此值 → 睁
EAR_CLOSED = 0.20  # EAR ≤ 此值 → 闭；中间带弃用（模糊样本）
N_DDD_FRAMES = 12000   # DDD 抽帧上限（两类各半）
N_NTHU_PER_VIDEO = 150
VAL_SUBJECTS = {'006'}
N_MAX = 40000      # 总裁剪上限（含模糊带）

# 眼轮廓索引（与疲劳推理/裁剪一致）
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

opts = vision.FaceLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=MODEL),
                                    num_faces=1, running_mode=vision.RunningMode.IMAGE)
LANDMARKER = vision.FaceLandmarker.create_from_options(opts)


def ear_of(pts):
    """6点眼轮廓 → EAR = (|p2-p6|+|p3-p5|) / (2|p1-p4|)"""
    p1, p2, p3, p4, p5, p6 = pts
    def d(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    return (d(p2, p6) + d(p3, p5)) / (2.0 * d(p1, p4) + 1e-6)


def crop_eye(img, pts):
    """围绕6点轮廓外扩裁剪方形眼区（边长=2.2×轮廓宽），与推理端一致"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx, cy = sum(xs) / 6, sum(ys) / 6
    half = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.1
    x0, x1 = int(cx - half), int(cx + half)
    y0, y1 = int(cy - half), int(cy + half)
    H, W = img.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    return img[y0:y1, x0:x1]


def process(img):
    """一张图 → [(crop, ear) × ≤2 眼]；检测不到返回 []"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = LANDMARKER.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    out = []
    if not res.face_landmarks:
        return out
    lm = res.face_landmarks[0]
    H, W = img.shape[:2]
    for idxs in (LEFT_EYE, RIGHT_EYE):
        pts = [(lm[i].x * W, lm[i].y * H) for i in idxs]
        ear = ear_of(pts)
        crop = crop_eye(img, pts)
        if crop is not None and crop.shape[0] >= 12:
            out.append((crop, ear))
    return out


_CNT = [0]  # 全局单调计数（单目录，跨 emit() 不重置防覆盖）


def ddd_source_of(p):
    """DDD 来源视频前缀(A-Z/ZA-ZC/a-z/za-zc)，供留来源划分"""
    m = re.match(r'([A-Za-z]{1,2})\d+', os.path.basename(p))
    return m.group(1)


def save_crop(img, count):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'{count:07d}.jpg')
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(p)
    return p


def emit(records):
    """records: (crop, ear, src)；软标签=EAR 连续值（含模糊带）"""
    rows = []
    for crop, ear, src in records:
        if _CNT[0] >= N_MAX:
            break
        p = save_crop(crop, _CNT[0])
        _CNT[0] += 1
        rows.append({'path': p, 'ear': round(float(ear), 3), 'src': src})
    return rows


def main():
    random.seed(42)
    all_rows = []

    # ---- DDD: 帧级无标签，EAR 打标签；两类各抽一半 ----
    drowsy = sorted(glob.glob(os.path.join(DDD_DIR, 'Drowsy', '*.png')))
    alert = sorted(glob.glob(os.path.join(DDD_DIR, 'Non Drowsy', '*.png')))
    random.shuffle(drowsy)
    random.shuffle(alert)
    picks = drowsy[:N_DDD_FRAMES // 2] + alert[:N_DDD_FRAMES // 2]
    recs = []
    for p in picks:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for crop, ear in process(img):
            recs.append((crop, ear, f'ddd:{ddd_source_of(p)}'))
    all_rows += emit(recs)
    print(f'DDD: 处理 {len(picks)} 帧，累计标签 {len(all_rows)}', flush=True)

    # ---- NTHU: 受试者 001/002/005 打标签（006 留作外部验证） ----
    frames = sorted(glob.glob(os.path.join(NTHU_DIR, 'drowsy', '*.jpg')) +
                    glob.glob(os.path.join(NTHU_DIR, 'notdrowsy', '*.jpg')))
    recs = []
    by_video = {}
    for p in frames:
        vid = '_'.join(os.path.basename(p).split('_')[:3])
        by_video.setdefault(vid, []).append(p)
    n_before = len(all_rows)
    for vid, ps in sorted(by_video.items()):
        if vid.split('_')[0] in VAL_SUBJECTS:
            continue
        step = max(1, len(ps) // N_NTHU_PER_VIDEO)
        for p in ps[::step][:N_NTHU_PER_VIDEO]:
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            for crop, ear in process(img):
                recs.append((crop, ear, f'nthu:{vid}'))
    all_rows += emit(recs)
    train_subj = sorted({v.split('_')[0] for v in by_video})
    print(f'NTHU: {len(by_video)} 视频（训练受试者 {train_subj}，排除 {sorted(VAL_SUBJECTS)}），'
          f'新增 {len(all_rows) - n_before} 条', flush=True)

    with open(LABELS, 'w', encoding='utf-8') as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    ears = [r['ear'] for r in all_rows]
    print(f'总标签 {len(all_rows)}（EAR 均值 {np.mean(ears):.3f} / ≤0.20 占 '
          f'{np.mean([e <= 0.20 for e in ears]):.3f} / 模糊带(0.20~0.26) '
          f'{np.mean([0.20 < e < 0.26 for e in ears]):.3f}）→ {LABELS}')


if __name__ == '__main__':
    main()
