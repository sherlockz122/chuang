# 诊断: NTHU 视频上管线三路径占比（无脸 / cascade眼 / 几何兜底）+ 真眼 p_open 分布
import sys, os, glob
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
from collections import Counter
from fatigue_infer import load, load_image, detect_face, eye_regions, TF
from PIL import Image
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
IMG_DIR = os.path.join(DATA, 'fatigue', 'nthu-ddd', 'NTHU DDD')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EYE_CAS = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = load()

for tag, vidpat in [('清醒', '001_glasses_nonsleepyCombination'),
                    ('困', '001_glasses_sleepyCombination'),
                    ('困', '001_glasses_slowBlinkWithNodding')]:
    frames = sorted(glob.glob(os.path.join(IMG_DIR, 'drowsy', f'{vidpat}_*.jpg')) +
                    glob.glob(os.path.join(IMG_DIR, 'notdrowsy', f'{vidpat}_*.jpg')))
    step = max(1, len(frames) // 100)
    sample = frames[::step][:100]
    path = Counter()
    p_open_all = []
    for p in sample:
        img = load_image(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fb = detect_face(gray)
        if fb is None:
            path['no_face'] += 1
            continue
        x, y, w, h = fb
        x0, y0 = max(0, int(x - w * 0.05)), max(0, int(y - h * 0.05))
        x1, y1 = min(gray.shape[1], int(x + w * 1.05)), min(gray.shape[0], int(y + h * 1.05))
        face_gray = gray[y0:y1, x0:x1]
        eyes = EYE_CAS.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 12))
        regions = eye_regions(face_gray)
        small = [e for e in regions if max(e.shape[:2]) < 80]
        if len(small) >= 2:
            path['cascade_2'] += 1
        elif len(small) == 1:
            path['cascade_1'] += 1
        else:
            path['fallback'] += 1
        for eye in regions:
            if max(eye.shape[:2]) >= 80:
                continue  # 兜底大块不算真眼
            eye_rgb = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
            xb = TF(Image.fromarray(eye_rgb)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p_open_all.append(float(torch.softmax(model(xb), 1)[0, 1].item()))
    po = np.array(p_open_all) if p_open_all else np.array([np.nan])
    print(f'{vidpat}  ({tag}):')
    print(f'  路径: {dict(path)}')
    print(f'  真眼p_open: n={len(po)} mean={po.mean():.3f} p<0.3占比={(po < 0.3).mean():.3f} '
          f'p<0.5占比={(po < 0.5).mean():.3f}')
