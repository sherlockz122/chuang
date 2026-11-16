# 调试: NTHU 帧的眼区裁剪长什么样 + EyeNet 对每只眼输出什么
# 输出: data/model/_dbg/*.png（人脸/眼区裁剪）+ 终端 p_open
import sys, os, glob
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
from fatigue_infer import load, load_image, detect_face, eye_regions, TF
from PIL import Image
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
DBG = os.path.join(DATA, 'model', '_dbg')
os.makedirs(DBG, exist_ok=True)


def save_png(path, img):
    """cv2.imwrite 不支持中文路径，用 imencode+tofile"""
    ok, buf = cv2.imencode('.png', img)
    if ok:
        buf.tofile(path)
IMG_DIR = os.path.join(DATA, 'fatigue', 'nthu-ddd', 'NTHU DDD')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

frames = sorted(glob.glob(os.path.join(IMG_DIR, 'drowsy', '001_glasses_slowBlink*.jpg')) +
                glob.glob(os.path.join(IMG_DIR, 'drowsy', '001_glasses_sleepy*.jpg')))
print(f'待检帧 {len(frames)}')
model = load()
step = max(1, len(frames) // 6)
for i, p in enumerate(frames[::step][:6]):
    img = load_image(p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fb = detect_face(gray)
    if fb is None:
        print(f'{os.path.basename(p)}: 无脸')
        continue
    x, y, w, h = fb
    x0, y0 = max(0, int(x - w * 0.05)), max(0, int(y - h * 0.05))
    x1, y1 = min(gray.shape[1], int(x + w * 1.05)), min(gray.shape[0], int(y + h * 1.05))
    face_gray = gray[y0:y1, x0:x1]
    save_png(os.path.join(DBG, f'face_{i}.png'), face_gray)
    probs = []
    for j, eye in enumerate(eye_regions(face_gray)):
        save_png(os.path.join(DBG, f'eye_{i}_{j}.png'), eye)
        eye_rgb = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
        xb = TF(Image.fromarray(eye_rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p_open = float(torch.softmax(model(xb), 1)[0, 1].item())
        probs.append(round(p_open, 3))
    print(f'{os.path.basename(p)} 脸{face_gray.shape} 眼区{len(probs)}个 p_open={probs}')
print(f'调试图已存 {DBG}')
