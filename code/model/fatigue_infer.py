# 环节五/04: 疲劳推理 v5——FaceLandmarker 眼区裁剪 + RGB 学生CNN(EAR回归蒸馏) → PERCLOS → safety_level
# 管线: 帧 → FaceLandmarker 眼区精确裁剪(兜底: Haar脸+Haar眼) → EyeNet-RGB 预测EAR → 时间窗闭合率
# 环境: mediapipe 1.x 无 Python3.13 wheel，用 D:/mp-env/Scripts/python.exe 运行（torch CPU）
# 用法:  python fatigue_infer.py <image_or_dir>
# 输出:  {"prob_drowsy": 0.xx, "safety_level": 0|1|2, "n_frames": N, "perclos": 0.xx,
#         "ear_perclos": 0.xx(目录模式, 教师EAR对照)}
# 映射口径: PERCLOS<0.2 清醒(0) / 0.2~0.4 低警惕(1) / >0.4 疲劳(2)
import sys, os, json, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'model')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TF = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
FACE_CAS = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
EYE_CAS = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
LM_MODEL = r'D:\mp-env\face_landmarker.task'
EAR_OPEN, EAR_CLOSED = 0.26, 0.20  # 教师阈值（仅用于对照输出）

PERCLOS_L1, PERCLOS_L2 = 0.2, 0.4

# 眼轮廓索引（与 pseudo_label_eyes.py 一致）
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

_model = None
_lm = None


class EyeNet(nn.Module):
    """与 train_eye_rgb.py 同架构（64×64 输入，1 输出回归EAR，~9.4万参数）"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def load():
    """RGB 回归学生模型（eye_rgb.pt）"""
    global _model
    if _model is None:
        _model = EyeNet().to(DEVICE)
        ckpt = torch.load(os.path.join(OUT, 'eye_rgb.pt'), map_location='cpu', weights_only=True)
        _model.load_state_dict(ckpt['model'])
        _model.eval()
    return _model


def _landmarker():
    global _lm
    if _lm is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LM_MODEL),
            num_faces=1, running_mode=vision.RunningMode.IMAGE)
        _lm = vision.FaceLandmarker.create_from_options(opts)
    return _lm


def load_image(path):
    """cv2.imread 不支持中文路径，用 fromfile+imdecode"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _ear_of(pts):
    p1, p2, p3, p4, p5, p6 = pts
    def d(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    return (d(p2, p6) + d(p3, p5)) / (2.0 * d(p1, p4) + 1e-6)


def landmark_eyes(img):
    """FaceLandmarker 主路径 → [(crop, ear) × ≤2]；无脸/失败 → []"""
    try:
        import mediapipe as mp
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    except Exception:
        return []
    if not res.face_landmarks:
        return []
    lm = res.face_landmarks[0]
    H, W = img.shape[:2]
    crops = []
    for idxs in (LEFT_EYE, RIGHT_EYE):
        pts = [(lm[i].x * W, lm[i].y * H) for i in idxs]
        ear = _ear_of(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx, cy = sum(xs) / 6, sum(ys) / 6
        half = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.1
        x0, x1 = int(cx - half), int(cx + half)
        y0, y1 = int(cy - half), int(cy + half)
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        crops.append((img[y0:y1, x0:x1], ear))
    return crops


def detect_face(gray):
    faces = FACE_CAS.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (x, y, w, h)


def haar_eyes(face_gray):
    """兜底: eye cascade，检测不到则几何裁剪(上25%~55%高度分左右两半)"""
    eyes = EYE_CAS.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 12))
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        return [face_gray[y:y + h, x:x + w] for x, y, w, h in eyes]
    if len(eyes) == 1:
        x, y, w, h = eyes[0]
        return [face_gray[y:y + h, x:x + w]]
    H, W = face_gray.shape
    band = face_gray[int(H * 0.25):int(H * 0.55), :]
    return [band[:, :W // 2], band[:, W // 2:]]


def _predict(eye_bgr):
    """回归学生: 输出 ŷ=sigmoid(logit)≈预测EAR（眼睁开度连续值）"""
    eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)  # 与训练一致: PIL 读 JPEG 得 RGB
    xb = TF(Image.fromarray(eye_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return float(torch.sigmoid(load()(xb)).item())


def frame_eye_states(img):
    """单帧: FaceLandmarker → [(y_hat, ear教师) × ≤2 眼]；失败降级 Haar 链；无→None"""
    mps = landmark_eyes(img)
    if mps:
        return [(_predict(c), e) for c, e in mps]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fb = detect_face(gray)
    if fb is None:
        return None
    x, y, w, h = fb
    x0, y0 = max(0, int(x - w * 0.05)), max(0, int(y - h * 0.05))
    x1, y1 = min(gray.shape[1], int(x + w * 1.05)), min(gray.shape[0], int(y + h * 1.05))
    face_gray = gray[y0:y1, x0:x1]
    eyes = [cv2.cvtColor(e, cv2.COLOR_GRAY2BGR) for e in haar_eyes(face_gray)]
    return [(_predict(e), None) for e in eyes]


def frame_closed(img):
    """单帧是否判定闭眼（任一只眼预测EAR≤0.20 即闭，教师口径）"""
    states = frame_eye_states(img)
    if states is None:
        return None
    return any(p <= EAR_CLOSED for p, _ in states)


def safety_level(perclos):
    if perclos < PERCLOS_L1:
        return 0
    if perclos < PERCLOS_L2:
        return 1
    return 2


def infer(path):
    """path: 单图 or 目录（目录=时间窗内帧序列 → PERCLOS）"""
    if os.path.isdir(path):
        frames = sorted(glob.glob(os.path.join(path, '*.jpg')) +
                        glob.glob(os.path.join(path, '*.png')))[:300]
        if not frames:
            return {'error': '目录内无 jpg/png'}
        closed = seen = 0
        ear_closed = ear_seen = 0
        for f in frames:
            img = load_image(f)
            if img is None:
                continue
            states = frame_eye_states(img)
            if states is None:
                continue
            seen += 1
            if any(p <= EAR_CLOSED for p, _ in states):
                closed += 1
            if any(e is not None for _, e in states):
                ear_seen += 1
                if any(e is not None and e <= EAR_CLOSED for _, e in states):
                    ear_closed += 1
        if seen == 0:
            return {'error': '未检出人脸'}
        perclos = closed / seen
        out = {'prob_drowsy': round(perclos, 3), 'perclos': round(perclos, 3),
               'safety_level': safety_level(perclos),
               'n_frames': seen, 'n_closed': closed}
        if ear_seen:
            ep = ear_closed / ear_seen
            out['ear_perclos'] = round(ep, 3)
            out['ear_safety_level'] = safety_level(ep)
        return out
    img = load_image(path)
    if img is None:
        return {'error': '无法读取图像'}
    c = frame_closed(img)
    return {'eye_closed': c, 'safety_level': safety_level(1.0 if c else 0.0),
            'prob_drowsy': 1.0 if c else 0.0, 'n_frames': 1}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python fatigue_infer.py <image_or_dir>')
        sys.exit(1)
    print(json.dumps(infer(sys.argv[1]), ensure_ascii=False))
