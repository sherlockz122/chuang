# 环节五/03: 疲劳状态分类器（NTHU-DDD 帧级二分类 → safety_level 信号源）
# 做法: 冻结 MobileNetV2 主干提取 1280 维特征 → 轻量 MLP 头（CPU 可跑）
# 输入:  ../data/fatigue/nthu-ddd/NTHU DDD/{drowsy,notdrowsy}/*.jpg
# 输出:  ../data/model/fatigue_head.pt + fatigue_report.json
# 口径: 按受试者划分（文件名前缀 001_...），防泄漏；分类概率 → safety_level 映射
import sys, os, json, glob, random
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'model')
IMG_DIR = os.path.join(DATA, 'fatigue', 'nthu-ddd', 'NTHU DDD')
random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 每类抽样上限（GPU 上可全量；CPU 回退时限制规模）
N_PER_CLASS = 36000 if DEVICE.type == 'cuda' else 8000
N_VAL_PER_CLASS = 8000 if DEVICE.type == 'cuda' else 1600

# 人脸检测：每位受试者镜头位置固定 → 采样帧检测取中位框，全量按框裁剪（Haar 全量 66k 帧太慢）
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


def detect_face_box(path):
    # cv2.imread 不支持中文路径，用 fromfile+imdecode
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (x, y, w, h)


def build_subject_crops(paths_by_subject):
    """返回 {subject: (x,y,w,h)}；检测不到的受试者回退中心裁剪"""
    crops = {}
    for subj, paths in paths_by_subject.items():
        boxes = []
        for p in random.sample(paths, min(20, len(paths))):
            b = detect_face_box(p)
            if b:
                boxes.append(b)
        if boxes:
            boxes = np.array(boxes)
            med = np.median(boxes, axis=0).astype(int)
            x, y, w, h = med
            # 外扩 1.5× 留额头/下巴，再按宽高取方形
            cx, cy = x + w / 2, y + h / 2
            s = int(max(w, h) * 1.5)
            x0, y0 = int(cx - s / 2), int(cy - s / 2)
            crops[subj] = (max(0, x0), max(0, y0), s, s)
        else:
            crops[subj] = None  # 回退中心裁剪
        print(f'  受试者 {subj}: {crops[subj]}（检测 {len(boxes)}/20 帧）')
    return crops


class FrameDataset(Dataset):
    def __init__(self, paths, labels, transform, subject_crops, img_size=(640, 480)):
        self.paths, self.labels, self.tf = paths, labels, transform
        self.crops, self.img_size = subject_crops, img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        box = self.crops[subject_of(self.paths[i])]
        if box is None:  # 中心裁剪回退（脸部位于画面中央）
            W, H = img.size
            img = img.crop((W * 0.2, H * 0.05, W * 0.8, H * 0.85))
        else:
            x, y, w, h = box
            W, H = img.size
            img = img.crop((min(x, W - 1), min(y, H - 1),
                            min(x + w, W), min(y + h, H)))
        return self.tf(img), self.labels[i]


def subject_of(p):
    base = os.path.basename(p)
    return base.split('_')[0]


def main():
    drowsy = sorted(glob.glob(os.path.join(IMG_DIR, 'drowsy', '*.jpg')))
    alert = sorted(glob.glob(os.path.join(IMG_DIR, 'notdrowsy', '*.jpg')))
    print(f'帧数: drowsy {len(drowsy)} / alert {len(alert)}')

    # 按受试者划分（留出法：后 20% 受试者做验证）
    drowsy_s = sorted(set(subject_of(p) for p in drowsy))
    alert_s = sorted(set(subject_of(p) for p in alert))
    subjects = sorted(set(drowsy_s) & set(alert_s))
    n_val_subj = max(1, len(subjects) // 5)
    val_subj = set(subjects[-n_val_subj:])
    tr_d = [p for p in drowsy if subject_of(p) not in val_subj]
    tr_a = [p for p in alert if subject_of(p) not in val_subj]
    va_d = [p for p in drowsy if subject_of(p) in val_subj]
    va_a = [p for p in alert if subject_of(p) in val_subj]
    tr_d, tr_a = random.sample(tr_d, min(N_PER_CLASS, len(tr_d))), random.sample(tr_a, min(N_PER_CLASS, len(tr_a)))
    va_d, va_a = random.sample(va_d, min(N_VAL_PER_CLASS, len(va_d))), random.sample(va_a, min(N_VAL_PER_CLASS, len(va_a)))
    print(f'受试者 {len(subjects)}，验证集 {len(val_subj)} 人；训练 {len(tr_d)}+{len(tr_a)}，验证 {len(va_d)}+{len(va_a)}')

    # 按受试者估计人脸框（每位受试者的镜头位置固定）
    by_subj = {}
    for p in tr_d + tr_a + va_d + va_a:
        by_subj.setdefault(subject_of(p), []).append(p)
    crops = build_subject_crops(by_subj)

    tf = transforms.Compose([transforms.Resize(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tr_ds = FrameDataset(tr_d + tr_a, [1] * len(tr_d) + [0] * len(tr_a), tf, crops)
    va_ds = FrameDataset(va_d + va_a, [1] * len(va_d) + [0] * len(va_a), tf, crops)
    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=64, num_workers=0)

    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    backbone.classifier = nn.Identity()
    backbone.to(DEVICE).eval()

    @torch.no_grad()
    def featurize(dl):
        xs, ys = [], []
        for xb, yb in dl:
            xs.append(backbone(xb.to(DEVICE)).cpu())
            ys.append(yb)
        return torch.cat(xs), torch.cat(ys)

    print(f'设备: {DEVICE}，提取训练特征...')
    Xtr, ytr = featurize(tr_dl)
    print('提取验证特征...')
    Xv, yv = featurize(va_dl)

    head = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    n0, n1 = (ytr == 0).sum().item(), (ytr == 1).sum().item()
    w = torch.tensor([(n0 + n1) / (2 * n0), (n0 + n1) / (2 * n1)])  # 类别均衡
    lossf = nn.CrossEntropyLoss(weight=w)
    for epoch in range(40):
        head.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = lossf(head(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        logits = head(Xv)
        acc = (logits.argmax(1) == yv).float().mean().item()
        probs = torch.softmax(logits, 1)
        # 三类混淆统计（清醒=0 正确 / 困=1 正确 / 误报 / 漏报）
        tn = ((probs.argmax(1) == 0) & (yv == 0)).sum().item()
        tp = ((probs.argmax(1) == 1) & (yv == 1)).sum().item()
        fp = ((probs.argmax(1) == 1) & (yv == 0)).sum().item()
        fn = ((probs.argmax(1) == 0) & (yv == 1)).sum().item()
    torch.save({'head': head.state_dict(), 'backbone': 'mobilenet_v2'},
               os.path.join(OUT, 'fatigue_head.pt'))
    report = {'val_acc': round(acc, 4), 'n_train': len(Xtr), 'n_val': len(Xv),
              'subjects': len(subjects), 'val_subjects': len(val_subj),
              'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
              'recall_drowsy': round(tp / (tp + fn), 4), 'precision_drowsy': round(tp / (tp + fp), 4)}
    json.dump(report, open(os.path.join(OUT, 'fatigue_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('报告:', json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
