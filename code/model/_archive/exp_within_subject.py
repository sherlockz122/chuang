# 诊断实验: 同受试者内 80/20 划分（故意泄漏，只用于定位瓶颈）
# 结论口径: 若 acc 高 → 瓶颈=跨人泛化（受试者太少）；若 acc 仍低 → 帧级任务本身不可学（需换眼睛级数据）
import sys, os, glob, random
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from train_fatigue import detect_face_box, subject_of, build_subject_crops, FrameDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
IMG_DIR = os.path.join(DATA, 'fatigue', 'nthu-ddd', 'NTHU DDD')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)
torch.manual_seed(42)

drowsy = sorted(glob.glob(os.path.join(IMG_DIR, 'drowsy', '*.jpg')))
alert = sorted(glob.glob(os.path.join(IMG_DIR, 'notdrowsy', '*.jpg')))
N_PER_CLASS = 12000  # 诊断用，少抽一点

tr_d, va_d = [], []
for subj in sorted(set(subject_of(p) for p in drowsy)):
    ps = [p for p in drowsy if subject_of(p) == subj]
    random.shuffle(ps)
    n = int(len(ps) * 0.8)
    tr_d += ps[:n]
    va_d += ps[n:]
tr_a, va_a = [], []
for subj in sorted(set(subject_of(p) for p in alert)):
    ps = [p for p in alert if subject_of(p) == subj]
    random.shuffle(ps)
    n = int(len(ps) * 0.8)
    tr_a += ps[:n]
    va_a += ps[n:]
tr_d = random.sample(tr_d, min(N_PER_CLASS, len(tr_d)))
tr_a = random.sample(tr_a, min(N_PER_CLASS, len(tr_a)))
print(f'训练 {len(tr_d)}+{len(tr_a)}，验证 {len(va_d)}+{len(va_a)}')

by_subj = {}
for p in tr_d + tr_a + va_d + va_a:
    by_subj.setdefault(subject_of(p), []).append(p)
crops = build_subject_crops(by_subj)

tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
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

print(f'设备: {DEVICE}，提取特征...')
Xtr, ytr = featurize(tr_dl)
Xv, yv = featurize(va_dl)

head = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
n0, n1 = (ytr == 0).sum().item(), (ytr == 1).sum().item()
w = torch.tensor([(n0 + n1) / (2 * n0), (n0 + n1) / (2 * n1)])
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
    pred = logits.argmax(1)
    tn = ((pred == 0) & (yv == 0)).sum().item()
    tp = ((pred == 1) & (yv == 1)).sum().item()
    fp = ((pred == 1) & (yv == 0)).sum().item()
    fn = ((pred == 0) & (yv == 1)).sum().item()
print(f'诊断(同受试者划分): val_acc={acc:.4f}  tn={tn} fp={fp} fn={fn} tp={tp}  '
      f'recall_drowsy={tp/(tp+fn):.4f}  precision={tp/(tp+fp):.4f}')
