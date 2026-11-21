# 诊断实验: 冻结特征上线性探针(1280→2) vs 小头(1280→64→2)+强正则 vs 原2层头
# 同一样本/划分/训练流程，量化"分类头容量"对跨来源泛化的影响
import sys, os, json, glob, re, random
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
IMG_DIR = os.path.join(DATA, 'fatigue', 'ddd_raw', 'Driver Drowsiness Dataset (DDD)')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)
torch.manual_seed(42)


def prefix_of(p):
    m = re.match(r'([A-Za-z]{1,2})[0-9]+', os.path.basename(p))
    return m.group(1)


class DddDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.tf = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB')), self.labels[i]


def main():
    drowsy = sorted(glob.glob(os.path.join(IMG_DIR, 'Drowsy', '*.png')))
    alert = sorted(glob.glob(os.path.join(IMG_DIR, 'Non Drowsy', '*.png')))
    src_d = sorted(set(prefix_of(p) for p in drowsy))
    src_a = sorted(set(prefix_of(p) for p in alert))
    val_src = set(random.sample(src_d, 4)) | set(random.sample(src_a, 4))
    tr_d = [p for p in drowsy if prefix_of(p) not in val_src]
    tr_a = [p for p in alert if prefix_of(p) not in val_src]
    va_d = [p for p in drowsy if prefix_of(p) in val_src]
    va_a = [p for p in alert if prefix_of(p) in val_src]
    print(f'留出验证来源 {sorted(val_src)}；训练 {len(tr_d)}+{len(tr_a)}，验证 {len(va_d)}+{len(va_a)}')

    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tr_dl = DataLoader(DddDataset(tr_d + tr_a, [1] * len(tr_d) + [0] * len(tr_a), tf),
                       batch_size=64, shuffle=True, num_workers=0)
    va_dl = DataLoader(DddDataset(va_d + va_a, [1] * len(va_d) + [0] * len(va_a), tf),
                       batch_size=64, num_workers=0)

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

    print('提取特征...')
    Xtr, ytr = featurize(tr_dl)
    Xv, yv = featurize(va_dl)

    def train_eval(head, tag, epochs=40, lr=1e-3, wd=1e-4, bs=512, dropout=0.0):
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
        n0, n1 = (ytr == 0).sum().item(), (ytr == 1).sum().item()
        w = torch.tensor([(n0 + n1) / (2 * n0), (n0 + n1) / (2 * n1)])
        lossf = nn.CrossEntropyLoss(weight=w)
        best_acc = 0.0
        for epoch in range(epochs):
            head.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                lossf(head(Xtr[idx]), ytr[idx]).backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                logits = head(Xv)
                acc = (logits.argmax(1) == yv).float().mean().item()
                pred = logits.argmax(1)
                recall = ((pred == 1) & (yv == 1)).sum().item() / max(1, (yv == 1).sum().item())
            if acc > best_acc:
                best_acc = acc
                best_recall = recall
        print(f'{tag}: 最优 val_acc={best_acc:.4f}  recall_drowsy={best_recall:.4f}')
        return best_acc

    # 1) 线性探针（容量最小，记不住身份）
    train_eval(nn.Linear(1280, 2), '线性探针 1280→2', epochs=60, lr=1e-2, wd=1e-2)
    # 2) 小头 + 强正则
    train_eval(nn.Sequential(nn.Linear(1280, 64), nn.ReLU(), nn.Linear(64, 2)),
               '小头 1280→64→2 wd=1e-2', epochs=60, lr=1e-3, wd=1e-2)
    # 3) 原 2 层头对照
    train_eval(nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)),
               '原头 1280→256→2', epochs=40, lr=1e-3, wd=1e-4)


if __name__ == '__main__':
    main()
