# 环节五/03b: 疲劳分类器 v2——DDD 数据集（41.8k 张 227×227 人脸裁剪，54 段来源视频）
# 与 v1(NTHU 4人镜像) 的差别: 受试者/视频多 → 按来源视频留出验证（跨人泛化）
# 输入:  ../data/fatigue/ddd_raw/Driver Drowsiness Dataset (DDD)/{Drowsy,Non Drowsy}/*.png
# 输出:  ../data/model/fatigue_head.pt + fatigue_report.json（覆盖 v1）
# 口径: 前缀=来源视频（单类标注）→ 每类留出 4 段视频做验证；帧级二分类 → safety_level
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
OUT = os.path.join(DATA, 'model')
IMG_DIR = os.path.join(DATA, 'fatigue', 'ddd_raw', 'Driver Drowsiness Dataset (DDD)')
random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_VAL_SRC_PER_CLASS = 4  # 每类留出 4 段来源视频（~15% 图像量）


def prefix_of(p):
    m = re.match(r'([A-Za-z]{1,2})[0-9]+', os.path.basename(p))
    return m.group(1)


class DddDataset(Dataset):
    """DDD 已是人脸裁剪图，直接 resize 进模型，无需 Haar"""

    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.tf = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB')), self.labels[i]


def main():
    drowsy = sorted(glob.glob(os.path.join(IMG_DIR, 'Drowsy', '*.png')) +
                    glob.glob(os.path.join(IMG_DIR, 'Drowsy', '*.jpg')))
    alert = sorted(glob.glob(os.path.join(IMG_DIR, 'Non Drowsy', '*.png')) +
                   glob.glob(os.path.join(IMG_DIR, 'Non Drowsy', '*.jpg')))
    print(f'帧数: drowsy {len(drowsy)} / alert {len(alert)}')

    # 按来源视频留出：每类随机抽 N 段视频做验证，其余训练
    src_d = sorted(set(prefix_of(p) for p in drowsy))
    src_a = sorted(set(prefix_of(p) for p in alert))
    val_src = set(random.sample(src_d, N_VAL_SRC_PER_CLASS)) | \
              set(random.sample(src_a, N_VAL_SRC_PER_CLASS))
    tr_d = [p for p in drowsy if prefix_of(p) not in val_src]
    tr_a = [p for p in alert if prefix_of(p) not in val_src]
    va_d = [p for p in drowsy if prefix_of(p) in val_src]
    va_a = [p for p in alert if prefix_of(p) in val_src]
    print(f'来源视频 {len(src_d)}+{len(src_a)}，留出验证 {sorted(val_src)}；'
          f'训练 {len(tr_d)}+{len(tr_a)}，验证 {len(va_d)}+{len(va_a)}')

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
        pred = logits.argmax(1)
        tn = ((pred == 0) & (yv == 0)).sum().item()
        tp = ((pred == 1) & (yv == 1)).sum().item()
        fp = ((pred == 1) & (yv == 0)).sum().item()
        fn = ((pred == 0) & (yv == 1)).sum().item()
        # 按来源视频的准确率分布（每段视频一个数，看泛化是否均匀）
        per_src = defaultdict(lambda: [0, 0])
        va_paths = va_d + va_a
        for i, p in enumerate(va_paths):
            per_src[prefix_of(p)][0] += 1
            per_src[prefix_of(p)][1] += int(pred[i] == yv[i])
    torch.save({'head': head.state_dict(), 'backbone': 'mobilenet_v2',
                'dataset': 'DDD', 'val_sources': sorted(val_src)},
               os.path.join(OUT, 'fatigue_head.pt'))
    report = {'dataset': 'DDD', 'val_acc': round(acc, 4),
              'n_train': len(Xtr), 'n_val': len(Xv),
              'n_sources': len(src_d) + len(src_a), 'val_sources': sorted(val_src),
              'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
              'recall_drowsy': round(tp / (tp + fn), 4),
              'precision_drowsy': round(tp / (tp + fp), 4),
              'per_source_acc': {k: round(v[1] / v[0], 4) for k, v in sorted(per_src.items())}}
    json.dump(report, open(os.path.join(OUT, 'fatigue_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('报告:', json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
