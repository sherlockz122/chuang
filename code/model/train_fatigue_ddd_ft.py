# 环节五/03d: 疲劳分类器 v3——DDD 端到端微调 + 增强（打破"来源视频身份"记忆）
# 与 03b 的差别: 解冻 MobileNetV2 后 2 个 stage 联合训练；RandomResizedCrop/翻转/颜色抖动；
#               类别均衡采样器；fp16 autocast 加速
# 输出:  ../data/model/fatigue_head.pt（含 backbone state，fatigue_infer 兼容加载）
import sys, os, json, glob, re, random
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'model')
IMG_DIR = os.path.join(DATA, 'fatigue', 'ddd_raw', 'Driver Drowsiness Dataset (DDD)')
random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_VAL_SRC_PER_CLASS = 4
EPOCHS = 6
BS = 32


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


TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.25, 0.25, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
VAL_TF = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


@torch.no_grad()
def evaluate(model, va_dl, va_paths, val_src):
    model.eval()
    preds, ys = [], []
    for xb, yb in va_dl:
        logits = model(xb.to(DEVICE))
        preds.append(logits.argmax(1).cpu())
        ys.append(yb)
    pred, y = torch.cat(preds), torch.cat(ys)
    acc = (pred == y).float().mean().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    per_src = {s: [0, 0] for s in val_src}
    for i, p in enumerate(va_paths):
        per_src[prefix_of(p)][0] += 1
        per_src[prefix_of(p)][1] += int(pred[i] == y[i])
    return acc, (tn, fp, fn, tp), {k: round(v[1] / v[0], 4) for k, v in per_src.items()}


def main():
    drowsy = sorted(glob.glob(os.path.join(IMG_DIR, 'Drowsy', '*.png')) +
                    glob.glob(os.path.join(IMG_DIR, 'Drowsy', '*.jpg')))
    alert = sorted(glob.glob(os.path.join(IMG_DIR, 'Non Drowsy', '*.png')) +
                   glob.glob(os.path.join(IMG_DIR, 'Non Drowsy', '*.jpg')))
    src_d = sorted(set(prefix_of(p) for p in drowsy))
    src_a = sorted(set(prefix_of(p) for p in alert))
    val_src = set(random.sample(src_d, N_VAL_SRC_PER_CLASS)) | \
              set(random.sample(src_a, N_VAL_SRC_PER_CLASS))
    tr_d = [p for p in drowsy if prefix_of(p) not in val_src]
    tr_a = [p for p in alert if prefix_of(p) not in val_src]
    va_d = [p for p in drowsy if prefix_of(p) in val_src]
    va_a = [p for p in alert if prefix_of(p) in val_src]
    print(f'来源 {len(src_d)}+{len(src_a)}，留出 {sorted(val_src)}；训练 {len(tr_d)}+{len(tr_a)}，验证 {len(va_d)}+{len(va_a)}')

    tr_paths, tr_y = tr_d + tr_a, [1] * len(tr_d) + [0] * len(tr_a)
    va_paths, va_y = va_d + va_a, [1] * len(va_d) + [0] * len(va_a)
    tr_ds = DddDataset(tr_paths, tr_y, TRAIN_TF)
    va_ds = DddDataset(va_paths, va_y, VAL_TF)

    # 类别均衡采样：每个 epoch 各类等量出现
    n1, n0 = len(tr_d), len(tr_a)
    weights = [1.0 / n1 if y else 1.0 / n0 for y in tr_y]
    sampler = WeightedRandomSampler(weights, num_samples=2 * min(n0, n1), replacement=True)
    tr_dl = DataLoader(tr_ds, batch_size=BS, sampler=sampler, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BS, num_workers=0)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    model.to(DEVICE)
    # 解冻后 2 个 stage（features[11:]）；前面保持冻结
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.features[11:].parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW([
        {'params': [p for p in model.features[11:].parameters()], 'lr': 3e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best = (0.0, None)
    for epoch in range(EPOCHS):
        model.train()
        run_loss, n = 0.0, 0
        for xb, yb in tr_dl:
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = lossf(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(xb)
            n += len(xb)
        acc, cf, per_src = evaluate(model, va_dl, va_paths, val_src)
        print(f'epoch {epoch + 1}/{EPOCHS}  loss={run_loss / n:.4f}  val_acc={acc:.4f}  '
              f'recall_drowsy={cf[3] / (cf[2] + cf[3]):.4f}  per_src={per_src}')
        if acc > best[0]:
            best = (acc, {'epoch': epoch + 1, 'state': {k: v.cpu() for k, v in model.state_dict().items()}})
    acc, cf, per_src = evaluate(model, va_dl, va_paths, val_src)
    torch.save({'model': best[1]['state'], 'backbone': 'mobilenet_v2_ft',
                'dataset': 'DDD', 'val_sources': sorted(val_src)},
               os.path.join(OUT, 'fatigue_head.pt'))
    report = {'dataset': 'DDD', 'method': 'fine-tune+aug+balanced',
              'val_acc': round(acc, 4), 'best_epoch': best[1]['epoch'],
              'n_sources': len(src_d) + len(src_a), 'val_sources': sorted(val_src),
              'confusion': {'tn': cf[0], 'fp': cf[1], 'fn': cf[2], 'tp': cf[3]},
              'recall_drowsy': round(cf[3] / (cf[2] + cf[3]), 4),
              'precision_drowsy': round(cf[3] / (cf[1] + cf[3]), 4),
              'per_source_acc': per_src}
    json.dump(report, open(os.path.join(OUT, 'fatigue_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('报告:', json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
