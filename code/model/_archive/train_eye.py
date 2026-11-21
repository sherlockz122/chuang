# 环节五/03e: 眼睛状态分类器（疲劳模块 v4 核心）——MRL Eye Dataset
# 设计: 从零训练微型 CNN(~15万参数)，输入 64×64 眼部裁剪，二分类开/闭
#       端侧小模型叙事: 眼睛状态是疲劳的最小语义单元，模型越小越适配车机
# 划分: 按受试者隔离（s0001~s0033 训练 / s0034~s0037 验证，文件名第1字段=受试者）
# 标签: 文件名第3字段 0=闭眼 1=睁眼
# 输出:  ../data/model/eye_state.pt + eye_report.json
import sys, os, json, glob, re, random
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'model')
MRL = os.path.join(DATA, 'fatigue', 'mrl', 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)
torch.manual_seed(42)
EPOCHS = 20
BS = 128
N_TRAIN_PER_CLASS = 20000  # 每类抽样上限（84k 全量太多，2万/类足够）


class EyeNet(nn.Module):
    """微型 CNN: 3 卷积块 + GAP，~15万参数"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


class EyeDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB')), self.labels[i]


TF_TRAIN = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
TF_VAL = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def parse(p):
    """返回 (subject_id, label)。权威标签=文件夹名(awake=1/sleepy=0)；
    此 fork 文件名第3字段与文件夹标签不一致，不可用作眼态"""
    base = os.path.basename(p)
    m = re.match(r'(s\d{4})_', base)
    return m.group(1), 1 if os.path.basename(os.path.dirname(p)) == 'awake' else 0


def main():
    paths = sorted(glob.glob(os.path.join(MRL, 'train', '*', '*.png')) +
                   glob.glob(os.path.join(MRL, 'val', '*', '*.png')) +
                   glob.glob(os.path.join(MRL, 'test', '*', '*.png')))
    subj, lab = zip(*[parse(p) for p in paths])
    subj, lab = np.array(subj), np.array(lab)
    uniq = sorted(set(subj))
    val_s = set(u for u in uniq if int(u[1:]) >= 34)  # MRL 官方口径: 后4名受试者验证
    vm = np.array([s in val_s for s in subj])
    print(f'共 {len(paths)} 张，受试者 {len(uniq)}，验证受试者 {sorted(val_s)}；'
          f'训练 {np.sum(~vm)}，验证 {np.sum(vm)}')
    print(f'标签分布: 睁眼 {np.sum(lab == 1)} / 闭眼 {np.sum(lab == 0)}')

    tr_p = [p for p, v in zip(paths, vm) if not v]
    va_p = [p for p, v in zip(paths, vm) if v]
    tr_l = [int(l) for l, v in zip(lab, vm) if not v]
    va_l = [int(l) for l, v in zip(lab, vm) if v]
    # 每类抽样上限
    idx_tr = [i for i, l in enumerate(tr_l) if l == 1][:N_TRAIN_PER_CLASS] + \
             [i for i, l in enumerate(tr_l) if l == 0][:N_TRAIN_PER_CLASS]
    tr_p = [tr_p[i] for i in idx_tr]
    tr_l = [tr_l[i] for i in idx_tr]

    tr_dl = DataLoader(EyeDataset(tr_p, tr_l, TF_TRAIN), batch_size=BS, shuffle=True, num_workers=0)
    va_dl = DataLoader(EyeDataset(va_p, va_l, TF_VAL), batch_size=BS, num_workers=0)

    model = EyeNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数: {n_params / 1000:.0f}K，设备 {DEVICE}')
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)  # 该fork标签带噪声，平滑防过信
    best = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            lossf(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for xb, yb in va_dl:
                preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
                ys.append(yb)
            pred, y = torch.cat(preds), torch.cat(ys)
            acc = (pred == y).float().mean().item()
            recall_open = ((pred == 1) & (y == 1)).sum().item() / max(1, (y == 1).sum().item())
            recall_closed = ((pred == 0) & (y == 0)).sum().item() / max(1, (y == 0).sum().item())
        print(f'epoch {epoch + 1:2d}  val_acc={acc:.4f}  睁眼召回={recall_open:.4f}  闭眼召回={recall_closed:.4f}')
        if acc > best:
            best = acc
            torch.save({'model': model.state_dict(), 'arch': 'EyeNet64'},
                       os.path.join(OUT, 'eye_state.pt'))
    report = {'val_acc': round(best, 4), 'n_train': len(tr_p), 'n_val': len(va_p),
              'n_params': n_params, 'n_subjects': len(uniq), 'val_subjects': sorted(val_s)}
    json.dump(report, open(os.path.join(OUT, 'eye_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('报告:', json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
