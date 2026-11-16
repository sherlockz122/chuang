# 环节五/03g: RGB 眼态学生模型 v2——EAR 回归蒸馏（软目标=连续EAR值）
# 与 v1 硬二分类的区别: 学生直接学眼睁开度 EAR 的连续值（BCEWithLogits 软目标），
# 决策面与教师对齐，眯眼/小眼落在中间值而不是被硬归到闭眼类
# 验证: 90/10 随机（仅早停）；真正的泛化测试是 ext_test_ddd.py 的留来源视频
# 输出:  data/model/eye_rgb.pt + eye_rgb_report.json
import sys, os, json
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
LABELS = os.path.join(DATA, 'fatigue', 'eye_rgb_labels.jsonl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
EPOCHS = 20
BS = 128


class EyeNet(nn.Module):
    """与 train_eye.py 同骨干，末层改 1 输出（回归 EAR 软目标，~9.4万参数）"""

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


class EyeDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(Image.open(r['path']).convert('RGB')), float(r['ear'])


TF_TRAIN = transforms.Compose([
    transforms.Resize((72, 72)), transforms.RandomCrop(64), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
TF_VAL = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main():
    rows = [json.loads(l) for l in open(LABELS, encoding='utf-8')]
    # 留来源: 剔除 DDD 留出的来源视频（验证集身份，见 ddd_split.py）
    from ddd_split import heldout_sources
    held = heldout_sources()
    rows = [r for r in rows if not (r['src'].startswith('ddd:')
                                    and r['src'][4:] in held)]
    np.random.seed(42)
    idx = np.random.permutation(len(rows))
    n_val = int(len(rows) * 0.1)
    tr_rows = [rows[i] for i in idx[n_val:]]
    va_rows = [rows[i] for i in idx[:n_val]]
    print(f'留出DDD来源 {sorted(held)}；训练 {len(tr_rows)}，验证 {len(va_rows)}')

    tr_dl = DataLoader(EyeDataset(tr_rows, TF_TRAIN), batch_size=BS, shuffle=True, num_workers=0)
    va_dl = DataLoader(EyeDataset(va_rows, TF_VAL), batch_size=BS, num_workers=0)

    model = EyeNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数: {n_params / 1000:.0f}K，设备 {DEVICE}')
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.BCEWithLogitsLoss()  # 软目标: EAR∈[0,1] 连续值
    best = 1e9
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            lossf(model(xb.to(DEVICE)).squeeze(1), yb.to(DEVICE)).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for xb, yb in va_dl:
                preds.append(torch.sigmoid(model(xb.to(DEVICE)).squeeze(1)).cpu())
                ys.append(yb)
            pred, y = torch.cat(preds), torch.cat(ys)
            mae = (pred - y).abs().mean().item()
            # 闭眼口径一致率(≤0.20)
            agr = ((pred <= 0.20) == (y <= 0.20)).float().mean().item()
        print(f'epoch {epoch + 1:2d}  val_MAE={mae:.4f}  闭眼口径一致率={agr:.4f}')
        if mae < best:
            best = mae
            torch.save({'model': model.state_dict(), 'arch': 'EyeNet64-EARreg'},
                       os.path.join(OUT, 'eye_rgb.pt'))
    report = {'val_mae': round(best, 4), 'n_train': len(tr_rows), 'n_val': len(va_rows),
              'n_params': n_params, 'teacher': 'MediaPipe-EAR(连续软目标)',
              'train_src': 'DDD(除留出来源)+NTHU-001/002/005'}
    json.dump(report, open(os.path.join(OUT, 'eye_rgb_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('报告:', json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
