# 环节四/02: 训练 MLP 编码通道（"说数字"：用户上下文 → 19 类 POI 偏好分布）
# 输入:  ../data/model/features.npz + meta.jsonl
# 输出:  ../data/model/mlp.pt + train_report.json
# 指标:  top-1/top-3 命中率（四项评估指标之"感知准确率"）
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(ROOT), 'data', 'model')
torch.manual_seed(42)
np.random.seed(42)

N_CLASS = 19  # 19 个 POI 类别（services 类五环内无监督样本，实际 18 类有样本）


class MLP(nn.Module):
    def __init__(self, dim_in, n_class):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_class),
        )

    def forward(self, x):
        return self.net(x)


def main():
    d = np.load(os.path.join(OUT, 'features.npz'))
    X, y = d['X'], d['y']
    meta = [json.loads(l) for l in open(os.path.join(OUT, 'meta.jsonl'), encoding='utf-8')]

    # 按用户划分 train/val（同一用户的停留不跨集，防泄漏）
    users = np.array([m['taxi_id'] for m in meta])
    uniq = np.unique(users)
    np.random.shuffle(uniq)
    val_users = set(uniq[: int(len(uniq) * 0.3)])
    val_mask = np.array([u in val_users for u in users])
    tr_mask = ~val_mask

    Xtr, ytr = torch.tensor(X[tr_mask]), torch.tensor(y[tr_mask])
    Xv, yv = torch.tensor(X[val_mask]), torch.tensor(y[val_mask])

    model = MLP(X.shape[1], N_CLASS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    best = {'acc3': 0.0}
    for epoch in range(120):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                lv = model(Xv)
                top1 = (lv.argmax(1) == yv).float().mean().item()
                top3 = sum(1 for i in range(len(yv))
                           if yv[i].item() in lv[i].topk(3).indices.tolist()) / len(yv)
                print(f'epoch {epoch + 1:3d}  loss {loss.item():.4f}  val top1 {top1:.3f}  top3 {top3:.3f}')
            if top3 > best['acc3']:
                best = {'acc3': top3, 'top1': top1, 'epoch': epoch + 1}
                torch.save(model.state_dict(), os.path.join(OUT, 'mlp.pt'))

    report = {'train_samples': int(tr_mask.sum()), 'val_samples': int(val_mask.sum()),
              'val_top1': round(best['top1'], 4), 'val_top3': round(best['acc3'], 4),
              'best_epoch': best['epoch'], 'n_class': N_CLASS, 'dim_in': X.shape[1]}
    json.dump(report, open(os.path.join(OUT, 'train_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('报告:', json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
