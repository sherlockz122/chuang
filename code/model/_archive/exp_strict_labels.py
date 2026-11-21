# 诊断实验: 250m 全量 vs 严格 150m 子集（同用户划分、同训练流程）→ 量化标签噪声的影响
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn
from train_mlp import MLP, N_CLASS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'model')
torch.manual_seed(42)
np.random.seed(42)

d = np.load(os.path.join(OUT, 'features.npz'))
X, y = d['X'], d['y']
meta = [json.loads(l) for l in open(os.path.join(OUT, 'meta.jsonl'), encoding='utf-8')]

# 回连停留对的 poi_dist_m
dist = {}
for l in open(os.path.join(DATA, 'aligned', 'stay_node_pairs.jsonl'), encoding='utf-8'):
    p = json.loads(l)
    if p.get('poi_node'):
        dist[(p['taxi_id'], p['t_start'])] = p['poi_dist_m']
md = [dist.get((m['taxi_id'], m['t']), None) for m in meta]
md = np.array(md)
print(f'全量样本 {len(meta)}；严格≤150m: {(md <= 150).sum()}；150~250m: {((md > 150) & (md <= 250)).sum()}')


def run(mask, tag):
    Xs, ys = X[mask], y[mask]
    users = np.array([meta[i]['taxi_id'] for i in range(len(meta))])[mask]
    uniq = np.unique(users)
    np.random.shuffle(uniq)
    val_users = set(uniq[: int(len(uniq) * 0.3)])
    vm = np.array([u in val_users for u in users])
    tm = ~vm
    Xtr, ytr = torch.tensor(Xs[tm]), torch.tensor(ys[tm])
    Xv, yv = torch.tensor(Xs[vm]), torch.tensor(ys[vm])
    model = MLP(Xs.shape[1], N_CLASS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best3 = 0.0
    for epoch in range(120):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lv = model(Xv)
            top3 = sum(1 for i in range(len(yv)) if yv[i].item() in lv[i].topk(3).indices.tolist()) / len(yv)
            if top3 > best3:
                best3 = top3
                top1 = (lv.argmax(1) == yv).float().mean().item()
    print(f'{tag}: 样本 {len(users)}（val {vm.sum()}），top1 {top1:.3f}，top3 {best3:.3f}')


run(np.ones(len(meta), bool), '250m 全量')
run(md <= 150, '严格≤150m')
