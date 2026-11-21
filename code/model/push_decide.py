# 环节五/02: 推送决策编排（双通道 + 约束 = 最终推送）——Vue demo 的后端入口
# 用法:  python push_decide.py --user U_10 --time "2008-02-03 14:00:00" --lng 116.383 --lat 39.907 \
#             [--safety 0] [--fatigue <帧图/帧目录>] [--highway secondary] [--last-push-min 0] [--no-llm]
# --fatigue: 真实疲劳信号源，调用 fatigue_infer 自动推断 safety_level（优先于 --safety）
# 输出:  最终决策 JSON（MLP top3 类别 → 附近候选 → 约束过滤 → LLM 定稿）
import sys, os, json, argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import yaml
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
CFG = os.path.join(ROOT, 'config')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import CATS, CAT2ID, user_history_features, region_adcode
from constraint_engine import check as constraint_check
import llm_channel as lc

cats = yaml.safe_load(open(os.path.join(CFG, 'categories.yaml'), encoding='utf-8'))
GROUP = {c['id']: g['id'] for g in cats['groups'] for c in g['categories']}
PAIRS = set()
for a, b in cats['complement_pairs']:
    PAIRS.add((a, b))
    PAIRS.add((b, a))


_MLP = None  # 进程级缓存：每次决策不再重读 mlp.pt


def load_mlp():
    global _MLP
    if _MLP is not None:
        return _MLP
    from train_mlp import MLP, N_CLASS
    m = MLP(64, N_CLASS)
    m.load_state_dict(torch.load(os.path.join(DATA, 'model', 'mlp.pt'), weights_only=True))
    m.eval()
    _MLP = m
    return m


def user_history_before(prof, t_str):
    """该时刻之前的历史（在线口径，与训练一致）"""
    cats_h, durs, days = Counter(), [], set()
    for s in prof['seq']:
        if s['arrive'] >= t_str:
            break
        if s['node_type'] != 'poi':
            continue
        cats_h[CAT2ID_CAT(s['node'])] += 1
        durs.append(s['dur_min'])
        days.add(s['arrive'][:10])
    return cats_h, durs, days


NODE_CAT = {}


def CAT2ID_CAT(node_id):
    if not NODE_CAT:
        nodes = json.load(open(os.path.join(DATA, 'graph_out', 'nodes.json'), encoding='utf-8'))
        NODE_CAT.update({n['node_id']: n.get('category', '') for n in nodes})
    return NODE_CAT.get(node_id, '')


def run_fatigue(fatigue_path):
    """真实疲劳信号源: 调用 mp-env 子进程的 fatigue_infer（感知子服务独立运行时）"""
    fat_py = r'D:\mp-env\Scripts\python.exe'
    fat_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fatigue_infer.py')
    if os.path.exists(fat_py):
        import subprocess
        out = subprocess.run([fat_py, fat_script, fatigue_path],
                             capture_output=True, text=True, encoding='utf-8').stdout.strip()
        try:
            return json.loads(out.splitlines()[-1])
        except Exception:
            return {'error': f'子进程输出解析失败: {out[:200]}'}
    import fatigue_infer as fi
    return fi.infer(fatigue_path)


def decide(user, time, lng, lat, safety=0, fatigue=None, highway='secondary',
           last_push_min=0.0, use_llm=True):
    """端到端推送决策（API/CLI 共用入口），返回完整决策 JSON"""
    fatigue_sig = None
    if fatigue:
        fatigue_sig = run_fatigue(fatigue)
        if 'safety_level' in fatigue_sig:
            safety = fatigue_sig['safety_level']

    prof = json.load(open(os.path.join(DATA, 'aligned', 'user_profiles',
                                       f'{user[2:]}.json'), encoding='utf-8'))
    hist_c, hist_d, hist_days = user_history_before(prof, time)

    # ---------- 通道1: MLP 说数字 ----------
    mlp = load_mlp()
    rid = region_adcode(lng, lat)  # 区域特征与训练口径一致（停留点所属行政区）
    feats = user_history_features(hist_c, hist_d, hist_days, time, rid)
    with torch.no_grad():
        logits = mlp(torch.tensor(feats).unsqueeze(0))[0]
    probs = torch.softmax(logits, dim=-1).numpy()
    top3_idx = probs.argsort()[::-1]
    top3 = [(CATS[i], round(float(probs[i]), 3)) for i in top3_idx if i < len(CATS)][:3]

    # ---------- 候选: 附近 POI ∩ (top3 类别 ∪ 功能互补) ----------
    cands = lc.find_candidates(lng, lat)
    top3_cats = {c for c, _ in top3}
    eligible = []
    for c in cands:
        cat = c['category']
        if cat in top3_cats:
            eligible.append(c)
        elif any((cat, tc) in PAIRS for tc in top3_cats):
            eligible.append(c)
    for c in eligible:
        c['group'] = GROUP.get(c['category'], 'destination')

    ctx = {'safety_level': safety, 'highway': highway,
           'hour': int(time[11:13]),
           'last_push_min': last_push_min if last_push_min > 0 else None}
    passed, blocked = constraint_check(eligible, ctx)

    decision = None
    if passed and use_llm:
        prompt = lc.build_prompt(user, time, lng, lat, passed, ctx, top3)
        tok, model, device = lc.load_model()
        msgs = [{'role': 'system', 'content': lc.SYSTEM},
                {'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        resp = tok.decode(gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        try:
            decision = json.loads(resp[resp.index('{'): resp.rindex('}') + 1])
        except Exception:
            decision = {'decision': 'parse_error', 'raw': resp}
        # 校验: LLM 只能从合规候选中选（0.5B 小模型偶发越界/幻觉），无效即回退 MLP 兜底
        if not isinstance(decision, dict) or decision.get('poi_id') not in {c['poi_id'] for c in passed}:
            decision = {'poi_id': passed[0]['poi_id'], 'reason': 'LLM输出无效, 回退MLP兜底: 合规候选取最近',
                        'confidence': round(float(top3[0][1]), 3)}
    elif passed:
        decision = {'poi_id': passed[0]['poi_id'], 'reason': 'MLP通道兜底: 合规候选取最近',
                    'confidence': round(float(top3[0][1]), 3)}
    # LLM 只回 poi_id+reason，名字/类别/距离由候选列表补全（前端展示用）
    if decision and isinstance(decision, dict) and decision.get('poi_id'):
        hit = next((c for c in eligible if c['poi_id'] == decision['poi_id']), None)
        if hit:
            decision['name'] = hit['name']
            decision['category'] = hit['category']
            decision['dist_m'] = hit['dist_m']
        # 定稿一致性: 有 MLP top3 类合规候选时，定稿必须落在 top3 类内
        # （说句子不能和说数字打架——LLM 偶发选互补类候选却口称另一类，如"加油站"实选交通枢纽）
        top3_cats = {c for c, _ in top3}
        if decision.get('category') not in top3_cats:
            fb = next((c for c in passed if c['category'] in top3_cats), None)
            if fb:
                decision = {'poi_id': fb['poi_id'],
                            'reason': f'定稿与MLP预测不一致, 回退MLP优先候选: {fb["name"]}',
                            'confidence': round(float(top3[0][1]), 3),
                            'name': fb['name'], 'category': fb['category'], 'dist_m': fb['dist_m']}

    return {'user': user, 'time': time, 'position': [lng, lat],
            'safety_level': safety, 'fatigue_signal': fatigue_sig, 'highway': highway,
            'mlp_top3_cats': top3,
            'candidates': [{'poi_id': c['poi_id'], 'name': c['name'],
                            'category': c['category'], 'dist_m': c['dist_m']} for c in eligible],
            'constraint_blocked': [{'poi': b['poi']['poi_id'], 'rule': b['rule']} for b in blocked],
            'decision': decision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True)
    ap.add_argument('--time', required=True)
    ap.add_argument('--lng', type=float, required=True)
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--safety', type=int, default=0)
    ap.add_argument('--fatigue', help='疲劳信号源: 摄像头帧图/帧目录，自动推断 safety_level')
    ap.add_argument('--highway', default='secondary')
    ap.add_argument('--last-push-min', type=float, default=0)
    ap.add_argument('--no-llm', action='store_true', help='跳过 LLM 通道（快速调试）')
    args = ap.parse_args()
    out = decide(args.user, args.time, args.lng, args.lat, safety=args.safety,
                 fatigue=args.fatigue, highway=args.highway,
                 last_push_min=args.last_push_min, use_llm=not args.no_llm)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
