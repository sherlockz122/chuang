# 指标三/四 正式测评: 决策合规率 + 各阶段端到端耗时（CPU，不依赖疲劳模型）
# 电池口径: 300 用户 × 每用户 3 个真实停留场景(时间=到达时刻, 位置=停留重心)
#           × 6 种环境组合(安全等级0/1/2 × 道路secondary/motorway) → 约 5400 次约束判定
# 输出:  ../data/model/eval_compliance.json
#   约束触发率(有候选被拦的场景占比) / 分规则触发率 / 通过后重验合规=100% / 耗时中位数
import sys, os, json, time, random
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
OUT = os.path.join(DATA, 'model')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import CATS, user_history_features, region_adcode
from constraint_engine import check as constraint_check
import llm_channel as lc
from push_decide import load_mlp, user_history_before, PAIRS, GROUP

random.seed(42)
np.random.seed(42)


def main():
    t0 = time.time()
    mlp = load_mlp()
    users = sorted(os.path.splitext(f)[0] for f in os.listdir(
        os.path.join(DATA, 'aligned', 'user_profiles')) if f.endswith('.json'))
    print(f'用户 {len(users)}，模型加载 {time.time() - t0:.1f}s')

    lat_mlp, lat_cand, lat_con = [], [], []
    n_scen = n_trig = 0
    rule_cnt = Counter()
    compliance_ok, compliance_total = 0, 0
    blocked_total = Counter()

    for uid in users:
        prof = json.load(open(os.path.join(DATA, 'aligned', 'user_profiles', f'{uid}.json'),
                              encoding='utf-8'))
        seq = [s for s in prof['seq'] if s['node_type'] == 'poi']
        if len(seq) < 2:
            continue
        for s in random.sample(seq, min(3, len(seq))):
            t_str = s['arrive']
            info = lc.node_info(s['node'])
            if not info:
                continue
            hist_c, hist_d, hist_days = user_history_before(prof, t_str)
            if not hist_c:
                continue
            # ---- MLP 通道（一次，环境无关） ----
            t_a = time.time()
            feats = user_history_features(hist_c, hist_d, hist_days, t_str,
                                          region_adcode(info['lng'], info['lat']))
            with torch.no_grad():
                probs = torch.softmax(mlp(torch.tensor(feats).unsqueeze(0)), -1)[0].numpy()
            top3 = [(CATS[i], float(probs[i])) for i in probs.argsort()[::-1][:3]]
            lat_mlp.append((time.time() - t_a) * 1000)

            # ---- 候选（一次，环境无关） ----
            t_b = time.time()
            cands = lc.find_candidates(info['lng'], info['lat'])
            top3_cats = {c for c, _ in top3}
            eligible = [c for c in cands
                        if c['category'] in top3_cats
                        or any((c['category'], tc) in PAIRS for tc in top3_cats)]
            for c in eligible:
                c['group'] = GROUP.get(c['category'], 'destination')
            lat_cand.append((time.time() - t_b) * 1000)

            # ---- 6 种环境组合的约束判定 ----
            for safety in (0, 1, 2):
                for highway in ('secondary', 'motorway'):
                    n_scen += 1
                    ctx = {'safety_level': safety, 'highway': highway,
                           'hour': int(t_str[11:13]),
                           'last_push_min': 5.0 if safety == 2 else None}  # 高疲劳时刚推过→触发冷却
                    t_c = time.time()
                    passed, blocked = constraint_check(eligible, ctx)
                    lat_con.append((time.time() - t_c) * 1000)
                    if blocked:
                        n_trig += 1
                        for b in blocked:
                            r = b['rule']
                            key = ('安全等级' if r.startswith('安全等级')
                                   else '深夜关闭' if r.startswith('深夜')
                                   else '高速禁推' if r.startswith('高速')
                                   else '推送冷却')
                            rule_cnt[key] += 1
                            blocked_total[b['poi']['category']] += 1
                    # 合规性重验: 通过集必须逐条通过（把通过集再喂一次，不应再有任何拦截）
                    passed2, blocked2 = constraint_check(passed, ctx)
                    compliance_total += len(passed)
                    compliance_ok += len(passed2)

    trig_rate = n_trig / n_scen * 100 if n_scen else 0
    report = {
        'n_scenarios': n_scen,
        'constraint_trigger_rate_pct': round(trig_rate, 2),
        'rule_trigger_counts': dict(rule_cnt.most_common()),
        'blocked_category_counts': dict(blocked_total.most_common()),
        'compliance_verified': {'passed_rechecked': compliance_ok, 'total_passed': compliance_total,
                                'recheck_pass_rate_pct':
                                    round(compliance_ok / compliance_total * 100, 2)
                                    if compliance_total else None},
        'latency_ms_median': {
            'mlp_feature+infer': round(float(np.median(lat_mlp)), 2),
            'candidate_search': round(float(np.median(lat_cand)), 2),
            'constraint_check': round(float(np.median(lat_con)), 3),
        },
        'latency_ms_p95': {
            'mlp_feature+infer': round(float(np.percentile(lat_mlp, 95)), 2),
            'candidate_search': round(float(np.percentile(lat_cand, 95)), 2),
            'constraint_check': round(float(np.percentile(lat_con, 95)), 3),
        },
    }
    json.dump(report, open(os.path.join(OUT, 'eval_compliance.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('报告:', json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
