# 环节五/07: 金标准情景题库——非回放验证体系核心（决策级验证）
# 口径: 把"何时该推什么/该拦什么"编成有标准答案的考题，端到端考决策链
#       (MLP→候选召回→约束引擎→LLM定稿)，逐题对金标准判卷，确定性可复现
# 用法: python scenario_bank.py            # 全链路(含 LLM 定稿)
#       python scenario_bank.py --fast     # 跳过 LLM(纯规则题快跑)
# 输出: data/model/scenario_eval.json + scenario_report.md + 控制台判卷表
import sys, os, json, time, argparse
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
sys.path.insert(0, os.path.join(ROOT, 'model'))
from push_decide import decide, load_mlp, user_history_before, CAT2ID_CAT  # noqa: E402
from features import CATS  # noqa: E402

# ---------------- 情景题库（前置条件由 probe_scenarios.py 从 300 用户真实画像筛出） ----------------
# 位置取真实 POI 坐标（保证目标类候选一定在场），时刻/安全/道路/冷却为构造输入
# expect 字段: block_cats=必须被拦的类别; block_rule=拦截原因必含子串; decision_null=决策必须为空;
#              push_cat=必须推该类; push_in_top3=必须推 MLP top3 之一
SCENARIOS = [
    {'id': 'S01', 'rule': 'safety', 'name': 'L1低警惕拦景区',
     'user': 'U_15', 'time': '2008-02-08 12:00:00', 'pos_cat': 'scenic',
     'safety': 1, 'highway': 'secondary', 'last_push_min': 0.0,
     'expect': {'block_cats': ['scenic'], 'block_rule': '不允许'}},
    {'id': 'S02', 'rule': 'safety', 'name': 'L2疲劳拦餐饮',
     'user': 'U_3', 'time': '2008-02-08 12:00:00', 'pos_cat': 'restaurant',
     'safety': 2, 'highway': 'secondary', 'last_push_min': 0.0,
     'expect': {'block_cats': ['restaurant', 'hotel'], 'block_rule': '不允许'}},
    {'id': 'S03', 'rule': 'safety', 'name': 'L2疲劳只放停车类',
     'user': 'U_3', 'time': '2008-02-08 12:00:00', 'pos_cat': 'parking',
     'safety': 2, 'highway': 'secondary', 'last_push_min': 0.0,
     'expect': {'push_cat': 'parking'}},
    {'id': 'S04', 'rule': 'safety', 'name': 'L0清醒白天放景区',
     'user': 'U_15', 'time': '2008-02-08 12:00:00', 'pos_cat': 'scenic',
     'safety': 0, 'highway': 'secondary', 'last_push_min': 0.0,
     'expect': {'not_blocked_cats': ['scenic']}},
    {'id': 'S05', 'rule': 'night', 'name': '深夜23点拦景区',
     'user': 'U_24', 'time': '2008-02-08 23:00:00', 'pos_cat': 'scenic',
     'safety': 0, 'highway': 'secondary', 'last_push_min': 0.0,
     'expect': {'block_cats': ['scenic'], 'block_rule': '深夜时段'}},
    {'id': 'S06', 'rule': 'night', 'name': '深夜放行餐饮(对照组)',
     'user': 'U_24', 'time': '2008-02-08 23:00:00', 'pos_cat': 'restaurant',
     'safety': 0, 'highway': 'secondary', 'last_push_min': 0.0,
     'expect': {'push_in_top3': True}},
    {'id': 'S07', 'rule': 'highway', 'name': '高速拦目的地型',
     'user': 'U_3', 'time': '2008-02-08 12:00:00', 'pos_cat': 'restaurant',
     'safety': 0, 'highway': 'motorway', 'last_push_min': 0.0,
     'expect': {'block_cats': ['restaurant', 'hotel'], 'block_rule': '高速路段'}},
    {'id': 'S08', 'rule': 'highway', 'name': '高速放行加油站(对照组)',
     'user': 'U_318', 'time': '2008-02-08 12:00:00', 'pos_cat': 'fuel',
     'safety': 0, 'highway': 'motorway', 'last_push_min': 0.0,
     'expect': {'push_cat': 'fuel'}},
    {'id': 'S09', 'rule': 'cooldown', 'name': '冷却期内全拦',
     'user': 'U_3', 'time': '2008-02-08 12:00:00', 'pos_cat': 'restaurant',
     'safety': 0, 'highway': 'secondary', 'last_push_min': 7.0,
     'expect': {'decision_null': True, 'block_rule': '距上次推送'}},
    {'id': 'S10', 'rule': 'combo', 'name': '疲劳+高速+冷却三重叠加',
     'user': 'U_3', 'time': '2008-02-08 12:00:00', 'pos_cat': 'restaurant',
     'safety': 2, 'highway': 'motorway', 'last_push_min': 7.0,
     'expect': {'decision_null': True, 'block_rule': '距上次推送'}},
    {'id': 'S11', 'rule': 'push', 'name': '午餐商圈推餐饮',
     'user': 'U_3', 'time': '2008-02-08 12:00:00', 'pos_cat': 'restaurant',
     'safety': 0, 'highway': 'secondary', 'last_push_min': 0.0,
     'expect': {'push_in_top3': True}},
    {'id': 'S12', 'rule': 'push', 'name': '途中推加油站',
     'user': 'U_318', 'time': '2008-02-08 12:00:00', 'pos_cat': 'fuel',
     'safety': 0, 'highway': 'secondary', 'last_push_min': 0.0,
     'expect': {'push_cat': 'fuel'}},
    {'id': 'S13', 'rule': 'cold', 'name': '冷启动用户稳健决策',
     'user': 'U_191', 'time': '2008-02-06 12:00:00', 'pos_cat': 'restaurant',
     'safety': 0, 'highway': 'secondary', 'last_push_min': 0.0,
     'expect': {'push_in_top3': True, 'max_p_max': 0.35}},
]

# ---------------- 位置与画像 ----------------
_NODES = None


def poi_pos(cat):
    """取一个带名字的真实 POI 坐标（保证目标类候选在场）"""
    global _NODES
    if _NODES is None:
        _NODES = json.load(open(os.path.join(DATA, 'graph_out', 'nodes.json'), encoding='utf-8'))
    for n in _NODES:
        if n.get('type') == 'poi' and n.get('category') == cat and n.get('name'):
            return n['lng'], n['lat']
    return None


def prof_of(uid):
    return json.load(open(os.path.join(DATA, 'aligned', 'user_profiles', f'{uid[2:]}.json'),
                          encoding='utf-8'))


def freq_top3(uid, t_str):
    """频次基线: 决策时刻之前历史最常去的 3 个类别（无任何模型/约束）"""
    prof = prof_of(uid)
    hist_c = Counter()
    for s in prof['seq']:
        if s['arrive'] >= t_str:
            break
        if s['node_type'] == 'poi':
            hist_c[CAT2ID_CAT(s['node'])] += 1
    return [c for c, _ in hist_c.most_common(3)]


# ---------------- 判卷 ----------------
def judge(sc, out):
    """对金标准判卷，返回 (pass, 明细)"""
    exp = sc['expect']
    cands = out['candidates']
    blocked = {b['poi']: b['rule'] for b in out.get('constraint_blocked', [])}
    dec = out.get('decision')
    checks, ok = [], True

    # 1) 必须拦截的类别: 该类别每个候选都在拦截名单且原因含规则子串
    for cat in exp.get('block_cats', []):
        for c in cands:
            if c['category'] != cat:
                continue
            rule = blocked.get(c['poi_id'], '')
            hit = bool(rule) and exp.get('block_rule', '') in rule
            checks.append(f'{"✓" if hit else "✗"} 拦截 {c["name"]}({cat}): {rule or "未拦截"}')
            if not hit:
                ok = False
    # 1b) 必须不被拦截的类别（对照组: 条件不满足时规则不得误伤）
    for cat in exp.get('not_blocked_cats', []):
        for c in cands:
            if c['category'] != cat:
                continue
            rule = blocked.get(c['poi_id'], '')
            hit = not rule
            checks.append(f'{"✓" if hit else "✗"} 不拦 {c["name"]}({cat}): {rule or "未拦截"}')
            if not hit:
                ok = False
    # 2) 决策不得指向被拦类别
    if dec and dec.get('category') in exp.get('block_cats', []):
        checks.append(f'✗ 决策指向被拦类别 {dec["category"]}')
        ok = False
    # 3) 必须为空决策
    if exp.get('decision_null'):
        hit = dec is None
        checks.append(f'{"✓" if hit else "✗"} 决策为空(全拦): 实际 {"有推送" if dec else "空"}')
        if not hit:
            ok = False
    # 4) 必须推送指定类别 / top3 之一
    top3 = [c for c, _ in out.get('mlp_top3_cats', [])]
    if exp.get('push_cat'):
        hit = bool(dec) and dec.get('category') == exp['push_cat']
        checks.append(f'{"✓" if hit else "✗"} 推送 {exp["push_cat"]}: 实际 {dec.get("name") if dec else "无"}')
        if not hit:
            ok = False
    elif exp.get('push_in_top3'):
        hit = bool(dec) and dec.get('category') in top3
        checks.append(f'{"✓" if hit else "✗"} 推送 top3 之一({"/".join(top3)}): 实际 {dec.get("category") if dec else "无"}')
        if not hit:
            ok = False
    # 5) 结构完整性: 推送的 poi_id 必须在候选中, 名字/类别/距离已补全, 理由非空
    if dec and dec.get('poi_id'):
        in_cands = dec['poi_id'] in {c['poi_id'] for c in cands}
        struct = in_cands and bool(dec.get('name')) and bool(dec.get('reason'))
        checks.append(f'{"✓" if struct else "✗"} 结构完整(候选内/有名字/有理由)')
        if not struct:
            ok = False
    # 6) 冷启动: MLP 概率不塌缩
    if 'max_p_max' in exp:
        maxp = max([p for _, p in out.get('mlp_top3_cats', [])] or [1.0])
        spread = maxp <= exp['max_p_max']
        checks.append(f'{"✓" if spread else "✗"} 概率不塌缩(max={maxp:.3f}≤{exp["max_p_max"]})')
        if not spread:
            ok = False
    return ok, checks


def freq_verdict(sc, out):
    """频次基线同卷作答: 无视约束推历史最常去类, 看是否违反金标准"""
    exp = sc['expect']
    f3 = freq_top3(sc['user'], sc['time'])
    cands = out['candidates']
    picked = None
    for c in sorted(f3, key=lambda x: f3.index(x)):
        hit = next((x for x in cands if x['category'] == c), None)
        if hit:
            picked = hit
            break
    # 频次基线的推送违反金标准?
    if exp.get('decision_null'):
        return not picked, f'频次基线{"无候选可推(恰好通过)" if not picked else f"会推 {picked["name"]}(无视冷却/安全) ✗"}'
    if exp.get('block_cats'):
        if picked and picked['category'] in exp['block_cats']:
            return False, f'频次基线会推被拦类别 {picked["name"]}({picked["category"]}) ✗'
    if exp.get('push_cat'):
        return bool(picked) and picked['category'] == exp['push_cat'], \
            f'频次基线推 {picked["name"] if picked else "无"}'
    return True, f'频次基线推 {picked["name"] if picked else "无"}(该类无约束冲突)'


# ---------------- 模块级验证矩阵 ----------------
def module_matrix():
    rd = lambda p: json.load(open(os.path.join(DATA, p), encoding='utf-8'))  # noqa: E731
    m = {}
    tr = rd('model/train_report.json')
    m['MLP预测'] = {'val_top1': tr['val_top1'], 'val_top3': tr['val_top3'],
                    'train_samples': tr['train_samples'], 'val_samples': tr['val_samples']}
    re = rd('model/replay_eval.json')
    m['回放评测'] = {'all_rate': re['all']['rate'], 'val_rate': re['val']['rate'],
                     'val_graded': re['val']['graded']}
    be = rd('model/baseline_eval.json')
    m['对照基线'] = {'随机_val': be['val']['rand']['rate'],
                     '频次_val': be['val']['freq']['rate']}
    ce = rd('model/compare_eval.json')
    m['分片对比'] = {'冷启动_diff': ce['by_tier']['冷启动(停留<8)']['diff'],
                     '最大时段_diff': max(v['diff'] for v in ce['by_hour'].values())}
    eye = rd('model/eye_rgb_report.json')
    ext = rd('model/ext_test_ddd.json')
    m['疲劳识别'] = {'val_mae': eye['val_mae'], 'params': eye['n_params'],
                     'DDD留来源准确率': ext['student']['heldout_video_acc'],
                     '留来源视频数': ext['student']['heldout_n']}
    m['数据对齐'] = {'对齐覆盖率': rd('aligned/align_stats.json')['align_coverage_pct'],
                     '高德属性对齐': rd('aligned/merge_stats.json')['matched_total']}
    return m


# ---------------- 主流程 ----------------
def run(use_llm=True):
    mlp = load_mlp()  # 预热(进程内缓存)
    rows = []
    t_all = []
    for sc in SCENARIOS:
        lng, lat = poi_pos(sc['pos_cat'])
        t0 = time.perf_counter()
        out = decide(sc['user'], sc['time'], lng, lat, safety=sc['safety'],
                     highway=sc['highway'], last_push_min=sc['last_push_min'],
                     use_llm=use_llm)
        el = round((time.perf_counter() - t0) * 1000, 1)
        t_all.append(el)
        ok, checks = judge(sc, out)
        fok, fmsg = freq_verdict(sc, out)
        rows.append({'id': sc['id'], 'name': sc['name'], 'rule': sc['rule'],
                     'user': sc['user'], 'time': sc['time'],
                     'safety': sc['safety'], 'highway': sc['highway'],
                     'last_push_min': sc['last_push_min'],
                     'mlp_top3': [c for c, _ in out['mlp_top3_cats']],
                     'candidates': [(c['name'], c['category']) for c in out['candidates']],
                     'blocked': out['constraint_blocked'],
                     'decision': out['decision'],
                     'pass': ok, 'checks': checks,
                     'freq_top3': freq_top3(sc['user'], sc['time']),
                     'freq_pass': fok, 'freq_msg': fmsg,
                     'elapsed_ms': el})
        print(f"\n=== {sc['id']} {sc['name']} [{'✓通过' if ok else '✗未通过'}] {el}ms ===")
        for c in checks:
            print('   ', c)
        print('    MLP top3:', '/'.join(c for c, _ in out['mlp_top3_cats']))
        print('    频次基线:', fmsg)

    n = len(rows)
    p = sum(r['pass'] for r in rows)
    fp = sum(r['freq_pass'] for r in rows)
    by_rule = {}
    for r in rows:
        by_rule.setdefault(r['rule'], [0, 0])
        by_rule[r['rule']][0] += 1
        by_rule[r['rule']][1] += 1 if r['pass'] else 0
    out = {
        '口径': '情景题库: 端到端决策对金标准判卷(约束四规则边界+组合+推送合理性+冷启动), 非回放; '
                '频次基线=同卷对照(历史最常去top3, 无约束)',
        'use_llm': use_llm,
        'summary': {'scenarios': n, 'pass': p, 'rate': round(p / n, 4),
                    'by_rule': {k: {'pass': v[1], 'total': v[0]} for k, v in by_rule.items()},
                    'avg_elapsed_ms': round(sum(t_all) / len(t_all), 1),
                    'max_elapsed_ms': max(t_all)},
        'freq_baseline': {'pass': fp, 'total': n, 'rate': round(fp / n, 4)},
        'matrix': module_matrix(),
        'scenarios': rows,
    }
    json.dump(out, open(os.path.join(DATA, 'model', 'scenario_eval.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f"\n===== 判卷汇总: {p}/{n} 通过 ({(p/n*100):.1f}%) | 频次基线对照: {fp}/{n} ({(fp/n*100):.1f}%) =====")
    print('分规则:', {k: f'{v["pass"]}/{v["total"]}' for k, v in out['summary']['by_rule'].items()})
    print(f'决策耗时: 平均 {out["summary"]["avg_elapsed_ms"]}ms / 最大 {out["summary"]["max_elapsed_ms"]}ms')
    write_report(out)
    return out


def write_report(out):
    """生成可读报告 data/model/scenario_report.md（实测/归档用）"""
    L = []
    L.append('# 金标准情景题库判卷报告\n')
    L.append(f"- 口径: {out['口径']}\n")
    L.append(f"- 判卷: **{out['summary']['pass']}/{out['summary']['scenarios']} 通过"
             f"（{out['summary']['rate']*100:.1f}%）** | 频次基线同卷: "
             f"{out['freq_baseline']['pass']}/{out['freq_baseline']['total']}"
             f"（{out['freq_baseline']['rate']*100:.1f}%）\n")
    L.append(f"- 决策耗时: 平均 {out['summary']['avg_elapsed_ms']}ms / 最大 {out['summary']['max_elapsed_ms']}ms\n\n")
    L.append('## 分规则通过率\n\n')
    L.append('| 规则 | 通过 |\n|---|---|\n')
    for k, v in out['summary']['by_rule'].items():
        L.append(f'| {k} | {v["pass"]}/{v["total"]} |\n')
    L.append('\n## 逐题判卷\n\n')
    for r in out['scenarios']:
        d = r['decision']
        dec_txt = (f'{d.get("name")}({d.get("category")})' if d else '空(全拦)')
        L.append(f"### {r['id']} {r['name']} {'✅' if r['pass'] else '❌'}\n\n")
        L.append(f"- 输入: {r['user']} @ {r['time']} · safety={r['safety']} · {r['highway']} · 距上次推送{r['last_push_min']}min\n")
        L.append(f"- MLP top3: {'/'.join(r['mlp_top3'])} · 决策: {dec_txt} · {r['elapsed_ms']}ms\n")
        for c in r['checks']:
            L.append(f"  - {c}\n")
        L.append(f"- 频次基线对照: {r['freq_msg']}\n\n")
    L.append('## 模块级验证矩阵\n\n')
    m = out['matrix']
    L.append('| 模块 | 指标 | 数字 |\n|---|---|---|\n')
    L.append(f"| 兴趣建模 MLP | val top1 / top3 | {m['MLP预测']['val_top1']} / {m['MLP预测']['val_top3']} |\n")
    L.append(f"| 回放评测 | 全体 / 验证集 top3 命中 | {m['回放评测']['all_rate']} / {m['回放评测']['val_rate']} |\n")
    L.append(f"| 对照基线 | 随机 / 频次(验证集) | {m['对照基线']['随机_val']} / {m['对照基线']['频次_val']} |\n")
    L.append(f"| 分片对比 | 冷启动 MLP-频次 | +{m['分片对比']['冷启动_diff']*100:.1f} 点 |\n")
    L.append(f"| 疲劳识别 | DDD 留来源 / val MAE / 参数 | {m['疲劳识别']['DDD留来源准确率']*100:.1f}% / {m['疲劳识别']['val_mae']} / {m['疲劳识别']['params']} |\n")
    L.append(f"| 数据对齐 | 对齐覆盖率 / 高德属性 | {m['数据对齐']['对齐覆盖率']}% / {m['数据对齐']['高德属性对齐']} 条 |\n")
    fp = os.path.join(DATA, 'model', 'scenario_report.md')
    open(fp, 'w', encoding='utf-8').write(''.join(L))
    print(f'报告已写入: {fp}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true', help='跳过 LLM 通道(规则题快跑)')
    args = ap.parse_args()
    run(use_llm=not args.fast)
