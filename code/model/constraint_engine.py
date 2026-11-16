# 环节五/01: 环境约束推送决策引擎（规则层，可解释）
# 口径: 唯一模型给出"想推什么"，约束引擎决定"能不能推、什么时候推"——
#       决策合规率 = 通过约束过滤后仍合法的推送占比（四项评估指标之三）
# 疲劳安全等级来源: safety_level 参数（demo 用模拟值；UTA-RLDD 疲劳识别模型后期接入同口径）
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, 'config')
cats = yaml.safe_load(open(os.path.join(CFG, 'categories.yaml'), encoding='utf-8'))
GROUP = {c['id']: g['id'] for g in cats['groups'] for c in g['categories']}

# 安全等级 -> 允许的 POI 类别（等级越高越保守，疲劳时强制休息类）
SAFETY_ALLOW = {
    0: None,                       # 清醒: 不限制
    1: {'fuel', 'charging', 'parking', 'toilet', 'convenience', 'restaurant', 'hotel'},
    2: {'parking', 'toilet', 'convenience', 'services'},  # 高疲劳: 强制休息/如厕
}
# 深夜关闭类（营业时间约束）
NIGHT_CLOSED = {'scenic', 'sports', 'school', 'cinema', 'shopping'}
# 高速/快速路上禁推的目的地型（分心约束）
NO_DEST_ON_HIGHWAY = {'motorway', 'motorway_link', 'trunk', 'trunk_link'}
PUSH_COOLDOWN_MIN = 15.0  # 同一用户 15 分钟内最多 1 推（打扰约束）


def check(candidate_pois, ctx):
    """对候选推送逐条判定合规性。
    candidate_pois: [{category, group, ...}]
    ctx: {safety_level, highway, hour, last_push_min}（last_push_min=None 表示无上次推送）
    返回: {pass: [...], block: [{poi, rule}], reasons: [...]}
    """
    passed, blocked = [], []
    for p in candidate_pois:
        cat, grp = p['category'], p['group']
        reasons = []
        allow = SAFETY_ALLOW.get(ctx['safety_level'])
        if allow is not None and cat not in allow:
            reasons.append(f'安全等级{ctx["safety_level"]}不允许{cat}')
        if 23 <= ctx['hour'] or ctx['hour'] < 6:
            if cat in NIGHT_CLOSED:
                reasons.append(f'深夜时段({ctx["hour"]}:00)该类已关闭')
        if ctx['highway'] in NO_DEST_ON_HIGHWAY and grp == 'destination':
            reasons.append(f'高速路段不推目的地型({cat})')
        if ctx['last_push_min'] is not None and ctx['last_push_min'] < PUSH_COOLDOWN_MIN:
            reasons.append(f'距上次推送仅{round(ctx["last_push_min"], 1)}分钟(<{PUSH_COOLDOWN_MIN})')
        if reasons:
            blocked.append({'poi': p, 'rule': '; '.join(reasons)})
        else:
            passed.append(p)
    return passed, blocked


if __name__ == '__main__':
    # 自检用例
    cands = [
        {'category': 'parking', 'group': 'instant', 'name': 'A停车场'},
        {'category': 'scenic', 'group': 'destination', 'name': 'B景区'},
        {'category': 'restaurant', 'group': 'destination', 'name': 'C餐厅'},
        {'category': 'toilet', 'group': 'instant', 'name': 'D厕所'},
    ]
    cases = [
        ({'safety_level': 0, 'highway': 'secondary', 'hour': 14, 'last_push_min': None},
         '清醒+白天+次干道: 全部通过'),
        ({'safety_level': 2, 'highway': 'secondary', 'hour': 14, 'last_push_min': None},
         '高疲劳: 只留 parking/toilet'),
        ({'safety_level': 0, 'highway': 'motorway', 'hour': 2, 'last_push_min': 5.0},
         '高速+深夜+冷却期内: 全部拦截'),
    ]
    for ctx, desc in cases:
        p, b = check(cands, ctx)
        print(f'[{desc}] 通过 {[x["category"] for x in p]} | 拦截 {[(x["poi"]["category"], x["rule"]) for x in b]}')
