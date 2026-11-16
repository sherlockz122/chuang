# 环节四/03: LLM 通道（"说句子"）——Qwen2.5-0.5B 读用户语义 JSON，输出结构化推送决策
# 用法:  python llm_channel.py --user U_10 --time "2008-02-03 14:00:00" --lng 116.39 --lat 39.92
# 输出:  JSON {decision, poi_id, reason, confidence}（demo 双通道之一；JSON 与模型解耦的体现）
import sys, os, json, argparse
sys.stdout.reconfigure(encoding='utf-8')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), 'data')
MODEL_DIR = os.path.join(os.path.dirname(ROOT), 'models', 'qwen2.5-0.5b-instruct')

SYSTEM = (
    '你是车载兴趣点推荐助手。根据MLP通道的兴趣预测、用户历史停留语义和当前驾驶环境，从候选 POI 中选一个最合适的推送。'
    '要求: 1) 只从候选中选，poi_id 必须原样复制；2) 优先选择 MLP 兴趣预测类别中的候选；3) reason 用一句中文说明理由，'
    '必须结合用户历史习惯、兴趣预测或当前环境约束（如疲劳等级、高速禁停、夜间歇业），不要只重复 POI 类别名；'
    '4) confidence 按把握给 0.0~1.0；5) 只输出 JSON: {"poi_id": "...", "reason": "...", "confidence": 0.0~1.0}，不要输出其他内容。'
)

CAT_ZH = {'fuel': '加油站', 'charging': '充电站', 'parking': '停车场', 'toilet': '公共厕所',
          'hospital': '医院', 'pharmacy': '药店', 'convenience': '便利店', 'car_repair': '修车店',
          'services': '服务区', 'restaurant': '餐饮', 'shopping': '购物', 'scenic': '景区',
          'hotel': '酒店', 'park': '公园广场', 'sports': '体育场馆', 'cinema': '电影院',
          'transport': '交通枢纽', 'office': '写字楼', 'school': '学校'}

SAFETY_ZH = {0: '清醒', 1: '低警惕', 2: '疲劳'}

NODE_INFO = {}


def node_info(node_id):
    """节点 ID → {name, category}（懒加载，45k 节点只读一次）"""
    if not NODE_INFO:
        nodes = json.load(open(os.path.join(DATA, 'graph_out', 'nodes.json'), encoding='utf-8'))
        NODE_INFO.update({n['node_id']: n for n in nodes})
    return NODE_INFO.get(node_id, {})


_MODEL = None  # 进程级缓存：Qwen 0.5B 只从磁盘加载一次（回放决策频繁调用，避免每次几秒的重复加载）


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype,
                                                 trust_remote_code=True).to(device)
    model.eval()
    _MODEL = (tok, model, device)
    return _MODEL


_POI_NODES = None


def find_candidates(lng, lat, radius_m=800):
    """当前位置附近半径内的 POI（图上 SPATIAL_NEAR 邻域；简化用节点坐标直接筛）"""
    global _POI_NODES
    if _POI_NODES is None:  # 45k 节点只读一次
        nodes = json.load(open(os.path.join(DATA, 'graph_out', 'nodes.json'), encoding='utf-8'))
        _POI_NODES = [(n['node_id'], n['lng'], n['lat'], n['name'] or CAT_ZH.get(n['category'], ''),
                       n['category']) for n in nodes if n['type'] == 'poi']
    deg = 111320.0
    out = []
    for nid, nlng, nlat, nm, ncat in _POI_NODES:
        d = ((nlng - lng) * deg) ** 2 + ((nlat - lat) * deg * 0.813) ** 2
        if d <= radius_m ** 2:
            out.append({'poi_id': nid, 'name': nm, 'category': ncat, 'dist_m': round(d ** 0.5)})
    out.sort(key=lambda x: x['dist_m'])
    return out[:10]


def build_prompt(user_id, t, lng, lat, cands, ctx=None, top3=None):
    """ctx: {'safety_level', 'highway', 'hour', 'last_push_min'} 环境约束（push_decide 传入）
    top3: MLP 通道兴趣预测 [(类别, 概率), ...]——"说数字"的结果汇入"说句子"的输入"""
    prof = json.load(open(os.path.join(DATA, 'aligned', 'user_profiles',
                                       f'{user_id[2:]}.json'), encoding='utf-8'))
    def poiname(nm, cat):
        zh = CAT_ZH.get(cat, cat)
        return nm if nm == zh or not zh else f'{nm}({zh})'

    hist = []
    for s in prof['seq'][-10:]:
        info = node_info(s['node'])
        nm = info.get('name') or CAT_ZH.get(info.get('category', ''), '地点')
        hist.append(f"{s['arrive']} 在{poiname(nm, info.get('category', ''))}停留{s['dur_min']:.0f}分钟")
    cand_txt = '\n'.join(
        f"- {c['poi_id']} | {poiname(c['name'], c['category'])} | {c['dist_m']}m"
        for c in cands)
    env_txt = ''
    if ctx:
        env_txt = (f"当前环境: 疲劳等级={SAFETY_ZH.get(ctx.get('safety_level', 0), '未知')}"
                   f"({ctx.get('safety_level', 0)}级)，道路={ctx.get('highway', '普通道路')}，"
                   f"时刻={ctx.get('hour', '?')}时")
        if ctx.get('last_push_min') is not None:
            env_txt += f"，距上次推送{ctx['last_push_min']:.0f}分钟"
        env_txt += '\n'
    mlp_txt = ''
    if top3:
        items = '; '.join(f'{CAT_ZH.get(c, c)}({p * 100:.0f}%)' for c, p in top3)
        mlp_txt = f"MLP通道兴趣预测(优先参考): {items}\n"
    return (f"当前时间: {t}，位置: ({lng},{lat})\n"
            f"{env_txt}"
            f"{mlp_txt}"
            f"用户最近历史:\n" + '\n'.join(hist) + '\n'
            f"候选 POI:\n{cand_txt}\n"
            f"请给出最合适的推送决策 JSON。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True)
    ap.add_argument('--time', required=True)
    ap.add_argument('--lng', type=float, required=True)
    ap.add_argument('--lat', type=float, required=True)
    args = ap.parse_args()

    tok, model, device = load_model()
    cands = find_candidates(args.lng, args.lat)
    if not cands:
        print(json.dumps({'decision': 'none', 'reason': '附近无候选 POI'}, ensure_ascii=False))
        return
    prompt = build_prompt(args.user, args.time, args.lng, args.lat, cands)
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    resp = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    try:
        js = json.loads(resp[resp.index('{'): resp.rindex('}') + 1])
    except Exception:
        js = {'decision': 'parse_error', 'raw': resp}
    print(json.dumps(js, ensure_ascii=False))


if __name__ == '__main__':
    main()
