# 阶段五/01: 完整演示系统后端——FastAPI 包 push_decide + 语义图/画像/疲劳/指标 HTTP API
# 环境: D:/dachuang-env/Scripts/python.exe（torch/fastapi/scipy）；疲劳感知走 mp-env 子进程
# 启动:  python demo/server.py [--port 8000]
# 前端:  demo/frontend/（Vue3+Vite+Leaflet，dev 模式 http://localhost:5173，构建后由本服务托管 /dist）
import sys, os, json, time, glob, shutil, subprocess, argparse, math
import datetime as dt
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.spatial import cKDTree
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
DATA = os.path.join(os.path.dirname(ROOT), 'data')
sys.path.insert(0, os.path.join(ROOT, 'model'))
from push_decide import decide, run_fatigue  # noqa: E402

app = FastAPI(title='车载POI动态推送 · 完整演示系统')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

GRAPH_OUT = os.path.join(DATA, 'graph_out')
ALIGNED = os.path.join(DATA, 'aligned')
MODEL_OUT = os.path.join(DATA, 'model')
PROFILES = os.path.join(ALIGNED, 'user_profiles')
FATIGUE_PRESETS = os.path.join(DATA, 'fatigue', 'demo_frames')

# ---------------- 启动时构建索引 ----------------
NODES, NODE_BY_ID, AMAP_ATTRS, TREE = [], {}, {}, None
ROADS, ROAD_TREE = [], None  # 道路段（地图渲染路网）
ROAD_GRAPH = None  # 段级邻接图（ROAD_CONN 权威连通 + 逐对端点求路口坐标），轨迹贴路用
PROF_SUMMARIES = []
LLM_TIMES = []  # LLM 通道耗时采样（端到端耗时指标）
TOTAL_TIMES = []  # 全链路耗时采样


def build_index():
    global NODES, NODE_BY_ID, AMAP_ATTRS, TREE, PROF_SUMMARIES, ROADS, ROAD_TREE
    nodes = json.load(open(os.path.join(GRAPH_OUT, 'nodes.json'), encoding='utf-8'))
    for n in nodes:
        if n.get('type') == 'poi':
            NODES.append(n)
            NODE_BY_ID[n['node_id']] = n
    TREE = cKDTree(np.array([[n['lng'], n['lat']] for n in NODES]))
    ap = os.path.join(ALIGNED, 'poi_amap_attrs.json')
    if os.path.exists(ap):
        AMAP_ATTRS = json.load(open(ap, encoding='utf-8'))  # osm_id -> attrs
    # 道路段（含端点，供地图画路网）
    rp = os.path.join(GRAPH_OUT, 'clean_roads.jsonl')
    if os.path.exists(rp):
        ROADS = [json.loads(l) for l in open(rp, encoding='utf-8')]
        ROAD_TREE = cKDTree(np.array([[r['lng'], r['lat']] for r in ROADS]))
    build_road_graph()  # 完整车行路网（roads_all，轨迹贴路用）
    from collections import Counter
    for fp in sorted(glob.glob(os.path.join(PROFILES, '*.json'))):
        p = json.load(open(fp, encoding='utf-8'))
        cats = Counter(s['node'] for s in p['seq'] if s['node_type'] == 'poi')
        node_cats = Counter(NODE_BY_ID.get(n, {}).get('category', '') for n in cats)
        PROF_SUMMARIES.append({
            'user_id': p['user_id'], 'n_stays': p['n_stays'],
            'days': [p['days'][0], p['days'][-1]],
            'top_cats': [c for c, _ in node_cats.most_common(3) if c],
        })


# ---------------- 路网贴路（轨迹 5 分钟采样点之间沿完整车行路网连线，不穿街区） ----------------
def build_road_graph():
    global ROAD_GRAPH
    seg_fp = os.path.join(GRAPH_OUT, 'roads_all.jsonl')
    conn_fp = os.path.join(GRAPH_OUT, 'roads_all_conn.json')
    # 仓库分发时大文件走 zip（GitHub 100MB 限制），首启自动解压
    if not os.path.exists(seg_fp) and os.path.exists(seg_fp + '.zip'):
        import zipfile
        with zipfile.ZipFile(seg_fp + '.zip') as z:
            z.extract('roads_all.jsonl', GRAPH_OUT)
        print('已从 roads_all.jsonl.zip 解压完整路网数据')
    if not (os.path.exists(seg_fp) and os.path.exists(conn_fp)):
        return
    segs = [json.loads(l) for l in open(seg_fp, encoding='utf-8')]
    raw_adj = json.load(open(conn_fp, encoding='utf-8'))  # seg_id(str) -> [[邻居, 路口x, 路口y], ...]
    # 统一转成整数索引（segs 的下标），与 KDTree 查询结果一致
    id2idx = {s['seg_id']: i for i, s in enumerate(segs)}
    adj = {id2idx[k]: [(id2idx[n[0]], n[1], n[2]) for n in v if n[0] in id2idx]
           for k, v in raw_adj.items() if k in id2idx}
    ROAD_GRAPH = {
        'segs': segs,
        'adj': adj,
        'tree': cKDTree(np.array([[s['lng'], s['lat']] for s in segs])),
    }
    print(f'完整路网贴路图就绪: 段 {len(segs)} / 连通段对 {sum(len(v) for v in adj.values()) // 2}')


def _snap_seg(p, coslat, radius=90):
    """GPS 点吸附到最近路段：返回 (段索引, 投影点)，radius 米内无路返回 None"""
    g = ROAD_GRAPH
    _, idxs = g['tree'].query(p, k=4)
    best = None
    for i in np.atleast_1d(idxs):
        r = g['segs'][int(i)]
        ax, ay, bx, by = r['x1'], r['y1'], r['x2'], r['y2']
        abx, aby = bx - ax, by - ay
        denom = abx * abx + aby * aby
        if denom == 0:
            continue
        t = ((p[0] - ax) * abx + (p[1] - ay) * aby) / denom
        t = max(0.0, min(1.0, t))
        px, py = ax + t * abx, ay + t * aby
        dd = ((p[0] - px) * coslat) ** 2 + (p[1] - py) ** 2
        if best is None or dd < best[0]:
            best = (dd, int(i), px, py)
    if best is None or best[0] > (radius / 111320.0) ** 2:
        return None
    return best[1], best[2], best[3]


def _road_path_between(pa, pb, _depth=0):
    """两点间沿路网连线：A*（f=g+h，h=段中点到目标直线距，g=段数），
    贪心 DFS 在密集路网死胡同里会烧光预算，A* 按最优优先扩展更稳。
    一级失败后中点插桩分段寻路（长间隙端点吸附差时有效）。
    返回坐标序列（首尾为两点在路上的投影）；失败返回 None（调用方直线兜底）"""
    g = ROAD_GRAPH
    if g is None:
        return None
    coslat = math.cos(math.radians((pa[1] + pb[1]) / 2))
    # 快照失败放宽到 150m 重试（GPS 噪声导致离路较远的点）
    sa = _snap_seg(pa, coslat)
    if sa is None:
        sa = _snap_seg(pa, coslat, radius=150)
    sb = _snap_seg(pb, coslat)
    if sb is None:
        sb = _snap_seg(pb, coslat, radius=150)
    if sa is None or sb is None:
        return None
    si_a, ax, ay = sa
    si_b, bx, by = sb
    if si_a == si_b:
        return [(ax, ay), (bx, by)]
    def h_d2(si):
        mx, my = g['segs'][si]['lng'], g['segs'][si]['lat']
        return ((mx - pb[0]) * coslat) ** 2 + (my - pb[1]) ** 2
    def junc(a, b):
        for n in g['adj'].get(a, ()):
            if n[0] == b:
                return n[1], n[2]
        return g['segs'][a]['lng'], g['segs'][a]['lat']
    import heapq
    H = math.hypot((pb[0] - pa[0]) * coslat, pb[1] - pa[1])  # 米级归一，与 h 同量纲
    H = max(H, 1e-6)
    came = {si_a: -1}
    gs = {si_a: 0.0}
    pq = [(h_d2(si_a) / H, 0.0, si_a)]
    found, expanded = None, 0
    while pq and expanded < 20000:
        f, gs0, cur = heapq.heappop(pq)
        if gs0 > gs.get(cur, float('inf')):
            continue
        expanded += 1
        if cur == si_b:
            found = cur
            break
        for n in g['adj'].get(cur, ()):
            nxt, g_new = n[0], gs0 + 1.0
            if g_new < gs.get(nxt, float('inf')):
                gs[nxt] = g_new
                came[nxt] = cur
                heapq.heappush(pq, (g_new + h_d2(nxt) / H, g_new, nxt))
    if found is not None:
        chain = []
        s = found
        while s != -1:
            chain.append(s)
            s = came[s]
        chain.reverse()
        path = [(ax, ay)]
        for k in range(1, len(chain)):
            path.append(junc(chain[k - 1], chain[k]))
        path.append((bx, by))
        return path
    # 一级失败: 中点插桩(放宽吸附到 300m)分段寻路
    if _depth == 0:
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        cosm = math.cos(math.radians(my))
        sw = _snap_seg((mx, my), cosm, radius=300)
        if sw is not None:
            p1 = _road_path_between(pa, (sw[1], sw[2]), _depth=1)
            p2 = _road_path_between((sw[1], sw[2]), pb, _depth=1)
            if p1 and p2 and len(p1) >= 2 and len(p2) >= 2:
                return p1[:-1] + p2
    return None


def expand_path(pts):
    """回放轨迹贴路：相邻采样点间沿路网连线并插值时间戳。
    原始采样点保留 dec 标记（决策时刻点，下采样时必保留）"""
    if ROAD_GRAPH is None or len(pts) < 2:
        return [{**p, 'dec': True} for p in pts]
    out = []
    n_fail = 0
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        path = _road_path_between((a['lng'], a['lat']), (b['lng'], b['lat']))
        if path is None or len(path) < 2:
            path = [(a['lng'], a['lat']), (b['lng'], b['lat'])]
            n_fail += 1
        ta, tb = dt.datetime.strptime(a['t'], '%Y-%m-%d %H:%M:%S'), \
                 dt.datetime.strptime(b['t'], '%Y-%m-%d %H:%M:%S')
        seg = len(path) - 1
        for j, (x, y) in enumerate(path):
            if out and j == 0:
                continue  # 与上一段终点重合
            tt = ta + (tb - ta) * (j / seg)
            out.append({'t': tt.strftime('%Y-%m-%d %H:%M:%S'), 'lng': round(x, 6),
                        'lat': round(y, 6), 'dec': j == 0 or j == seg})
    if n_fail:
        print(f'贴路直线兜底 {n_fail}/{len(pts)-1} 段（路网未连通或无路吸附）')
    return out


build_index()
print(f'索引就绪: POI {len(NODES)} / 道路段 {len(ROADS)} / 用户 {len(PROF_SUMMARIES)}'
      f' / 高德属性 {len(AMAP_ATTRS)}')

# 启动预热: MLP + LLM 进程级常驻，回放第一次决策不再冷启动（实测 8.0s → 2.0s）
import push_decide as _pd  # noqa: E402
_pd.load_mlp()
try:
    import llm_channel as _lc  # noqa: E402
    _lc.load_model()
    print('模型预热完成: MLP + Qwen2.5-0.5B 已常驻内存')
except Exception as e:
    print(f'LLM 预热跳过({e})，首次决策将冷加载')


def node_with_attrs(n):
    d = dict(n)
    att = AMAP_ATTRS.get(n['node_id'].split('_', 2)[-1])  # OSM_N_xxx -> xxx 查高德属性
    if att:
        d['amap'] = att
    return d


# ---------------- API ----------------
class DecideReq(BaseModel):
    user: str
    time: str
    lng: float
    lat: float
    safety: int = 0
    highway: str = 'secondary'
    use_llm: bool = True
    last_push_min: float = 0.0  # 行驶回放用：距上次推送的模拟分钟数


@app.get('/api/health')
def health():
    return {'ok': True, 'nodes': len(NODES), 'users': len(PROF_SUMMARIES)}


@app.get('/api/users')
def users():
    return {'users': PROF_SUMMARIES}


def prof_path(uid):
    """画像文件名是 10.json 而 user_id 是 U_10，两种形式都兼容"""
    fp = os.path.join(PROFILES, f'{uid}.json')
    if not os.path.exists(fp) and uid.startswith('U_'):
        fp = os.path.join(PROFILES, f'{uid[2:]}.json')
    return fp


@app.get('/api/user/{uid}')
def user_profile(uid: str):
    fp = prof_path(uid)
    if not os.path.exists(fp):
        raise HTTPException(404, '用户不存在')
    return json.load(open(fp, encoding='utf-8'))


@app.get('/api/traj/{uid}')
def user_traj(uid: str):
    """用户停留点坐标序列（地图轨迹展示用，附节点类别）"""
    fp = prof_path(uid)
    if not os.path.exists(fp):
        raise HTTPException(404, '用户不存在')
    p = json.load(open(fp, encoding='utf-8'))
    pts = []
    for s in p['seq']:
        n = NODE_BY_ID.get(s['node'])
        if not n:
            continue
        pts.append({'lng': n['lng'], 'lat': n['lat'], 'arrive': s['arrive'],
                    'leave': s['leave'], 'category': n.get('category', ''),
                    'node_type': s['node_type'], 'name': n.get('name', '')})
    return {'user_id': p['user_id'], 'points': pts}


DRIVE_CACHE = {}  # (uid, day) -> 回放响应（贴路展开按天缓存）


@app.get('/api/drive/{uid}')
def drive_traj(uid: str, day: str = None):
    """行驶回放轨迹：该用户真实 GPS 点序列（clean/<id>.jsonl），可指定日期。
    相邻采样点（~5 分钟间隔）之间沿路网贪心连线贴路，避免穿街区直线。
    gt_stays=当日真实停留序列，作回放评测的判卷答案（决策时刻之后司机实际去了哪）"""
    num = uid[2:] if uid.startswith('U_') else uid
    key = (num, day or '')
    if key in DRIVE_CACHE:
        return DRIVE_CACHE[key]
    fp = os.path.join(DATA, 'tdrive', 'clean', f'{num}.jsonl')
    if not os.path.exists(fp):
        raise HTTPException(404, '该用户无原始轨迹')
    pts = []
    for line in open(fp, encoding='utf-8'):
        o = json.loads(line)
        if day and not o['t'].startswith(day):
            continue
        pts.append({'t': o['t'], 'lng': o['lng'], 'lat': o['lat']})
    days = sorted({p['t'][:10] for p in pts})
    # 原始采样点先下采样（保留时间顺序），再沿路网展开贴路
    if len(pts) > 800:
        step = len(pts) / 800
        pts = [pts[int(i * step)] for i in range(800)]
    pts = expand_path(pts)
    if len(pts) > 6000:
        keep = {i for i, p in enumerate(pts) if p.get('dec')}
        step = len(pts) / 6000
        idxs = sorted(keep | {int(i * step) for i in range(6000)})
        pts = [pts[i] for i in idxs]
    gt = []
    pf = prof_path(uid)
    if os.path.exists(pf):
        prof = json.load(open(pf, encoding='utf-8'))
        for s in prof['seq']:
            if s['node_type'] != 'poi':
                continue
            if day and not s['arrive'].startswith(day):
                continue
            n = NODE_BY_ID.get(s['node'])
            gt.append({'arrive': s['arrive'], 'leave': s['leave'],
                       'category': n.get('category', '') if n else '',
                       'name': n.get('name', '') if n else ''})
    resp = {'user_id': uid, 'days': days, 'points': pts, 'gt_stays': gt}
    if len(DRIVE_CACHE) > 300:
        DRIVE_CACHE.clear()
    DRIVE_CACHE[key] = resp
    return resp


@app.get('/api/pois')
def pois_viewport(minlng: float, minlat: float, maxlng: float, maxlat: float):
    """视口内 POI（地图底图点渲染，随平移/缩放按需加载），距中心排序，上限 2000"""
    clng, clat = (minlng + maxlng) / 2, (minlat + maxlat) / 2
    rad = max(abs(maxlng - minlng), abs(maxlat - minlat)) * 0.72  # 半对角线（度，粗略）
    idx = TREE.query_ball_point([clng, clat], rad)
    out = []
    for i in idx:
        n = NODES[i]
        if minlng <= n['lng'] <= maxlng and minlat <= n['lat'] <= maxlat:
            out.append({'node_id': n['node_id'], 'lng': n['lng'], 'lat': n['lat'],
                        'name': n.get('name', ''), 'category': n.get('category', ''),
                        'group': n.get('group', 'destination')})
    out.sort(key=lambda p: (p['lng'] - clng) ** 2 + (p['lat'] - clat) ** 2)
    return {'center': [clng, clat], 'count': len(out), 'pois': out[:2000]}


@app.get('/api/nearby')
def nearby(lng: float, lat: float, r: float = 800):
    """半径内 POI + 道路段（地图渲染用），r 单位米，上限 5000m"""
    r = min(max(r, 100), 5000)
    idx = TREE.query_ball_point([lng, lat], r / 111320.0)
    out = [node_with_attrs(NODES[i]) for i in idx]
    out.sort(key=lambda n: (n['lng'] - lng) ** 2 + (n['lat'] - lat) ** 2)
    roads = []
    if ROAD_TREE is not None:
        ridx = ROAD_TREE.query_ball_point([lng, lat], r / 111320.0)
        for i in ridx[:4000]:
            rr = ROADS[i]
            roads.append({'x1': rr['x1'], 'y1': rr['y1'], 'x2': rr['x2'], 'y2': rr['y2'],
                          'highway': rr.get('highway', '')})
    return {'center': [lng, lat], 'radius_m': r, 'pois': out[:1500], 'roads': roads}


@app.post('/api/decide')
def api_decide(req: DecideReq):
    t0 = time.perf_counter()
    try:
        out = decide(req.user, req.time, req.lng, req.lat, safety=req.safety,
                     highway=req.highway, use_llm=req.use_llm,
                     last_push_min=req.last_push_min)
    except Exception as e:  # 决策链路异常不再吐 HTML 堆栈，给前端可读信息
        raise HTTPException(500, detail=f'决策链路异常: {e}')
    out['elapsed_ms'] = round((time.perf_counter() - t0) * 1000, 1)
    TOTAL_TIMES.append(out['elapsed_ms'])
    if req.use_llm:
        LLM_TIMES.append(out['elapsed_ms'])
    return out


@app.post('/api/fatigue/images')
def fatigue_images(files: list[UploadFile] = File(...)):
    """上传帧序列 → mp-env 疲劳推理（PERCLOS + safety_level）。中文路径坑：一律临时 ASCII 目录"""
    tmp = os.path.join(os.environ.get('TEMP', '.'), 'fatigue_demo')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(files[:300]):
        ext = os.path.splitext(f.filename or 'x.jpg')[1] or '.jpg'
        with open(os.path.join(tmp, f'{i:03d}{ext}'), 'wb') as fo:
            fo.write(f.file.read())
    if not os.listdir(tmp):
        raise HTTPException(400, '未收到帧')
    sig = run_fatigue(tmp)
    return {'n_frames': len(os.listdir(tmp)), 'signal': sig}


@app.get('/api/fatigue/presets')
def fatigue_presets():
    """预置帧目录（演示一键切换清醒/困倦）"""
    out = {}
    for name in sorted(os.listdir(FATIGUE_PRESETS)):
        d = os.path.join(FATIGUE_PRESETS, name)
        if os.path.isdir(d):
            out[name] = len(os.listdir(d))
    return {'presets': out}


@app.post('/api/fatigue/presets/{name}')
def fatigue_preset_infer(name: str):
    d = os.path.join(FATIGUE_PRESETS, name)
    if not os.path.isdir(d):
        raise HTTPException(404, '预置不存在')
    t0 = time.perf_counter()
    sig = run_fatigue(d)
    sig['elapsed_ms'] = round((time.perf_counter() - t0) * 1000, 1)
    return {'preset': name, 'signal': sig}


@app.get('/api/metrics')
def metrics():
    """四项指标 + 消融占位，全部读实测 JSON 报告"""
    def rd(path):
        return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else None
    align = rd(os.path.join(ALIGNED, 'align_stats.json'))
    mlp = rd(os.path.join(MODEL_OUT, 'train_report.json'))
    eye = rd(os.path.join(MODEL_OUT, 'ext_test_ddd.json'))
    eye_rgb = rd(os.path.join(MODEL_OUT, 'eye_rgb_report.json'))
    merge = rd(os.path.join(ALIGNED, 'merge_stats.json'))
    replay = rd(os.path.join(MODEL_OUT, 'replay_eval.json'))
    return {
        '对齐覆盖率': align.get('align_coverage_pct') if align else None,
        '感知准确率': {
            'mlp': mlp, 'fatigue': {
                'heldout_video_acc': eye['student']['heldout_video_acc'] if eye else None,
                'frame_val_mae': eye_rgb['val_mae'] if eye_rgb else None,
            },
        },
        '端到端耗时': {
            'last_ms': TOTAL_TIMES[-1] if TOTAL_TIMES else None,
            'mean_ms': round(float(np.mean(TOTAL_TIMES)), 1) if TOTAL_TIMES else None,
            'n_samples': len(TOTAL_TIMES),
        },
        '高德属性': merge,
        '回放评测': replay,
        'graph': {'nodes': len(NODES), 'edges': None},
    }


# ---------------- 静态前端（构建产物） ----------------
DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'dist')
if os.path.isdir(DIST):
    app.mount('/', StaticFiles(directory=DIST, html=True), name='frontend')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=args.port)
