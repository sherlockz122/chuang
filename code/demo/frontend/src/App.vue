<script setup>
// 单应用单屏：地图主舞台 + 行驶回放（真实 GPS 轨迹上动态决策）+ 推送流 + 决策链路 + 指标抽屉
import { ref, reactive, computed, onMounted, onBeforeUnmount } from 'vue'
import { api, CAT_NAMES } from './api'
import TopBar from './components/TopBar.vue'
import LeftPanel from './components/LeftPanel.vue'
import MapStage from './components/MapStage.vue'
import StepBar from './components/StepBar.vue'
import PushCard from './components/PushCard.vue'
import MetricsDrawer from './components/MetricsDrawer.vue'
import DriveBar from './components/DriveBar.vue'

const users = ref([])
const prof = ref(null)
const traj = ref(null)
const nearby = ref([])
const viewpois = ref([])   // 视口内 POI（随地图平移/缩放按需加载）
const result = ref(null)
const metrics = ref(null)
const showMetrics = ref(false)
const busy = ref(false)
const error = ref('')
const toast = ref('')
let toastTimer = null

const state = reactive({
  uid: 'U_318',
  time: '2008-02-04 10:30:00',
  lng: 116.34, lat: 39.94,
  safety: 0,
  highway: 'secondary',
  useLlm: true,
  fatigue: null,          // 疲劳信号（预置推理结果）
  fatigueName: '清醒',     // 当前疲劳场景名
  presets: {},
})

// ---------- 行驶回放：车辆沿真实 GPS 轨迹行驶，途中定频跑完整决策链 ----------
const drive = reactive({
  days: [], day: '',       // 可选日期（clean 轨迹按天）
  points: [], idx: 0,      // 当日轨迹点与当前进度
  gtStays: [],             // 当日真实停留（回放评测判卷答案）
  frac: 0,                 // 当前段插值进度 0~1（平滑移动）
  segMin: 0,               // 当前段内已推进的模拟分钟（匀速播放计时）
  playing: false,
  speed: 1.5,              // 播放速度 = 模拟分钟/真实秒（默认 1.5，观感适中）
  follow: true,            // 地图跟随车辆
  accMin: 0,               // 距上次决策累计的模拟分钟
  lastPushT: null,         // 上次实际推送的模拟时刻
  lastPushPoi: null,       // 上次推送的 POI（防重复打扰）
  busyDecide: false,
  nPush: 0, nBlock: 0, nSilent: 0,
  nGrad: 0, nHit: 0,       // 判卷次数 / 预测命中次数
  ended: false,            // 回放是否已自然结束（结束后按钮变"重播"，从头再来）
})
const TICK_MS = 100         // 回放节拍：每 100ms 推进一次
let tickTimer = null
let tickGen = 0             // 回放代次：stop/重启时 +1，残留定时器回调凭此自检退出
let lastKind = null         // 上一次决策的形态(push/block/silent)，用于提示节流
let blockStreak = 0         // 连续拦截次数（合并显示，防刷屏）

const mapCenter = computed(() => [state.lat, state.lng])

function showToast(msg, ms = 3500) {
  toast.value = msg
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { toast.value = '' }, ms)
}

function simMinBetween(a, b) {
  const pa = a.split(/[- :]/).map(Number)
  const pb = b.split(/[- :]/).map(Number)
  return (Date.UTC(pb[0], pb[1] - 1, pb[2], pb[3], pb[4], pb[5])
        - Date.UTC(pa[0], pa[1] - 1, pa[2], pa[3], pa[4], pa[5])) / 60000
}

function interpTime(a, b, frac) {
  const pa = a.split(/[- :]/).map(Number)
  const pb = b.split(/[- :]/).map(Number)
  const da = Date.UTC(pa[0], pa[1] - 1, pa[2], pa[3], pa[4], pa[5])
  const db = Date.UTC(pb[0], pb[1] - 1, pb[2], pb[3], pb[4], pb[5])
  const d = new Date(da + (db - da) * frac)
  const z = n => String(n).padStart(2, '0')
  return `${d.getUTCFullYear()}-${z(d.getUTCMonth() + 1)}-${z(d.getUTCDate())} ` +
         `${z(d.getUTCHours())}:${z(d.getUTCMinutes())}:${z(d.getUTCSeconds())}`
}

// 回放评测：决策时刻 t 之后 3 小时内司机真实到达的下一个停留点（判卷答案）
function nextStayWithin(t, hours = 3) {
  for (const s of drive.gtStays) {
    if (s.arrive > t && simMinBetween(t, s.arrive) <= hours * 60) return s
  }
  return null
}

async function loadUsers() {
  const d = await api.users()
  users.value = d.users
}

async function selectUser(uid) {
  stopReplay()
  state.uid = uid
  state.lng = 116.34; state.lat = 39.94
  result.value = null
  error.value = ''
  try {
    const [p, t, dr] = await Promise.all([api.profile(uid), api.traj(uid), api.drive(uid)])
    prof.value = p
    traj.value = t
    drive.days = dr.days
    // 默认演示天：离线回放评测里判卷多、命中多的天（U_318=02-04），否则取最后一天
    const DEMO_DAYS = { U_318: '2008-02-04' }
    drive.day = DEMO_DAYS[uid] && dr.days.includes(DEMO_DAYS[uid])
      ? DEMO_DAYS[uid] : (dr.days.length ? dr.days[dr.days.length - 1] : '')
    // 时间默认取该用户最后一天
    const days = p.days || []
    state.time = (days.length ? days[days.length - 1] : '2008-02-08') + ' 10:30:00'
    await loadNearby()
  } catch (e) { error.value = String(e) }
}

async function loadNearby() {
  const d = await api.nearby(state.lng, state.lat, 800)
  nearby.value = d.pois
}

async function loadViewport(bounds) {
  if (!bounds) { viewpois.value = []; return }
  const b = { minlng: bounds.west, minlat: bounds.south, maxlng: bounds.east, maxlat: bounds.north }
  const d = await api.pois(b)
  viewpois.value = d.pois
}

// ---------- 回放控制 ----------
function stopReplay(msg) {
  drive.playing = false
  tickGen++                      // 使任何已排队/残留的 tick 回调立即失效
  clearTimeout(tickTimer)
  if (msg) showToast(msg, 6000)
}

function stopReset() {
  stopReplay()
  drive.points = []
  drive.idx = 0
  drive.frac = 0
  drive.segMin = 0
  drive.ended = false
}

async function startReplay() {
  busy.value = true
  error.value = ''
  try {
    const d = await api.drive(state.uid, drive.day)
    drive.points = d.points
    if (!drive.points.length) { showToast(`${drive.day} 无轨迹记录`); return }
    drive.gtStays = d.gt_stays || []
    drive.idx = 0
    drive.frac = 0
    drive.segMin = 0
    drive.accMin = 0
    drive.nPush = drive.nBlock = drive.nSilent = 0
    drive.nGrad = drive.nHit = 0
    drive.lastPushT = null
    drive.lastPushPoi = null
    lastKind = null
    blockStreak = 0
    drive.ended = false
    result.value = null
    const p0 = drive.points[0]
    state.lng = p0.lng; state.lat = p0.lat; state.time = p0.t
    loadNearby()
    drive.playing = true
    showToast('回放评测中：车辆沿该用户当日真实 GPS 轨迹行驶（时间加速）。每约 20 个模拟分钟系统做一次推送决策，并拿司机后来的真实停留当判卷答案打分。绿色卡=推送（命中时带 ✓ 事后验证），红色卡=被安全约束拦截。', 10000)
    scheduleTick()
  } catch (e) { error.value = String(e) } finally { busy.value = false }
}

function togglePlay() {
  if (drive.playing) { drive.playing = false; tickGen++; clearTimeout(tickTimer) }
  else if (!drive.ended) { drive.playing = true; scheduleTick() }  // 结束后不原地恢复，必须重播
}

function onDrivePlay() {
  if (drive.playing) togglePlay()
  else if (drive.ended) startReplay()      // 已播完：重播 = 从头开始
  else if (drive.points.length) togglePlay()  // 暂停后恢复
  else startReplay()                          // 首次/重置后启动
}

function scheduleTick() {
  if (!drive.playing) return
  const gen = tickGen
  tickTimer = setTimeout(() => {
    if (gen !== tickGen) return   // 已被 stop/暂停/重播作废的残留回调，直接退出
    tick()
  }, TICK_MS)
}

function tick() {
  if (!drive.playing || !drive.points.length) return
  try {
    // 匀速播放: 每帧按 速度(模拟分钟/真实秒) 推进模拟时间，跨点按时间比例插值——
    // 视觉车速恒定，不再因相邻点时间差不同而忽快忽慢
    const dtSim = drive.speed * TICK_MS / 1000
    drive.accMin += dtSim
    drive.segMin += dtSim
    const n = drive.points.length
    while (drive.idx < n - 1) {
      const prev = drive.points[drive.idx]
      const next = drive.points[drive.idx + 1]
      let segMin = simMinBetween(prev.t, next.t)
      if (!isFinite(segMin)) segMin = 0.2      // 时间戳异常兜底，防 NaN 卡死
      segMin = Math.max(segMin, 0.02)
      if (drive.segMin >= segMin) {
        drive.segMin -= segMin
        drive.idx++
        if (drive.idx % 6 === 0) loadNearby()  // 随车刷新周边 POI
        continue
      }
      const frac = drive.segMin / segMin
      drive.frac = frac
      state.lng = prev.lng + (next.lng - prev.lng) * frac
      state.lat = prev.lat + (next.lat - prev.lat) * frac
      state.time = interpTime(prev.t, next.t, frac)
      break
    }
    if (drive.idx >= n - 1) {
      drive.ended = true
      const score = drive.nGrad > 0 ? ` · 判卷 ${drive.nGrad} 次命中 ${drive.nHit} 次（${Math.round(drive.nHit / drive.nGrad * 100)}%）` : ''
      stopReplay(`回放结束：推送 ${drive.nPush} · 拦截 ${drive.nBlock} · 静默 ${drive.nSilent}${score}`)
      return
    }
    // 每累计 20 个模拟分钟做一次推送决策（按真实推进的模拟时间触发）
    if (drive.accMin >= 20 && !drive.busyDecide) {
      drive.accMin = 0
      const lastPushMin = drive.lastPushT ? simMinBetween(drive.lastPushT, state.time) : 0
      fireDecide(state.time, lastPushMin)
    }
    scheduleTick()
  } catch (e) {
    // 任何异常都不允许回放死循环或静默卡死：停表并提示
    stopReplay('回放异常终止：' + String(e))
  }
}

async function fireDecide(tStr, lastPushMin) {
  drive.busyDecide = true
  try {
    const d = await api.decide({
      user: state.uid, time: tStr, lng: state.lng, lat: state.lat,
      safety: state.safety, highway: state.highway,
      use_llm: state.useLlm, last_push_min: lastPushMin,
    })
    // ---- 回放评测：拿司机后来的真实停留判卷 ----
    const ns = nextStayWithin(tStr)
    if (ns) {
      drive.nGrad++
      const top3 = (d.mlp_top3_cats || []).map(x => x[0])
      const top3Hit = top3.includes(ns.category)
      if (top3Hit) drive.nHit++
      // 每张卡都带判卷结果: 拿司机后来真实到达的停留点当答案
      d._grade = { hit: top3Hit, afterMin: Math.round(simMinBetween(tStr, ns.arrive)), stay: ns }
      const pushed = d.decision && d.decision.poi_id
        ? (d.candidates || []).find(c => c.poi_id === d.decision.poi_id) : null
      if (pushed && pushed.category === ns.category) {
        d._verify = { afterMin: Math.round(simMinBetween(tStr, ns.arrive)), stay: ns }
      }
    }
    // ---- 分类计数 + 消息节流（防刷屏） ----
    let kind
    if (d.decision && d.decision.poi_id) {
      const samePoi = drive.lastPushPoi === d.decision.poi_id && drive.lastPushT
        && simMinBetween(drive.lastPushT, tStr) < 45
      if (samePoi) {
        d._suppressed = true  // 同 POI 冷却：不重复弹卡
        kind = 'silent'
        drive.nSilent++
        if (lastKind !== 'silent') showToast(`${tStr.slice(11, 16)} 该 POI 刚推送过（冷却中），不重复打扰`, 3000)
      } else {
        kind = 'push'
        drive.nPush++
        drive.lastPushPoi = d.decision.poi_id
        drive.lastPushT = tStr
        blockStreak = 0
      }
    } else if (d.constraint_blocked && d.constraint_blocked.length) {
      kind = 'block'
      drive.nBlock++
      blockStreak++
    } else {
      kind = 'silent'
      drive.nSilent++
      if (lastKind !== 'silent') showToast(`${tStr.slice(11, 16)} 决策：无合适推送（无合规候选）`, 2600)
      blockStreak = 0
    }
    lastKind = kind
    result.value = d
    loadNearby()  // 候选/推荐标记需要周边 POI 坐标
  } catch (e) { error.value = String(e) } finally { drive.busyDecide = false }
}

async function loadPresets() {
  const d = await api.fatiguePresets()
  state.presets = d.presets
}

async function inferFatigue(name) {
  busy.value = true
  error.value = ''
  try {
    const d = await api.fatigueInferPreset(name)
    state.fatigue = d.signal
    state.fatigueName = { drowsy: '困倦', alert: '清醒' }[name] || name
    state.safety = d.signal.safety_level ?? 0
    if (!result.value) result.value = { fatigue_signal: d.signal }
    else result.value.fatigue_signal = d.signal
  } catch (e) { error.value = String(e) } finally { busy.value = false }
}

async function runDecide() {
  busy.value = true
  error.value = ''
  try {
    result.value = await api.decide({
      user: state.uid, time: state.time,
      lng: state.lng, lat: state.lat,
      safety: state.safety, highway: state.highway,
      use_llm: state.useLlm,
    })
    await loadNearby()
  } catch (e) { error.value = String(e) } finally { busy.value = false }
}

async function openMetrics() {
  if (!metrics.value) metrics.value = await api.metrics()
  showMetrics.value = true
}

function onMapClick(lng, lat) {
  state.lng = lng; state.lat = lat
  loadNearby()
}

onMounted(async () => {
  await Promise.all([loadUsers(), loadPresets()])
  await selectUser(state.uid)
})

onBeforeUnmount(() => { clearTimeout(tickTimer); clearTimeout(toastTimer) })
</script>

<template>
  <div class="app">
    <TopBar
      :users="users" :state="state"
      :result="result" :busy="busy"
      @select-user="selectUser" @decide="runDecide"
      @metrics="openMetrics"
    />
    <div class="main">
      <LeftPanel :prof="prof" :fatigue="state.fatigue" :fatigue-name="state.fatigueName"
                 :presets="state.presets" :busy="busy"
                 @infer="inferFatigue" @time="v => state.time = v"
                 :time="state.time" :highway="state.highway"
                 @highway="v => state.highway = v" :use-llm="state.useLlm"
                 @llm="v => state.useLlm = v" :safety="state.safety" />
      <div class="stage">
        <MapStage :center="mapCenter" :traj="traj" :nearby="nearby" :viewpois="viewpois"
                  :result="result" :drive="drive" @map-click="onMapClick"
                  @viewport="loadViewport" />
        <DriveBar :drive="drive" :busy="busy"
                  @play="onDrivePlay" @stop="stopReset"
                  @speed="v => drive.speed = v" @follow="v => drive.follow = v"
                  @day="v => drive.day = v" />
        <PushCard v-if="result && result.decision !== undefined" :result="result"
                  :safety="state.safety" :cat-names="CAT_NAMES" :block-streak="blockStreak" />
        <div v-if="result && result.mlp_top3_cats" class="predchip">
          📊 MLP 预测: {{ result.mlp_top3_cats.slice(0, 3).map(([c, p]) => `${CAT_NAMES[c] || c} ${Math.round(p * 100)}%`).join(' · ') }}
        </div>
        <div v-if="toast" class="toast">{{ toast }}</div>
      </div>
    </div>
    <StepBar v-if="result" :result="result" :cat-names="CAT_NAMES" />
    <div v-if="busy" class="busy">运行中…</div>
    <div v-if="error" class="err">{{ error }}</div>
    <MetricsDrawer v-if="showMetrics" :metrics="metrics" @close="showMetrics = false" />
  </div>
</template>
