<script setup>
// 地图主舞台：高德标准底图 + 视口 POI + 停留地标记 + 行驶回放（蓝色尾迹+车辆）+ 候选高亮/推荐标记 + 图例
// 坐标：后端数据 WGS-84 → 渲染前转 GCJ-02 对齐高德瓦片（toLeaflet）
import { ref, onMounted, onBeforeUnmount, watch } from 'vue'
import L from 'leaflet'
import { GROUP_COLOR } from '../api'
import { toLeaflet, gcj2wgs } from '../geo'

const props = defineProps(['center', 'traj', 'nearby', 'viewpois', 'result', 'drive'])
const emit = defineEmits(['mapClick', 'viewport'])

const el = ref(null)
let map, trajLayer, poiLayer, candLayer, recLayer, posMarker
let driveLayer, carMarker, carEl, lastDeg = null
let trailLine = null
let vpTimer = null, lastVp = null, isPanning = false

function colorOf(g) { return GROUP_COLOR[g] || '#7a7a7a' }

onMounted(() => {
  const c = props.center
  const [gLat, gLng] = (c && c.length === 2) ? toLeaflet(c[1], c[0]) : [39.94, 116.34]
  map = L.map(el.value, { zoomControl: true }).setView([gLat, gLng], 14)
  // 高德标准彩色底图（style=7，自带路网+路名；style=8 是灰色底图，灰线观感差）
  L.tileLayer('https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
    { subdomains: ['1', '2', '3', '4'], maxZoom: 18, attribution: '© 高德地图' }).addTo(map)
  trajLayer = L.layerGroup().addTo(map)
  poiLayer = L.layerGroup().addTo(map)
  candLayer = L.layerGroup().addTo(map)
  recLayer = L.layerGroup().addTo(map)
  driveLayer = L.layerGroup().addTo(map)
  map.on('click', e => {
    if (props.drive && props.drive.playing) return  // 行驶中忽略点选
    if (posMarker) map.removeLayer(posMarker)
    posMarker = L.circleMarker(e.latlng, { radius: 7, color: '#111', weight: 2,
      fillColor: '#fff', fillOpacity: 1 }).addTo(map)
    const [wLng, wLat] = gcj2wgs(e.latlng.lng, e.latlng.lat)  // 点选位置转回 WGS 给后端
    emit('mapClick', +wLng.toFixed(6), +wLat.toFixed(6))
  })
  // 视口变化 → 按需加载 POI（防抖 250ms；缩小过头就清空；中心几乎没动就不重载）
  map.on('moveend', () => {
    isPanning = false
    clearTimeout(vpTimer)
    vpTimer = setTimeout(() => {
      if (map.getZoom() < 13) { emit('viewport', null); lastVp = null; return }
      const c = map.getCenter(), z = map.getZoom()
      if (lastVp && lastVp.z === z &&
          map.project(c, z).distanceTo(map.project(lastVp.c, z)) < 60) {
        return  // 小抖动（如跟随平移后的轻微移动）不重载视口
      }
      lastVp = { z, c }
      const b = map.getBounds()
      const sw = gcj2wgs(b.getWest(), b.getSouth())  // 视口转回 WGS 查后端
      const ne = gcj2wgs(b.getEast(), b.getNorth())
      emit('viewport', { west: sw[0], east: ne[0], south: sw[1], north: ne[1] })
    }, 250)
  })
  drawTraj()
  drawPoi()
  updateTrail()
  map.fire('moveend')  // 初始视口加载
})

// ---------- 常停留地（只画点不连线，避免跨城折线误导；回放中隐藏） ----------
function drawTraj() {
  if (!trajLayer || !props.traj) return
  trajLayer.clearLayers()
  if (props.drive && props.drive.playing) return
  const pts = (props.traj.points || []).filter(p => p.node_type === 'poi')
  if (!pts.length) return
  for (const p of pts) {
    L.circleMarker(toLeaflet(p.lng, p.lat), { radius: 4.5, color: '#fff', weight: 1,
      fillColor: '#64748b', fillOpacity: .85 })
      .bindTooltip(`${p.name || 'POI'} · ${p.category} · ${p.arrive}`).addTo(trajLayer)
  }
}

// ---------- 视口 POI（高缩放级别显示名称） ----------
function drawPoi() {
  if (!poiLayer) return
  poiLayer.clearLayers()
  const ps = props.viewpois || []
  const z = map.getZoom()
  const showLabel = z >= 16 && ps.length <= 150
  for (const p of ps) {
    const m = L.circleMarker(toLeaflet(p.lng, p.lat), {
      radius: z >= 15 ? 5 : 4, color: '#fff', weight: 1,
      fillColor: colorOf(p.group), fillOpacity: .8,
    })
    if (showLabel && p.name) {
      m.bindTooltip(p.name, { permanent: true, direction: 'right', className: 'poi-label' })
    }
    m.addTo(poiLayer)
  }
}

// ---------- 行驶回放：车身后蓝色尾迹（原地更新，不整条重建，避免闪现） ----------
function updateTrail() {
  if (!driveLayer) return
  const pts = props.drive.points
  if (!pts || !pts.length || props.drive.idx < 1) {
    driveLayer.clearLayers()
    trailLine = null
    return
  }
  const i = props.drive.idx
  const win = Math.min(200, Math.max(40, Math.round(pts.length / 40)))
  const tail = pts.slice(Math.max(0, i - win), i + 1).map(p => toLeaflet(p.lng, p.lat))
  const c = props.center
  if (c && c.length === 2) tail.push(toLeaflet(c[1], c[0]))  // 尾迹端接插值中的车辆
  if (trailLine) {
    trailLine.setLatLngs(tail)
  } else {
    driveLayer.clearLayers()
    trailLine = L.polyline(tail, { color: '#2563eb', weight: 4, opacity: .9 }).addTo(driveLayer)
  }
}

function carIcon(deg) {
  return L.divIcon({
    className: 'car-icon',
    html: `<div class="car" style="transform: rotate(${deg.toFixed(0)}deg)">➤</div>`,
    iconSize: [28, 28], iconAnchor: [14, 14],
  })
}

// 车辆位置每 100ms 由 App 插值推进（props.center），这里只负责渲染
function updateCar() {
  if (!map || !props.drive || !props.drive.points || !props.drive.points.length) {
    if (carMarker) { map.removeLayer(carMarker); carMarker = null; carEl = null; lastDeg = null }
    return
  }
  const pts = props.drive.points
  const i = props.drive.idx
  const p = pts[i]
  const c = props.center
  const [lat, lng] = (c && c.length === 2) ? c : [p.lat, p.lng]  // 插值坐标，平滑滑行
  const prev = pts[Math.max(0, i - 1)]
  const deg = Math.atan2(p.lng - prev.lng, p.lat - prev.lat) * 180 / Math.PI
  if (!carMarker) {
    lastDeg = deg
    carMarker = L.marker(toLeaflet(lng, lat), { icon: carIcon(deg) }).addTo(map)
    carEl = carMarker.getElement() && carMarker.getElement().querySelector('.car')
  } else {
    carMarker.setLatLng(toLeaflet(lng, lat))
    if (carEl && Math.abs(deg - lastDeg) > 2) {
      lastDeg = deg
      carEl.style.transform = `rotate(${deg.toFixed(0)}deg)`  // 直接转 DOM，不重建图标
    }
  }
  updateTrail()
  if (props.drive.follow && props.drive.playing) followCar()
}

// 跟随视角：车辆贴近屏幕边缘（20% 区域外）才平移一次，不每帧 setView 整图抖动
function followCar() {
  if (isPanning || !carMarker) return
  const pt = map.latLngToContainerPoint(carMarker.getLatLng())
  const s = map.getSize()
  if (pt.x < s.x * 0.2 || pt.x > s.x * 0.8 || pt.y < s.y * 0.2 || pt.y > s.y * 0.8) {
    isPanning = true
    map.panTo(carMarker.getLatLng(), { animate: true, duration: 0.4, noMoveStart: true })
  }
}

// ---------- 决策可视化：候选绿/红圈 + 本次推送金圈 ----------
function drawResult() {
  if (!candLayer || !recLayer) return
  candLayer.clearLayers(); recLayer.clearLayers()
  const r = props.result
  if (!r) return
  const byId = {}
  for (const p of (props.viewpois || [])) byId[p.node_id] = p
  for (const p of (props.nearby || [])) byId[p.node_id] = p
  for (const c of (r.candidates || [])) {
    const n = byId[c.poi_id]
    if (!n) continue
    const blocked = (r.constraint_blocked || []).some(b => b.poi === c.poi_id)
    L.circleMarker(toLeaflet(n.lng, n.lat), { radius: 9, color: blocked ? '#f87171' : '#4ade80',
      weight: 2.5, fillOpacity: 0.25 })
      .bindTooltip(`${c.name} · ${c.dist_m}m ${blocked ? '· 被约束拦截' : ''}`).addTo(candLayer)
  }
  const d = r.decision
  if (d && d.poi_id && byId[d.poi_id]) {
    const n = byId[d.poi_id]
    L.circleMarker(toLeaflet(n.lng, n.lat), { radius: 11, color: '#f59e0b', weight: 3,
      fillColor: '#fbbf24', fillOpacity: .85 })
      .bindPopup(`<b>${n.name || d.poi_id}</b><br>${d.reason || ''}`).addTo(recLayer).openPopup()
  }
}

watch(() => props.traj, drawTraj)
watch(() => props.drive && props.drive.playing, (playing) => {
  drawTraj()
  if (playing && carMarker) {
    // 播放开始时一次性对准车辆并保证可读缩放，之后交给 followCar 平移
    map.setView(carMarker.getLatLng(), Math.max(map.getZoom(), 15), { animate: false })
  }
})
watch(() => props.viewpois, drawPoi)
watch(() => props.result, drawResult)
watch(() => props.drive && props.drive.points, () => { trailLine = null; updateTrail() })
// 平滑移动：frac 每 100ms 变一次
watch(() => props.drive && (props.drive.idx + (props.drive.frac || 0)), updateCar)
watch(() => props.center, c => {
  if (map && (!props.drive || !props.drive.playing) && c && c.length === 2) {
    map.setView(toLeaflet(c[1], c[0]), Math.max(map.getZoom(), 14))
  }
})

onBeforeUnmount(() => { if (map) map.remove() })
</script>

<template>
  <div ref="el" class="map"></div>
  <div class="legend">
    <div class="lg-title">图例</div>
    <div><span class="dot" style="background:#2f6fed"></span>即时需求类 POI</div>
    <div><span class="dot" style="background:#e88a1a"></span>目的地类 POI</div>
    <div><span class="dot" style="background:#7a3fd0"></span>锚点类 POI</div>
    <div><span class="dot" style="background:#64748b"></span>用户常停留地</div>
    <div><span class="lg-line"></span>已行驶轨迹（蓝色尾迹）</div>
    <div><span class="lg-car">➤</span>车辆当前位置</div>
    <div><span class="lg-ring" style="border-color:#4ade80"></span>候选·通过约束</div>
    <div><span class="lg-ring" style="border-color:#f87171"></span>候选·被拦截</div>
    <div><span class="lg-ring" style="border-color:#f59e0b;background:rgba(251,191,36,.5)"></span>本次推送的 POI</div>
  </div>
</template>

<style scoped>
.map { position: absolute; inset: 0; }
.car { font-size: 20px; color: #e11d48; text-shadow: 0 0 3px #fff; line-height: 1.3; }

/* 图例（地图左下角） */
.legend {
  position: absolute; bottom: 20px; left: 10px; z-index: 1000;
  background: rgba(17, 24, 39, .85); color: #e5e7eb;
  padding: 8px 12px; border-radius: 8px; font-size: 12px; line-height: 1.9;
  box-shadow: 0 2px 10px rgba(0, 0, 0, .4);
}
.lg-title { font-weight: 700; margin-bottom: 2px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  border: 1px solid #fff; margin-right: 6px; vertical-align: middle; }
.lg-line { display: inline-block; width: 22px; height: 0; margin-right: 6px;
  border-top: 3px solid #2563eb; vertical-align: middle; }
.lg-car { display: inline-block; width: 12px; color: #e11d48; font-size: 12px;
  margin: 0 4px 0 2px; vertical-align: middle; }
.lg-ring { display: inline-block; width: 11px; height: 11px; border-radius: 50%;
  border: 2px solid; margin-right: 6px; vertical-align: middle; }
</style>

<style>
/* POI 名称标注（Leaflet 永久 tooltip，需全局样式） */
.poi-label {
  background: transparent; border: none; box-shadow: none;
  color: #1f2937; font-size: 11px; padding: 0;
  text-shadow: 0 0 3px #fff, 0 0 3px #fff;
}
.poi-label::before { display: none; }
</style>
