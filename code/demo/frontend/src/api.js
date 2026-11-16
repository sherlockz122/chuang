// 后端 HTTP API 封装（dev 走 vite proxy，build 后同源）
// 非 2xx 时抛可读错误（后端 detail），而不是把 HTML 堆栈当 JSON 解析报 SyntaxError
const J = async r => {
  const d = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`)
  return d
}

export const api = {
  users: () => fetch('/api/users').then(J),
  profile: uid => fetch(`/api/user/${uid}`).then(J),
  traj: uid => fetch(`/api/traj/${uid}`).then(J),
  drive: (uid, day) => fetch(`/api/drive/${uid}${day ? `?day=${day}` : ''}`).then(J),
  nearby: (lng, lat, r) => fetch(`/api/nearby?lng=${lng}&lat=${lat}&r=${r}`).then(J),
  pois: b => fetch(`/api/pois?minlng=${b.minlng}&minlat=${b.minlat}&maxlng=${b.maxlng}&maxlat=${b.maxlat}`).then(J),
  decide: body => fetch('/api/decide', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(J),
  fatiguePresets: () => fetch('/api/fatigue/presets').then(J),
  fatigueInferPreset: name => fetch(`/api/fatigue/presets/${name}`, { method: 'POST' }).then(J),
  metrics: () => fetch('/api/metrics').then(J),
}

export const CAT_NAMES = {
  fuel: '加油站', charging: '充电站', parking: '停车场', toilet: '公共厕所',
  hospital: '医院', pharmacy: '药店', convenience: '便利店', car_repair: '修车店',
  services: '服务区', restaurant: '餐饮', shopping: '购物', scenic: '景区',
  hotel: '酒店', park: '公园广场', sports: '体育场馆', cinema: '电影院',
  transport: '交通枢纽', office: '写字楼', school: '学校',
}

export const GROUP_COLOR = { instant: '#2f6fed', destination: '#e88a1a', anchor: '#7a3fd0' }
