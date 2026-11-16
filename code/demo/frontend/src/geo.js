// WGS-84 ↔ GCJ-02 坐标转换：后端数据全是 WGS-84，高德瓦片是 GCJ-02，
// 渲染前必须转换（否则 POI/轨迹与底图道路偏移几百米）
const PI = Math.PI
const A = 6378245.0
const EE = 0.00669342162296594323

function outOfChina(lng, lat) {
  return lng < 72.004 || lng > 137.8347 || lat < 0.8293 || lat > 55.8271
}

function transformLat(x, y) {
  let r = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * Math.sqrt(Math.abs(x))
  r += (20.0 * Math.sin(6.0 * x * PI) + 20.0 * Math.sin(2.0 * x * PI)) * 2.0 / 3.0
  r += (20.0 * Math.sin(y * PI) + 40.0 * Math.sin(y / 3.0 * PI)) * 2.0 / 3.0
  r += (160.0 * Math.sin(y / 12.0 * PI) + 320 * Math.sin(y * PI / 30.0)) * 2.0 / 3.0
  return r
}

function transformLng(x, y) {
  let r = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * Math.sqrt(Math.abs(x))
  r += (20.0 * Math.sin(6.0 * x * PI) + 20.0 * Math.sin(2.0 * x * PI)) * 2.0 / 3.0
  r += (20.0 * Math.sin(x * PI) + 40.0 * Math.sin(x / 3.0 * PI)) * 2.0 / 3.0
  r += (150.0 * Math.sin(x / 12.0 * PI) + 300.0 * Math.sin(x / 30.0 * PI)) * 2.0 / 3.0
  return r
}

export function wgs2gcj(lng, lat) {
  if (outOfChina(lng, lat)) return [lng, lat]
  let dLat = transformLat(lng - 105.0, lat - 35.0)
  let dLng = transformLng(lng - 105.0, lat - 35.0)
  const radLat = lat / 180.0 * PI
  let magic = Math.sin(radLat)
  magic = 1 - EE * magic * magic
  const sqrtMagic = Math.sqrt(magic)
  dLat = (dLat * 180.0) / ((A * (1 - EE)) / (magic * sqrtMagic) * PI)
  dLng = (dLng * 180.0) / (A / sqrtMagic * Math.cos(radLat) * PI)
  return [lng + dLng, lat + dLat]
}

export function gcj2wgs(lng, lat) {
  if (outOfChina(lng, lat)) return [lng, lat]
  let wgsLng = lng, wgsLat = lat
  for (let i = 0; i < 4; i++) {
    const [gLng, gLat] = wgs2gcj(wgsLng, wgsLat)
    wgsLng -= gLng - lng
    wgsLat -= gLat - lat
  }
  return [wgsLng, wgsLat]
}

// 便捷函数：数据点 → Leaflet [lat, lng]（已转 GCJ，可直接叠加高德瓦片）
export const toLeaflet = (lng, lat) => {
  const [gLng, gLat] = wgs2gcj(lng, lat)
  return [gLat, gLng]
}
