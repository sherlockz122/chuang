<script setup>
// 底部决策链路：五步中间产物（MLP说数字 → 候选 → 约束 → LLM定稿），点击展开细节
import { ref, computed } from 'vue'
import { CAT_NAMES, GROUP_COLOR } from '../api'
const props = defineProps(['result', 'catNames'])
const open = ref(null)

const steps = computed(() => {
  const r = props.result
  if (!r) return []
  const blockedN = (r.constraint_blocked || []).length
  return [
    { key: 'mlp', title: 'MLP 说数字', sub: (r.mlp_top3_cats || []).map(([c, p]) =>
      `${CAT_NAMES[c] || c} ${(p * 100).toFixed(0)}%`).join(' · ') },
    { key: 'cand', title: '候选召回', sub: `${r.candidates?.length || 0} 个（top3类 ∪ 互补对）` },
    { key: 'cons', title: '约束引擎', sub: blockedN ? `拦截 ${blockedN} 个` : '全部通过',
      bad: blockedN > 0 },
    { key: 'llm', title: r.decision ? (r.decision.reason || '').includes('兜底') ? 'MLP 兜底' : 'LLM 定稿' : '决策',
      sub: r.decision ? (r.decision.poi_id || '推送被拦截') : '无候选可推' },
  ]
})

// 约束详情按规则归并：同一原因拦截的候选合到一条
const consGroups = computed(() => {
  const map = new Map()
  for (const b of (props.result?.constraint_blocked || [])) {
    if (!map.has(b.rule)) map.set(b.rule, [])
    map.get(b.rule).push(b.poi)
  }
  return [...map.entries()]
})

function catColor(c) {
  // 类别→组颜色（后端无组信息时按已知映射）
  const g = { fuel: 'instant', charging: 'instant', parking: 'instant', toilet: 'instant',
    hospital: 'instant', pharmacy: 'instant', convenience: 'instant', car_repair: 'instant',
    services: 'instant', restaurant: 'destination', shopping: 'destination',
    scenic: 'destination', hotel: 'destination', park: 'destination', sports: 'destination',
    cinema: 'destination', transport: 'anchor', office: 'anchor', school: 'anchor' }[c]
  return GROUP_COLOR[g] || '#7a7a7a'
}
</script>

<template>
  <footer class="stepbar">
    <div class="steps">
      <div v-for="(s, i) in steps" :key="s.key" class="step"
           :class="{ bad: s.bad, open: open === s.key }" @click="open = open === s.key ? null : s.key">
        <div class="idx">{{ i + 1 }}</div>
        <div class="t">
          <div class="tt">{{ s.title }}</div>
          <div class="ss">{{ s.sub }}</div>
        </div>
      </div>
    </div>
    <div v-if="open" class="detail">
      <template v-if="open === 'mlp'">
        <div class="bars">
          <div v-for="[c, p] in result.mlp_top3_cats" :key="c" class="bar-row">
            <span class="bn">{{ catNames[c] || c }}</span>
            <div class="bar"><div class="fill" :style="{ width: (p * 100) + '%', background: catColor(c) }"></div></div>
            <span class="bv">{{ (p * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </template>
      <template v-else-if="open === 'cand'">
        <table class="tbl">
          <tr><th>POI</th><th>类别</th><th>距离</th><th>状态</th></tr>
          <tr v-for="c in result.candidates" :key="c.poi_id">
            <td>{{ c.name || c.poi_id }}</td>
            <td>{{ catNames[c.category] || c.category }}</td>
            <td>{{ c.dist_m }}m</td>
            <td>{{ result.constraint_blocked.some(b => b.poi === c.poi_id) ? '⛔ 拦截' : '✅ 通过' }}</td>
          </tr>
        </table>
      </template>
      <template v-else-if="open === 'cons'">
        <div v-if="!result.constraint_blocked.length" class="okline">四条约束规则全部通过，候选可推送</div>
        <div v-for="[rule, pois] in consGroups" :key="rule" class="blk">
          {{ rule }}（{{ pois.join('、') }}）
        </div>
      </template>
      <template v-else-if="open === 'llm'">
        <pre class="json">{{ JSON.stringify(result.decision, null, 1) }}</pre>
      </template>
    </div>
  </footer>
</template>

<style scoped>
.stepbar { background: #10141f; border-top: 1px solid #232c40; color: #d7deea; }
.steps { display: flex; }
.step { flex: 1; display: flex; gap: 8px; padding: 8px 12px; cursor: pointer;
  border-right: 1px solid #1a2133; align-items: center; }
.step:hover { background: #161c2b; }
.step.open { background: #1a2133; }
.step.bad .tt { color: #f87171; }
.idx { width: 22px; height: 22px; border-radius: 50%; background: #2f6fed; color: #fff;
  font-size: 12px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.tt { font-size: 13px; font-weight: 600; }
.ss { font-size: 11px; color: #8b95ab; margin-top: 1px; }
.detail { border-top: 1px solid #1a2133; padding: 10px 14px; max-height: 220px; overflow-y: auto; }
.bar-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.bn { width: 70px; font-size: 12px; color: #cbd5e1; }
.bar { flex: 1; height: 14px; background: #1d2331; border-radius: 7px; overflow: hidden; }
.fill { height: 100%; border-radius: 7px; }
.bv { width: 52px; font-size: 12px; }
.tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
.tbl th, .tbl td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #1a2133; }
.tbl th { color: #8b95ab; }
.okline { color: #4ade80; font-size: 13px; }
.blk { color: #f87171; font-size: 12px; margin: 3px 0; }
.json { font-size: 11px; color: #9fb3d1; white-space: pre-wrap; }
</style>
