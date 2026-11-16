<script setup>
// 推送卡/拦截卡：像真实车载通知从地图右侧滑出
import { computed } from 'vue'
const props = defineProps(['result', 'safety', 'catNames', 'blockStreak'])

const decision = computed(() => props.result?.decision)
const suppressed = computed(() => !!props.result?._suppressed)  // 同 POI 冷却
const blocked = computed(() => suppressed.value || !decision.value || !decision.value.poi_id)
// 推送目标的名字兜底：候选名 → 决策自带名 → 类别中文名 → 泛称
const pushName = computed(() => {
  const d = decision.value
  if (!d) return ''
  const cand = (props.result?.candidates || []).find(c => c.poi_id === d.poi_id)
  return cand?.name || d.name || props.catNames[d.category] || '附近设施'
})
const pushCat = computed(() => {
  const d = decision.value
  const cand = (props.result?.candidates || []).find(c => c.poi_id === d.poi_id)
  return cand?.category || d.category
})
const pushDist = computed(() => {
  const d = decision.value
  const cand = (props.result?.candidates || []).find(c => c.poi_id === d.poi_id)
  return cand?.dist_m ?? d.dist_m ?? null
})
// MLP 说数字（预测依据）+ 约束引擎结论，直接亮在卡上
const predText = computed(() =>
  (props.result?.mlp_top3_cats || []).slice(0, 3)
    .map(([c, p]) => `${props.catNames[c] || c} ${Math.round(p * 100)}%`).join(' · '))
const consText = computed(() => {
  const n = (props.result?.constraint_blocked || []).length
  return n ? `拦截 ${n} 个候选（详情见底部步骤条③）` : '全部通过'
})
// 拦截原因按规则归并（同一原因拦截多个候选只显示一条 + 数量）
const reasons = computed(() => {
  if (!props.result) return []
  if (suppressed.value) return ['该 POI 刚推送过（45 分钟内冷却），避免重复打扰']
  const cnt = new Map()
  for (const b of (props.result.constraint_blocked || [])) {
    cnt.set(b.rule, (cnt.get(b.rule) || 0) + 1)
  }
  const out = [...cnt.entries()].map(([rule, n]) =>
    n > 1 ? `${rule}（${n} 个候选同因拦截）` : rule)
  if (props.safety >= 2 && !out.length) out.push('疲劳等级 L2：暂停一切推送（安全门控）')
  return out
})
</script>

<template>
  <Transition name="slide">
    <div v-if="result" class="card" :class="blocked ? 'blocked' : 'push'">
      <template v-if="!blocked">
        <div class="tag">📢 推送</div>
        <div class="title">{{ pushName }}</div>
        <div class="meta">{{ catNames[pushCat] || '' }} · 距离约 {{ pushDist ?? '?' }}m</div>
        <div class="reason">{{ decision.reason }}</div>
        <div class="pred">📊 MLP 预测: {{ predText }}</div>
        <div class="cons">⚖️ 约束引擎: {{ consText }}</div>
        <div v-if="result._verify" class="verify">
          ✓ 事后验证：司机 {{ result._verify.afterMin }} 分钟后到达
          「{{ result._verify.stay.name || catNames[result._verify.stay.category] || '同类 POI' }}」——推送类别命中
        </div>
      </template>
      <template v-else>
        <div class="tag">{{ suppressed ? '⏳ 冷却' : '🛡 拦截' }}</div>
        <div class="title">{{ suppressed ? '暂不重复推送' : '本次不推送' }}</div>
        <div v-if="!suppressed && predText" class="pred">📊 模型本来想推: {{ predText }}</div>
        <div class="reason" v-for="r in reasons" :key="r">· {{ r }}</div>
        <div v-if="blockStreak >= 2" class="reason">· 已连续拦截 {{ blockStreak }} 次（安全门控生效中）</div>
      </template>
      <div v-if="result._grade" class="grade" :class="result._grade.hit ? 'hit' : 'miss'">
        {{ result._grade.hit ? '✓ 判卷命中' : '✗ 判卷未命中' }}：司机 {{ result._grade.afterMin }} 分钟后到达
        「{{ result._grade.stay.name || catNames[result._grade.stay.category] || '同类 POI' }}」——预测类别{{ result._grade.hit ? '在 top3 内' : '不在 top3 内' }}
      </div>
      <div class="foot">安全等级 L{{ safety }} · 决策耗时 {{ result.elapsed_ms ?? '-' }}ms</div>
    </div>
  </Transition>
</template>

<style scoped>
.card { position: absolute; top: 14px; right: 14px; width: 260px; z-index: 1000;
  border-radius: 12px; padding: 14px; box-shadow: 0 6px 24px rgba(0,0,0,.4);
  backdrop-filter: blur(6px); }
.push { background: rgba(16, 42, 24, .92); border: 1px solid #2f6f4f; color: #e6f7ec; }
.blocked { background: rgba(64, 18, 20, .92); border: 1px solid #8f3a3a; color: #fdeaea; }
.tag { font-size: 12px; font-weight: 700; letter-spacing: 1px; opacity: .85; }
.title { font-size: 17px; font-weight: 700; margin: 6px 0 2px; }
.meta { font-size: 12px; opacity: .8; }
.reason { font-size: 12px; margin-top: 8px; opacity: .92; line-height: 1.5; }
.pred { font-size: 11px; margin-top: 8px; padding: 5px 8px; border-radius: 6px;
  background: rgba(255,255,255,.08); line-height: 1.5; }
.cons { font-size: 11px; margin-top: 5px; opacity: .75; }
.grade { font-size: 12px; margin-top: 8px; padding: 5px 8px; border-radius: 6px; line-height: 1.5;
  border: 1px dashed rgba(255,255,255,.3); }
.grade.hit { color: #4ade80; background: rgba(74,222,128,.08); }
.grade.miss { color: #fca5a5; background: rgba(252,165,165,.08); }
.verify { font-size: 12px; margin-top: 8px; line-height: 1.5;
  color: #4ade80; border-top: 1px dashed rgba(255,255,255,.25); padding-top: 6px; }
.foot { font-size: 11px; margin-top: 10px; opacity: .6; border-top: 1px dashed rgba(255,255,255,.25); padding-top: 6px; }
.slide-enter-active, .slide-leave-active { transition: all .35s ease; }
.slide-enter-from, .slide-leave-to { transform: translateX(60px); opacity: 0; }
</style>
