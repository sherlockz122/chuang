<script setup>
import { computed } from 'vue'
import { CAT_NAMES } from '../api'
const props = defineProps(['prof', 'fatigue', 'fatigueName', 'presets', 'busy',
  'time', 'highway', 'useLlm', 'safety'])
const emit = defineEmits(['infer', 'time', 'highway', 'llm'])

const durTotal = computed(() => {
  const d = (props.prof?.seq || []).filter(s => s.node_type === 'poi')
  return d.length ? Math.round(d.reduce((a, s) => a + (s.dur_min || 0), 0) / 60) : 0
})
</script>

<template>
  <aside class="left">
    <section>
      <h3>用户画像</h3>
      <div v-if="prof">
        <div class="row"><span>停留事件</span><b>{{ prof.n_stays }}</b></div>
        <div class="row"><span>活跃天数</span><b>{{ (prof.days || []).length }}</b></div>
        <div class="row"><span>总停留时长</span><b>{{ durTotal }} 小时</b></div>
        <div class="row"><span>时间范围</span><b>{{ (prof.days || [])[0] }} ~ {{ (prof.days || [])[prof.days.length - 1] }}</b></div>
      </div>
      <div v-else class="dim">未加载</div>
    </section>

    <section>
      <h3>疲劳感知</h3>
      <div class="row">
        <span>场景</span>
        <select @change="emit('infer', $event.target.value)">
          <option value="">选择预置帧…</option>
          <option v-for="(n, name) in presets" :key="name" :value="name">
            {{ { drowsy: '困倦', alert: '清醒' }[name] || name }}（{{ n }}帧）
          </option>
        </select>
      </div>
      <div v-if="fatigue" class="sig">
        <div class="row"><span>当前场景</span><b>{{ fatigueName }}</b></div>
        <div class="row"><span>PERCLOS</span><b>{{ fatigue.perclos ?? '-' }}</b></div>
        <div class="row"><span>教师EAR对照</span><b>{{ fatigue.ear_perclos ?? '-' }}</b></div>
        <div class="row"><span>安全等级</span><b :class="'lv' + (fatigue.safety_level ?? 0)">
          L{{ fatigue.safety_level ?? 0 }}</b></div>
        <div class="row"><span>推理耗时</span><b>{{ fatigue.elapsed_ms ? (fatigue.elapsed_ms / 1000).toFixed(1) + 's' : '-' }}</b></div>
      </div>
      <div v-else class="dim">未运行疲劳推理（默认按清醒推流）</div>
    </section>

    <section>
      <h3>环境设置</h3>
      <div class="row"><span>道路等级</span>
        <select :value="highway" @change="emit('highway', $event.target.value)">
          <option value="secondary">普通道路</option>
          <option value="motorway">高速</option>
        </select>
      </div>
      <div class="row"><span>LLM 定稿通道</span>
        <label class="switch"><input type="checkbox" :checked="useLlm" @change="emit('llm', $event.target.checked)" />
          <span>{{ useLlm ? '开' : '关' }}</span></label>
      </div>
    </section>
  </aside>
</template>

<style scoped>
.left { width: 240px; flex-shrink: 0; background: #141926; border-right: 1px solid #232c40;
  padding: 12px; overflow-y: auto; color: #d7deea; }
section { margin-bottom: 18px; }
h3 { font-size: 12px; color: #8b95ab; text-transform: uppercase; letter-spacing: 1px;
  margin: 0 0 8px; }
.row { display: flex; justify-content: space-between; align-items: center;
  font-size: 13px; padding: 4px 0; gap: 8px; }
.row span { color: #8b95ab; }
.row b { color: #e8ecf5; font-weight: 600; }
.dim { color: #55607a; font-size: 12px; }
.sig { background: #1a2133; border-radius: 8px; padding: 6px 10px; margin-top: 6px; }
select { background: #1d2331; color: #e8ecf5; border: 1px solid #333b4f;
  border-radius: 6px; padding: 3px 6px; font-size: 12px; max-width: 140px; }
.lv0 { color: #4ade80; } .lv1 { color: #facc15; } .lv2 { color: #f87171; }
.switch { display: flex; align-items: center; gap: 4px; font-size: 12px; }
</style>
