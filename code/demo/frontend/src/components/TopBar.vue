<script setup>
import { computed } from 'vue'
const props = defineProps(['users', 'state', 'result', 'busy'])
const emit = defineEmits(['select-user', 'decide', 'metrics'])

const safetyText = computed(() => ({ 0: '清醒', 1: '低警惕', 2: '疲劳' })[props.state.safety] ?? '-')
const safetyCls = computed(() => ({ 0: 'ok', 1: 'warn', 2: 'bad' })[props.state.safety] ?? '')
</script>

<template>
  <header class="topbar">
    <div class="brand">🚗 车载 POI 动态推送系统</div>
    <div class="ctl">
      <label>用户
        <select :value="state.uid" @change="emit('select-user', $event.target.value)">
          <option v-for="u in users" :key="u.user_id" :value="u.user_id">
            {{ u.user_id }}（停留{{ u.n_stays }}次）
          </option>
        </select>
      </label>
      <label>时间 <input v-model="state.time" style="width:150px" /></label>
      <span class="chip" :class="safetyCls">疲劳等级 {{ state.safety }} · {{ safetyText }}</span>
      <button class="primary" :disabled="busy" @click="emit('decide')">⚡ 一键决策</button>
      <button class="ghost" @click="emit('metrics')">指标</button>
    </div>
  </header>
</template>

<style scoped>
.topbar { display: flex; align-items: center; justify-content: space-between;
  padding: 10px 16px; background: #10141f; color: #e8ecf5; flex-wrap: wrap; gap: 8px; }
.brand { font-weight: 700; font-size: 15px; }
.ctl { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.ctl label { font-size: 13px; display: flex; align-items: center; gap: 6px; }
select, input { background: #1d2331; color: #e8ecf5; border: 1px solid #333b4f;
  border-radius: 6px; padding: 4px 8px; font-size: 13px; }
.chip { font-size: 12px; padding: 3px 10px; border-radius: 12px; font-weight: 600; }
.chip.ok { background: #123b22; color: #4ade80; }
.chip.warn { background: #3d3512; color: #facc15; }
.chip.bad { background: #4a1416; color: #f87171; }
button { border: none; border-radius: 8px; padding: 7px 14px; font-size: 13px; cursor: pointer; }
button.primary { background: #2f6fed; color: #fff; font-weight: 600; }
button.primary:disabled { opacity: .5; cursor: wait; }
button.ghost { background: #1d2331; color: #cbd5e1; border: 1px solid #333b4f; }
</style>
