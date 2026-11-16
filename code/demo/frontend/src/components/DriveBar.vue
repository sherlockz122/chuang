<script setup>
// 行驶回放控制条：日期/播放/停止/速度/跟随视角 + 实时状态（覆盖在地图舞台顶部）
const props = defineProps(['drive', 'busy'])
const emit = defineEmits(['play', 'stop', 'day', 'speed', 'follow'])
</script>

<template>
  <div class="drivebar">
    <label>日期
      <select :value="drive.day" :disabled="drive.playing || busy" @change="e => emit('day', e.target.value)">
        <option v-for="d in drive.days" :key="d" :value="d">{{ d }}</option>
      </select>
    </label>
    <button class="btn play" :disabled="busy" @click="emit('play')">
      {{ drive.playing ? '⏸ 暂停' : (drive.ended ? '▶ 重播' : '▶ 播放') }}
    </button>
    <button class="btn stop" :disabled="!drive.playing" @click="emit('stop')">■</button>
    <label>速度
      <select :value="drive.speed" @change="e => emit('speed', +e.target.value)">
        <option :value="1">1 分钟/秒</option>
        <option :value="1.5">1.5 分钟/秒</option>
        <option :value="2">2 分钟/秒</option>
        <option :value="5">5 分钟/秒</option>
        <option :value="10">10 分钟/秒</option>
      </select>
    </label>
    <label class="chk">
      <input type="checkbox" :checked="drive.follow" @change="e => emit('follow', e.target.checked)" />
      跟随视角
    </label>
    <div class="info">
      <span v-if="drive.points.length" class="t">{{ drive.points[drive.idx]?.t }}</span>
      <span v-if="drive.points.length">第 {{ drive.idx + 1 }}/{{ drive.points.length }} 点</span>
      <span class="cnt">推 <b>{{ drive.nPush }}</b> · 拦 <b>{{ drive.nBlock }}</b> · 静 <b>{{ drive.nSilent }}</b>
        · 判卷 <b>{{ drive.nGrad }}</b>/命中 <b>{{ drive.nHit }}</b></span>
      <span v-if="drive.busyDecide" class="deciding">⚡ 决策中…</span>
      <span v-else-if="drive.ended" class="deciding">✓ 已播完（可点重播）</span>
    </div>
  </div>
</template>

<style scoped>
.drivebar {
  position: absolute; top: 10px; left: 10px; z-index: 1100;
  display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  background: rgba(17, 24, 39, .88); color: #e5e7eb;
  padding: 6px 10px; border-radius: 8px; font-size: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, .4);
}
.drivebar select {
  background: #1f2937; color: #e5e7eb; border: 1px solid #374151;
  border-radius: 4px; padding: 2px 4px; font-size: 12px;
}
.drivebar select:disabled { opacity: .5; }
.btn {
  border: none; border-radius: 4px; padding: 3px 10px; cursor: pointer; font-size: 12px;
}
.btn:disabled { cursor: not-allowed; opacity: .5; }
.btn.play { background: #2563eb; color: #fff; }
.btn.stop { background: #374151; color: #e5e7eb; }
.chk { display: flex; gap: 4px; align-items: center; }
.info { display: flex; gap: 12px; align-items: center; }
.t { color: #93c5fd; }
.cnt b { color: #fbbf24; }
.deciding { color: #fde047; animation: pulse 1s infinite; }
@keyframes pulse { 50% { opacity: .4; } }
</style>
