<script setup>
// 指标抽屉：四项指标 + 消融占位（被问"怎么证明"时拉出）
defineProps(['metrics'])
const emit = defineEmits(['close'])
const fmt = v => (typeof v === 'number' ? (v * 100).toFixed(1) + '%' : (v ?? '—'))
</script>

<template>
  <div class="mask" @click.self="emit('close')">
    <div class="drawer">
      <div class="head">📊 指标中心 <button @click="emit('close')">✕</button></div>
      <div v-if="metrics">
        <div class="cards">
          <div class="m"><div class="k">对齐覆盖率</div><div class="v">{{ metrics['对齐覆盖率'] }}%</div>
            <div class="d">停留事件→POI 语义节点（五环内，剔除等待类）</div></div>
          <div class="m"><div class="k">感知准确率</div>
            <div class="v">疲劳 {{ fmt(metrics['感知准确率']?.fatigue?.heldout_video_acc) }}</div>
            <div class="v2">MLP top3 {{ fmt(metrics['感知准确率']?.mlp?.val_top3) }}</div>
            <div class="d">疲劳=DDD留来源外部验证 11/12；MLP=按用户划分 val</div></div>
          <div class="m"><div class="k">端到端耗时</div>
            <div class="v">{{ metrics['端到端耗时']?.last_ms }} ms</div>
            <div class="d">输入→决策输出全链路（含疲劳推理/LLM 通道）</div></div>
          <div class="m"><div class="k">高德属性对齐</div>
            <div class="v">{{ metrics['高德属性']?.matched_total }} 条</div>
            <div class="d">同名/子串 + 同类兜底，样例人工核验</div></div>
          <div class="m"><div class="k">回放评测</div>
            <div class="v">{{ fmt(metrics['回放评测']?.val?.rate) }}</div>
            <div class="v2">判卷 {{ metrics['回放评测']?.val?.graded }} 次 / 命中 {{ metrics['回放评测']?.val?.hit }} 次</div>
            <div class="d">验证集用户真实轨迹逐点判卷：每 20 模拟分钟一决策，3 小时内下一真实停留是否在 MLP top3 中</div></div>
        </div>
        <h4>消融实验（12 月，占位）</h4>
        <table class="ab">
          <tr><th>变体</th><th>预期对比</th></tr>
          <tr><td>无约束引擎</td><td>决策合规率 ↓</td></tr>
          <tr><td>仅 MLP / 仅 LLM / 双通道</td><td>端到端效果对比</td></tr>
          <tr><td>有无环境上下文</td><td>推荐合理性</td></tr>
        </table>
      </div>
      <div v-else class="dim">加载中…</div>
    </div>
  </div>
</template>

<style scoped>
.mask { position: fixed; inset: 0; background: rgba(8, 10, 16, .55); z-index: 2000;
  display: flex; justify-content: flex-end; }
.drawer { width: 420px; background: #10141f; border-left: 1px solid #232c40;
  padding: 18px; color: #d7deea; overflow-y: auto; }
.head { display: flex; justify-content: space-between; font-weight: 700; font-size: 15px; }
.head button { background: none; border: none; color: #8b95ab; font-size: 16px; cursor: pointer; }
.cards { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
.m { background: #161c2b; border-radius: 10px; padding: 12px; }
.k { font-size: 12px; color: #8b95ab; }
.v { font-size: 22px; font-weight: 700; margin: 4px 0; color: #60a5fa; }
.v2 { font-size: 14px; font-weight: 600; color: #4ade80; }
.d { font-size: 11px; color: #55607a; line-height: 1.4; margin-top: 4px; }
h4 { font-size: 13px; color: #8b95ab; margin: 18px 0 8px; }
.ab { width: 100%; border-collapse: collapse; font-size: 12px; }
.ab th, .ab td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #1a2133; }
.dim { color: #55607a; margin-top: 16px; }
</style>
