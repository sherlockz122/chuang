# 金标准情景题库判卷报告
- 口径: 情景题库: 端到端决策对金标准判卷(约束四规则边界+组合+推送合理性+冷启动), 非回放; 频次基线=同卷对照(历史最常去top3, 无约束)
- 判卷: **13/13 通过（100.0%）** | 频次基线同卷: 8/13（61.5%）
- 决策耗时: 平均 1923.5ms / 最大 6036.0ms

## 分规则通过率

| 规则 | 通过 |
|---|---|
| safety | 4/4 |
| night | 2/2 |
| highway | 2/2 |
| cooldown | 1/1 |
| combo | 1/1 |
| push | 2/2 |
| cold | 1/1 |

## 逐题判卷

### S01 L1低警惕拦景区 ✅

- 输入: U_15 @ 2008-02-08 12:00:00 · safety=1 · secondary · 距上次推送0.0min
- MLP top3: park/restaurant/parking · 决策: 停车场(parking) · 6036.0ms
  - ✓ 拦截 荷花市场(scenic): 安全等级1不允许scenic
  - ✓ 结构完整(候选内/有名字/有理由)
- 频次基线对照: 频次基线推 北京市什刹海运动学校(该类无约束冲突)

### S02 L2疲劳拦餐饮 ✅

- 输入: U_3 @ 2008-02-08 12:00:00 · safety=2 · secondary · 距上次推送0.0min
- MLP top3: restaurant/hotel/transport · 决策: 空(全拦) · 6.7ms
  - ✓ 拦截 Under Renovation (used to be Bookworm)(restaurant): 安全等级2不允许restaurant
  - ✓ 拦截 Panorama(restaurant): 安全等级2不允许restaurant
  - ✓ 拦截 Iqiyi(restaurant): 安全等级2不允许restaurant
  - ✓ 拦截 Hong Kong Lion Hotpot(restaurant): 安全等级2不允许restaurant
  - ✓ 拦截 兰州拉面(restaurant): 安全等级2不允许restaurant
  - ✓ 拦截 餐饮(restaurant): 安全等级2不允许restaurant
  - ✓ 拦截 Duplex Noodle Place(restaurant): 安全等级2不允许restaurant
  - ✓ 拦截 D-Lounge(restaurant): 安全等级2不允许restaurant
- 频次基线对照: 频次基线会推被拦类别 Under Renovation (used to be Bookworm)(restaurant) ✗

### S03 L2疲劳只放停车类 ✅

- 输入: U_3 @ 2008-02-08 12:00:00 · safety=2 · secondary · 距上次推送0.0min
- MLP top3: restaurant/hotel/transport · 决策: 清上园停车场(parking) · 2504.2ms
  - ✓ 推送 parking: 实际 清上园停车场
  - ✓ 结构完整(候选内/有名字/有理由)
- 频次基线对照: 频次基线推 好利来

### S04 L0清醒白天放景区 ✅

- 输入: U_15 @ 2008-02-08 12:00:00 · safety=0 · secondary · 距上次推送0.0min
- MLP top3: park/restaurant/parking · 决策: 荷花市场(scenic) · 2391.6ms
  - ✓ 不拦 荷花市场(scenic): 未拦截
  - ✓ 结构完整(候选内/有名字/有理由)
- 频次基线对照: 频次基线推 北京市什刹海运动学校(该类无约束冲突)

### S05 深夜23点拦景区 ✅

- 输入: U_24 @ 2008-02-08 23:00:00 · safety=0 · secondary · 距上次推送0.0min
- MLP top3: restaurant/hotel/parking · 决策: 停车场(parking) · 2300.9ms
  - ✓ 拦截 荷花市场(scenic): 深夜时段(23:00)该类已关闭
  - ✓ 结构完整(候选内/有名字/有理由)
- 频次基线对照: 频次基线推 星巴克(该类无约束冲突)

### S06 深夜放行餐饮(对照组) ✅

- 输入: U_24 @ 2008-02-08 23:00:00 · safety=0 · secondary · 距上次推送0.0min
- MLP top3: restaurant/hotel/parking · 决策: 餐饮(restaurant) · 1674.9ms
  - ✓ 推送 top3 之一(restaurant/hotel/parking): 实际 restaurant
  - ✓ 结构完整(候选内/有名字/有理由)
- 频次基线对照: 频次基线推 Under Renovation (used to be Bookworm)(该类无约束冲突)

### S07 高速拦目的地型 ✅

- 输入: U_3 @ 2008-02-08 12:00:00 · safety=0 · motorway · 距上次推送0.0min
- MLP top3: restaurant/hotel/transport · 决策: Zhongluanxin Insurance Agency Co. Ltd(office) · 2909.4ms
  - ✓ 拦截 Under Renovation (used to be Bookworm)(restaurant): 高速路段不推目的地型(restaurant)
  - ✓ 拦截 Panorama(restaurant): 高速路段不推目的地型(restaurant)
  - ✓ 拦截 Iqiyi(restaurant): 高速路段不推目的地型(restaurant)
  - ✓ 拦截 Hong Kong Lion Hotpot(restaurant): 高速路段不推目的地型(restaurant)
  - ✓ 拦截 兰州拉面(restaurant): 高速路段不推目的地型(restaurant)
  - ✓ 拦截 餐饮(restaurant): 高速路段不推目的地型(restaurant)
  - ✓ 拦截 Duplex Noodle Place(restaurant): 高速路段不推目的地型(restaurant)
  - ✓ 拦截 D-Lounge(restaurant): 高速路段不推目的地型(restaurant)
  - ✓ 结构完整(候选内/有名字/有理由)
- 频次基线对照: 频次基线会推被拦类别 Under Renovation (used to be Bookworm)(restaurant) ✗

### S08 高速放行加油站(对照组) ✅

- 输入: U_318 @ 2008-02-08 12:00:00 · safety=0 · motorway · 距上次推送0.0min
- MLP top3: fuel/restaurant/office · 决策: 中国石化(fuel) · 1919.2ms
  - ✓ 推送 fuel: 实际 中国石化
  - ✓ 结构完整(候选内/有名字/有理由)
- 频次基线对照: 频次基线推 中国石化

### S09 冷却期内全拦 ✅

- 输入: U_3 @ 2008-02-08 12:00:00 · safety=0 · secondary · 距上次推送7.0min
- MLP top3: restaurant/hotel/transport · 决策: 空(全拦) · 6.1ms
  - ✓ 决策为空(全拦): 实际 空
- 频次基线对照: 频次基线会推 Under Renovation (used to be Bookworm)(无视冷却/安全) ✗

### S10 疲劳+高速+冷却三重叠加 ✅

- 输入: U_3 @ 2008-02-08 12:00:00 · safety=2 · motorway · 距上次推送7.0min
- MLP top3: restaurant/hotel/transport · 决策: 空(全拦) · 5.8ms
  - ✓ 决策为空(全拦): 实际 空
- 频次基线对照: 频次基线会推 Under Renovation (used to be Bookworm)(无视冷却/安全) ✗

### S11 午餐商圈推餐饮 ✅

- 输入: U_3 @ 2008-02-08 12:00:00 · safety=0 · secondary · 距上次推送0.0min
- MLP top3: restaurant/hotel/transport · 决策: 餐饮(restaurant) · 1409.7ms
  - ✓ 推送 top3 之一(restaurant/hotel/transport): 实际 restaurant
  - ✓ 结构完整(候选内/有名字/有理由)
- 频次基线对照: 频次基线推 Under Renovation (used to be Bookworm)(该类无约束冲突)

### S12 途中推加油站 ✅

- 输入: U_318 @ 2008-02-08 12:00:00 · safety=0 · secondary · 距上次推送0.0min
- MLP top3: fuel/restaurant/office · 决策: 中国石化(fuel) · 2445.7ms
  - ✓ 推送 fuel: 实际 中国石化
  - ✓ 结构完整(候选内/有名字/有理由)
- 频次基线对照: 频次基线推 中国石化

### S13 冷启动用户稳健决策 ✅

- 输入: U_191 @ 2008-02-06 12:00:00 · safety=0 · secondary · 距上次推送0.0min
- MLP top3: restaurant/hotel/transport · 决策: 餐饮(restaurant) · 1395.2ms
  - ✓ 推送 top3 之一(restaurant/hotel/transport): 实际 restaurant
  - ✓ 结构完整(候选内/有名字/有理由)
  - ✓ 概率不塌缩(max=0.135≤0.35)
- 频次基线对照: 频次基线推 无(该类无约束冲突)

## 模块级验证矩阵

| 模块 | 指标 | 数字 |
|---|---|---|
| 兴趣建模 MLP | val top1 / top3 | 0.2947 / 0.547 |
| 回放评测 | 全体 / 验证集 top3 命中 | 0.5933 / 0.5077 |
| 对照基线 | 随机 / 频次(验证集) | 0.1585 / 0.4626 |
| 分片对比 | 冷启动 MLP-频次 | +19.5 点 |
| 疲劳识别 | DDD 留来源 / val MAE / 参数 | 91.7% / 0.0268 / 93825 |
| 数据对齐 | 对齐覆盖率 / 高德属性 | 86.27% / 1826 条 |
