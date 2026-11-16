# new_route —— 车载兴趣点动态推送（新路线）

数据驱动开发路线：**POI语义图 → 强制ID对齐 → 端侧小模型（MLP+JSON双通道）→ 环境约束推送 → Vue demo**

## 目录

```
config/   城市/类别/Schema 配置（city.yaml, categories.yaml, schema_v1.md）
graph/    环节一：POI 语义图构建（取数→清洗→构图→质检）
align/    环节二：多源数据强制 ID 对齐（地图匹配 + 时间戳对齐）
model/    环节三+四：端侧小模型 + MLP 编码 + JSON prompt + 环境约束
demo/     环节五：Vue 界面 demo
```

## 数据目录（不进 git，位于 ../data/）

```
osm_beijing/  BBBike 北京提取包（.osm.pbf + .shp.zip）
amap_poi/     高德多边形搜索抓取结果（GCJ-02 原始 → 转换后）
tdrive/       T-Drive 北京出租车轨迹
fatigue/      疲劳驾驶状态数据集（NTHU-DDD 等）
aligned/      对齐产出（统一 ID + 时空对齐 + 每用户语义记录）
graph_out/    语义图产出（nodes/edges JSON + graph.pkl）
```

## 进度

阶段一（POI语义图）进行中。开发计划见桌面 `开发计划清单-新路线.md`。
