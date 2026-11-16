# POI 语义图 Schema v1

> 版本：v1（2026-11-02）｜ 状态：设计定稿，随开发迭代修订
> 坐标系统一为 WGS-84（EPSG:4326）。高德数据 GCJ-02 → WGS-84 转换发生在清洗阶段，之后全链路只用 WGS-84。

## 1. 节点 ID 规则

| 节点类型 | 主键格式 | 说明 |
|---|---|---|
| POI（OSM） | `OSM_N_{osm_id}` | OSM node 的全局唯一 id，稳定不变 |
| POI（高德） | `AMAP_{amap_id}` | 高德 POI id，仅作属性补充源；与 OSM 同名同点（<50m）时合并到 OSM 主键 |
| 道路 | `OSM_W_{way_id}` | OSM way 的 id，道路以**实体视角**作节点（线图变换，与交叉口图等价可互转） |
| 区域 | `REG_{adcode}` | 行政区 adcode 或商圈自定义 id（商圈用 `REG_BIZ_{序号}`） |

## 2. nodes.json 字段

```json
{
  "node_id": "OSM_N_244071099",
  "type": "poi | road | region",
  "name": "中石化加油站(西直门)",
  "lng": 116.352, "lat": 39.945,
  "category": "fuel",
  "group": "instant",
  "source": "osm | amap | both",
  "amap_code": "010100",
  "road_class": "primary",        // 仅 type=road
  "adcode": "110108",              // 仅 type=region
  "osm_tags": {"amenity": "fuel", "name:zh": "..."},
  "props": {}
}
```

## 3. edges.json 字段

| edge_type | 含义 | 权重/属性 |
|---|---|---|
| `SPATIAL_NEAR` | poi–road ≤ 50m；poi–poi r=500m 且 k≤20 | `distance_m` |
| `IN_REGION` | poi–region 点在多边形内 | — |
| `ROAD_CONN` | road–road 共享交叉口（连通性） | `shared_nodes` |
| `FUNC_COMP` | 功能互补先验表 + 后期轨迹共现加权 | `strength` 0~1 |
| `TRAJ_COOCCUR` | 轨迹共现（阶段二接入） | `count`, `weight` |

```json
{
  "edge_id": "e_0001",
  "src": "OSM_N_244071099",
  "dst": "OSM_W_238742",
  "etype": "SPATIAL_NEAR",
  "props": {"distance_m": 23.4}
}
```

## 4. 产出物清单（阶段一）

| 文件 | 位置 | 内容 |
|---|---|---|
| `nodes.json` / `edges.json` | `new_route/data/graph_out/` | 构图产出 |
| `graph.pkl` | 同上 | NetworkX MultiGraph 对象 |
| `stats_report.md` | `new_route/docs/` | 质检统计报告 |
| `graph_map.html` | `new_route/docs/` | folium 分层着色图（答辩素材） |

## 5. 设计口径备忘

- 道路=节点是**实体视角**设计（线图变换）：与"交叉口=节点"经典路网图等价可互转，连通性由 ROAD_CONN 边承载；地图匹配时抽出道路子图即还原路网。
- 三类实体做节点，语义关系做边；高德只作属性补充，主键永远 OSM（osmid 全局稳定）。
- 区域节点承载"商圈/行政区"粒度的情境，是环境约束（区域级车流/安全等级）挂接点。
