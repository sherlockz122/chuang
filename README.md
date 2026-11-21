# 车载兴趣点动态推送系统（演示程序 + 工程文档 + 验证工具）

基于轻量化多模态融合感知的车载兴趣点动态推送机制研究——完整演示程序：
在对的时刻、以安全的方式把对的 POI 推给车里的人。
唯一端侧模型双通道（MLP 说数字 + JSON prompt 说句子）+ 约束引擎四规则安全门控。

## 目录结构

| 路径 | 内容 |
|---|---|
| `code/` | 全部工程代码（demo 后端 FastAPI + 前端 Vue、model 模型与评测脚本、graph 语义图构建、config） |
| `code/docs/` | 工程文档：`演示程序操作说明.md`（产品使用+回放评测口径）、`答辩演示彩排清单.md` 等 |
| `data/` | 运行时数据：T-Drive 300 司机子集、POI 语义图（49,136 节点/232,857 边）、完整车行路网（51.8 万段）、对齐画像、模型权重（mlp.pt/eye_rgb.pt）、各评测报告 |
| `models/` | LLM 权重目录（可选，见下） |
| `setup_llm.py` | 一键下载 Qwen2.5-0.5B-Instruct 权重 |

## 环境要求

- **Python 环境**（demo 后端/模型/评测）：Python 3.11+，依赖 torch / fastapi / uvicorn / scipy / transformers / numpy / scikit-learn / yaml。可直接 `pip install torch fastapi uvicorn scipy transformers numpy scikit-learn pyyaml`
- **疲劳感知**（可选）：需 `D:/mp-env` 独立环境（mediapipe 1.2.0 + opencv + torch CPU，Python 3.11）——仅"疲劳预置推理"功能用到，不影响回放主流程
- **前端**：已带构建产物 `code/demo/frontend/dist`，**无需 Node**；改前端代码才需要 Node 26 + npm 11

## 快速启动

```bash
cd code
python demo/server.py --port 8000
# 启动约 40 秒（建索引 + 预热模型），然后浏览器打开 http://127.0.0.1:8000
```

详细操作（界面六区、回放用法、疲劳演示、判卷口径）见 `code/docs/演示程序操作说明.md`。

## LLM 通道（可选）

Qwen2.5-0.5B-Instruct 权重约 1GB，因 GitHub 单文件限制不入库：

```bash
python setup_llm.py        # 默认 ModelScope 源（国内快）
python setup_llm.py hf     # 或 HuggingFace 源
```

不下载也能完整运行：前端左侧"LLM 定稿通道"关掉，决策走 MLP 兜底（合规候选取最近）。

## 验证工具（非回放验证）

```bash
cd code/model
python scenario_bank.py          # 金标准情景题库: 13 题端到端判卷 + 频次基线对照
python replay_eval.py            # 离线回放评测: 300 用户逐点判卷
python baseline_eval.py          # 随机/频次对照基线
python compare_eval.py           # MLP vs 频次基线 分时段/分活跃度对比
python lead_time.py              # 预测提前量分布
```

## 关键指标（当前版本，2026-11-21 审计后口径）

| 指标 | 数值 |
|---|---|
| MLP 预测 val top3 | **52.52%**（随机 15.8%；2026-11-21 修复训练标签泄漏后的诚实值） |
| 回放评测（离线 300 用户） | 全体 59.63% / 验证集 49.74% |
| 情景题库 | 13/13 通过（频次基线同卷 8/13） |
| 疲劳识别（DDD 留来源） | 91.7%（val MAE 0.0268） |
| 数据对齐覆盖率 | 86.27%（高德属性 1,826 条） |
| 端到端决策耗时 | 2~3s（模型进程级缓存+预热） |
| 预测提前量 | 中位 76 分钟（≥15min 占 91.3%） |

## 注意事项

- `.amap_key.env`（高德 API key）不入库；涉及高德接口的脚本需自行配置 key
- 地图瓦片为高德栅格（GCJ-02），前端已自动做坐标转换
- 轨迹贴路 A* 已跑通全量路网；默认演示天 U_318/2008-02-04
