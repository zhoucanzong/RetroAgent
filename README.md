# RetroAgent

基于 **LLM 中枢决策 + 专用化学工具** 的逆合成路线规划系统。

## 设计理念

> 智能只存在于 Planner（LLM）。专用化学模型（ONNX、模板库、库存）退化为 Tool —— 纯函数，不做决策。

```
Planner (LLM)              ← 唯一决策中枢
  ├─ 理解化学结构
  ├─ 交叉验证 Tool 输出
  ├─ 选择搜索策略
  └─ 判断路线完成

Tools (6个)                ← 纯数据提供者
  ├─ disconnect             ONNX 模板预测 + 官能团检测
  ├─ propose                RDChiral 模板应用 + fallback 扫描
  ├─ evaluate               可行性评分 + 库存检查
  ├─ check_stock            ZINC 17.4M 分子查询
  ├─ search_literature      模板分类检索
  ├─ recommend_conditions   反应条件推荐
  └─ bash                   RDKit 计算 / 文件操作

Shared Blackboard           ← 纯状态容器（无决策逻辑）
```

## 架构参考

- **mini-swe-agent**: Agent 控制循环、Environment 协议、异常体系
- **AiZynthFinder**: ONNX 策略网络、USPTO 模板库、RDChiral 模板应用、ZINC 库存

## 项目结构

```
retroagent/
├── __init__.py              # 版本、路径配置
├── blackboard.py            # SharedBlackboard 状态容器
├── agents/
│   ├── planner.py           # RetroPlanner (控制循环)
│   └── config.py            # PlannerConfig
├── environments/
│   └── __init__.py          # RetroEnvironment (工具分发器)
├── tools/
│   ├── __init__.py          # RetroTool 协议
│   ├── disconnect.py        # DisconnectionTool (ONNX 推理 + 官能团检测)
│   ├── propose.py           # ProposalTool (模板应用 + fallback 扫描)
│   ├── evaluate.py          # EvaluationTool (可行性 + 库存评分)
│   ├── stock.py             # StockTool (ZINC 库存查询)
│   ├── literature.py        # LiteratureTool (模板分类检索)
│   ├── condition.py         # ConditionTool (反应条件推荐)
│   └── bash_tool.py         # BashTool (subprocess 执行)
├── config/
│   ├── default.yaml         # 默认配置（模型路径 + LLM + Agent + Environment）
│   └── config.local.yaml    # 本地覆盖（gitignored，放 API key）
├── loops/                   # Phase 3: Loop Engineering
└── run/
    └── retro.py             # CLI 入口

models/                      # 模型文件平铺存放
├── uspto_model.onnx                   # 扩展策略网络 (2048→42554)
├── uspto_filter_model.onnx            # 反应可行性过滤网络
├── uspto_ringbreaker_model.onnx       # 环断裂专用策略网络
├── full_uspto_truncated_42554.hdf5    # USPTO 模板库（截断到 ONNX 维度）
├── full_uspto_03_05_19_unique_templates.hdf5  # 原始 USPTO 模板库 (46,695 条)
└── zinc_stock_17_04_20.hdf5           # ZINC 库存 (17.4M InChI Keys)
```

## 快速开始

### 环境要求

- Python ≥ 3.10
- RDKit、ONNX Runtime、RDChiral、HDF5

### 安装

```bash
cd RetroAgent
python3 -m venv .venv
.venv/bin/pip install rdkit onnxruntime h5py rdchiral jinja2 pydantic pyyaml typer pandas openai
```

aizynthfinder 的 Python 版本要求 (<3.13) 与当前 Python 3.14 不兼容，因此项目通过文件系统直接导入其 `chem` 模块：

```python
import sys
sys.path.insert(0, 'aizynthfinder-master')
from aizynthfinder.chem import TreeMolecule
from aizynthfinder.chem.reaction import TemplatedRetroReaction
```

模型文件已经平铺放在 `models/` 目录下。

### 配置

```bash
# 1. 复制模板文件
cp retroagent/config/config.template.yaml retroagent/config/config.local.yaml

# 2. 编辑 config.local.yaml，填入你的 API key
#    config.local.yaml 已加入 .gitignore，不会被提交

# 3. 或者用环境变量（优先级最高）
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"
export LLM_BASE_URL="https://api.openai.com/v1"
```

配置加载优先级：环境变量 > `config.local.yaml` > `default.yaml`。

### 运行工具测试

```bash
PYTHONPATH=. .venv/bin/python3 -m retroagent.run.retro test-tools "CC(=O)Oc1ccccc1C(=O)O"
```

输出示例：

```
--- DisconnectionTool ---
  Molecule: 13 atoms, 1 rings
  Functional groups present: ['carboxylic_acid', 'alcohol', 'phenol', 'aldehyde']
  Functional groups absent:  ['ester', 'amide', 'amine', ...]
  50 templates, 0 substructure-matched
    ✗ N-acylation to amide (score=0.726)      ← LLM 发现: 分子无酰胺，不信任
    ✗ Heteroaryl N-alkylation (score=0.034)

--- ProposalTool ---
  propose(use_fallback=False): 0 reactions    ← 模型预测全部不匹配
  propose(use_fallback=True):  19 reactions   ← LLM 决定 fallback 扫描

--- EvaluationTool ---
  rxn_2: feasibility=0.92 stock=1.000 total=0.95

--- StockTool ---
  water acid ✓ salicylic acid ✓ acetic anhydride ✓ acetic acid ✓
```

## 当前状态

| 阶段 | 状态 | 内容 |
|------|------|------|
| Phase 1.1 | ✓ | 项目骨架 + RetroTool 协议 |
| Phase 1.2 | ✓ | SharedBlackboard 状态容器 |
| Phase 1.3 | ✓ | RetroEnvironment 工具分发器 + BashTool |
| Phase 1.4 | ✓ | 5 个核心 Tool (disconnect/propose/evaluate/stock/literature/condition) |
| Phase 1.5 | ✓ | RetroPlanner 控制循环 + PlannerConfig + System Template |
| Phase 1.6 | ✓ | 集成测试 — aspirin 合成 |
| Phase 1.7 | ✓ | YAML 配置系统 + 模型路径平铺 + OpenAI client |
| Phase 2 | 🔧 | LLM 驱动端到端规划（需配置 API key） |
| Phase 3 | 待启动 | Loop Engineering (Inner/Outer/Retrospective) |
| Phase 4 | 待启动 | 完整工具集 + Benchmark 评估 |

## 关键设计决策

1. **Tool 不做策略决策**：`propose(use_fallback=True)` 由 LLM 决定是否调用，Tool 不自动 fallback
2. **Tool 诚实报告质量**：`disconnect` 返回 `matching` 标志 + 官能团分析，让 LLM 交叉验证
3. **ONNX 模型仅作为弱信号**：USPTO 训练数据偏向复杂药物分子，对简单目标预测不准。LLM 可通过化学知识绕过模型
4. **LLM 不必微调**：Outer Loop 只更新 Tool 内部模型参数，LLM 保持通用推理能力

## License

MIT
