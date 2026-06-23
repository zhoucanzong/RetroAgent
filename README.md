# RetroAgent

<div align="center">

🧪 **LLM-driven retrosynthesis and chiral ligand design**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

RetroAgent 是一个基于 **LLM 中枢决策 + 专用化学工具** 的逆合成路线规划与手性配体设计系统。

> 智能只存在于 **Planner（LLM）**。专用化学模型（ONNX、模板库、库存、RDKit）全部退化为 **Tool** —— 纯函数，不做决策。

---

## ✨ 核心能力

| 模式 | 输入 | 核心工具链 | 输出 |
|------|------|-----------|------|
| **Retrosynthesis** | 目标分子 SMILES | `disconnect` → `propose` → `evaluate` → `check_stock` | 完整合成路线 |
| **Ligand Design** | 自然语言约束 | `design_ligand` → `analyze_chirality` → `classify_ligand` | 候选手性配体 |
| **Chiral Analysis** | 任意 SMILES | `analyze_chirality` + `classify_ligand` | 手性类型 / R/S / 配位原子 |

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Planner (LLM)                               │
│              唯一决策中枢：选择策略、交叉验证、终止判断                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Retrosynthesis │      │  Ligand Design  │      │   Chiral Eval   │
│     Workflow    │      │    Workflow     │      │    Workflow     │
│                 │      │                 │      │                 │
│ disconnect ──▶  │      │ design_ligand   │      │ analyze_chirality│
│ propose ────▶   │      │ analyze_chirality│     │ classify_ligand │
│ evaluate ───▶   │      │ classify_ligand │      │ evaluate        │
│ check_stock ──▶ │      │ check_stock     │      │ check_stock     │
└───────────────┘      └─────────────────┘      └─────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    Shared Blackboard   │
                    │   纯状态容器，无决策逻辑  │
                    └───────────────────────┘
```

### 架构参考

- **[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)**：Agent 控制循环、Environment 协议、异常体系
- **[AiZynthFinder](https://github.com/MolecularAI/aizynthfinder)**：ONNX 策略网络、USPTO 模板库、RDChiral 模板应用、ZINC 库存

---

## 📁 项目结构

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
│   ├── disconnect.py        # DisconnectionTool (ONNX 推理 + 官能团检测 + 环断裂)
│   ├── propose.py           # ProposalTool (模板应用 + fallback 扫描)
│   ├── evaluate.py          # EvaluationTool (可行性 + 库存评分)
│   ├── stock.py             # StockTool (ZINC 库存查询)
│   ├── literature.py        # LiteratureTool (模板分类检索)
│   ├── condition.py         # ConditionTool (反应条件推荐)
│   ├── bash_tool.py         # BashTool (subprocess 执行)
│   ├── chirality.py         # ChiralityTool (立体化学分析)
│   ├── ligand_category.py   # LigandCategoryTool (配体骨架分类)
│   └── conditional_ligand.py # ConditionalLigandTool (约束→候选配体)
├── config/
│   ├── default.yaml         # 默认配置（模型路径 + LLM + Agent + Environment）
│   └── config.local.yaml    # 本地覆盖（gitignored，放 API key）
├── loops/                   # Phase 3: Loop Engineering
└── run/
    └── retro.py             # CLI 入口

models/                      # 模型文件平铺存放（不提交到 Git）
├── uspto_model.onnx                       # 扩展策略网络 (2048→42554)
├── uspto_filter_model.onnx                # 反应可行性过滤网络
├── uspto_ringbreaker_model.onnx           # 环断裂专用策略网络
├── full_uspto_truncated_42554.hdf5        # USPTO 模板库（截断到 ONNX 维度）
├── full_uspto_03_05_19_unique_templates.hdf5  # 完整 USPTO 模板库 (46,695 条)
├── uspto_unique_templates.csv             # USPTO 模板库（CSV 格式）
├── uspto_ringbreaker_unique_templates.csv # 环断裂模板库
└── zinc_stock_17_04_20.hdf5               # ZINC 库存 (17.4M InChI Keys)
```

### 模型下载

模型文件需手动下载并平铺放在 `models/` 目录下：

- **Zenodo 7797465**：`uspto_model.onnx`, `uspto_filter_model.onnx`, `uspto_ringbreaker_model.onnx`
- **Zenodo 7341155**：`uspto_unique_templates.csv.gz`, `uspto_ringbreaker_unique_templates.csv.gz`
- **Figshare 12334577**：`zinc_stock.hdf5`, `full_uspto_03_05_19_unique_templates.hdf5`

> 大文件不提交到 Git，见 `.gitignore`。

---

## 🚀 快速开始

### 环境要求

- Python ≥ 3.10
- RDKit、ONNX Runtime、RDChiral、HDF5、OpenAI SDK

### 安装

```bash
cd RetroAgent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> aizynthfinder 的 Python 版本要求 (`<3.13`) 与当前 Python 3.14 不兼容，因此项目通过文件系统直接导入其 `chem` 模块：

```python
import sys
sys.path.insert(0, 'aizynthfinder-master')
from aizynthfinder.chem import TreeMolecule
from aizynthfinder.chem.reaction import TemplatedRetroReaction
```

### 配置

创建本地覆盖文件（已加入 `.gitignore`）：

```bash
cat > retroagent/config/config.local.yaml << 'EOF'
llm:
  api_key: "sk-..."
  model: "deepseek-v4-flash"
  base_url: "https://api.deepseek.com"
EOF
```

或使用环境变量（优先级最高）：

```bash
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"
export LLM_BASE_URL="https://api.openai.com/v1"
```

配置加载优先级：**环境变量 > `config.local.yaml` > `default.yaml`**。

---

## 🛠️ 使用示例

### 1. 工具测试（无需 API key）

```bash
PYTHONPATH=. .venv/bin/python3 -m retroagent.run.retro test-tools "CC(=O)Oc1ccccc1C(=O)O"
```

### 2. 逆合成规划

```bash
PYTHONPATH=. .venv/bin/python3 -m retroagent.run.retro run "CC(=O)Oc1ccccc1C(=O)O"
```

### 3. 手性配体设计

```bash
PYTHONPATH=. .venv/bin/python3 -m retroagent.run.retro run \
  "Point chirality ligand with P and O donor atoms" \
  --mode design
```

### 4. 保存轨迹

```bash
PYTHONPATH=. .venv/bin/python3 -m retroagent.run.retro run "..." -o /tmp/traj.json
```

---

## 🔄 工作流说明

### 逆合成工作流

```
目标 SMILES
    │
    ▼
search_literature ──▶ 查已知路线
    │
    ▼
disconnect ──▶ 断键建议（带 matching 标志 + 官能团分析）
    │
    ▼
LLM 判断模型预测是否可信 ──┬── 可信 ──▶ propose(model templates)
                             │
                             └── 不可信 ──▶ propose(use_fallback=True)
    │
    ▼
evaluate ──▶ 可行性 + 库存评分
    │
    ▼
check_stock ──▶ 验证原料可及性
    │
    ▼
全部前体 in stock ? 完成 : 递归展开
```

### 手性配体设计工作流

```
自然语言约束
    │
    ▼
design_ligand ──▶ 生成候选 SMILES
    │
    ▼
analyze_chirality ──▶ 验证手性类型 / 立体中心 / R/S
    │
    ▼
classify_ligand ──▶ 验证骨架 / 齿数 / 配位原子
    │
    ▼
evaluate + check_stock ──▶ 可行性与可及性
    │
    ▼
LLM 选择最佳候选并提交
```

---

## 📊 当前状态

| 阶段      | 状态   | 内容                                                                         |
| --------- | ------ | ---------------------------------------------------------------------------- |
| Phase 1.1 | ✅     | 项目骨架 + RetroTool 协议                                                    |
| Phase 1.2 | ✅     | SharedBlackboard 状态容器                                                    |
| Phase 1.3 | ✅     | RetroEnvironment 工具分发器 + BashTool                                       |
| Phase 1.4 | ✅     | 5 个核心 Tool (disconnect/propose/evaluate/stock/literature/condition)       |
| Phase 1.5 | ✅     | RetroPlanner 控制循环 + PlannerConfig + System Template                      |
| Phase 1.6 | ✅     | 集成测试 — aspirin 合成                                                     |
| Phase 1.7 | ✅     | YAML 配置系统 + 模型路径平铺 + OpenAI client                                 |
| Phase 1.8 | ✅     | 手性配体设计扩展：ChiralityTool / LigandCategoryTool / ConditionalLigandTool |
| Phase 1.9 | ✅     | 环断裂策略网络 + 模板库接入                                                   |
| Phase 2   | ✅     | LLM 驱动端到端规划（逆合成 + 配体设计均已跑通）                              |
| Phase 3   | ⏳     | Loop Engineering (Inner/Outer/Retrospective)                                 |
| Phase 4   | ⏳     | 完整工具集 + Benchmark 评估                                                  |

---

## 🧠 关键设计决策

1. **Tool 不做策略决策**：`propose(use_fallback=True)` 由 LLM 决定是否调用，Tool 不自动 fallback
2. **Tool 诚实报告质量**：`disconnect` 返回 `matching` 标志 + 官能团分析，让 LLM 交叉验证
3. **ONNX 模型仅作为弱信号**：USPTO 训练数据偏向复杂药物分子，对简单目标预测不准。LLM 可通过化学知识绕过模型
4. **LLM 不必微调**：Outer Loop 只更新 Tool 内部模型参数，LLM 保持通用推理能力
5. **兼容非原生 tool-calling 的模型**：LLMClient 同时支持 OpenAI `tool_calls` 和文本 JSON 块解析，适配 DeepSeek 等模型

---

## 📜 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

RetroAgent builds upon [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (MIT) and
[AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) (MIT), which remain under their
respective licenses.
