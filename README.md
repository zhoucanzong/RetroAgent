# RetroAgent

<div align="center">

🧪 **LLM-driven retrosynthesis and chiral ligand design**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

RetroAgent 是一个借鉴 Claude agent 设计哲学的化学推理系统——**LLM 是唯一的决策中枢，专用化学工具全部退化为纯函数（只计算和获取事实，从不替模型做判断）**，并由独立的隔离审查 sub-agent 纠错、免费层文献检索接地。覆盖逆合成路线规划、手性配体与金属催化剂设计、以及文献查证。

> 智能只存在于 **Planner（LLM）**。专用化学模型（ONNX、模板库、库存、RDKit）与外部检索（文献、URL）全部退化为 **Tool** —— 纯函数，不做决策。判断（路线好坏、催化剂对错）由 LLM 在隔离的审查上下文里完成。

---

## ✨ 核心能力

| 模式                      | 输入            | 核心工具链                                                        | 输出                      |
| ------------------------- | --------------- | ----------------------------------------------------------------- | ------------------------- |
| **Retrosynthesis**  | 目标分子 SMILES | `disconnect` → `propose` → `evaluate` → `check_stock`  | 完整合成路线              |
| **Ligand Design**   | 自然语言约束    | `design_ligand` → `analyze_chirality` → `classify_ligand` | 候选手性配体              |
| **Catalyst Assembly** | 结构化约束 | `design_catalyst` (配位数/d电子/labile位点计算) → Auditor 审查 | 金属催化剂描述符 + 事实报告 |
| **Literature Search** | 查询词 | `web_search` (Crossref/S2/PubChem) → `fetch_url` | 文献命中 + 摘要 |
| **Chiral Analysis** | 任意 SMILES     | `analyze_chirality` + `classify_ligand`                       | 手性类型 / R/S / 配位原子 |

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
│   ├── evaluate.py          # EvaluationTool (可行性 + 库存评分 + 常用试剂识别)
│   ├── stock.py             # StockTool (ZINC 库存查询)
│   ├── literature.py        # LiteratureTool (模板分类检索)
│   ├── condition.py         # ConditionTool (反应条件推荐)
│   ├── bash_tool.py         # BashTool (subprocess 执行)
│   ├── chirality.py         # ChiralityTool (立体化学分析)
│   ├── ligand_category.py   # LigandCategoryTool (SMARTS 配体骨架分类)
│   ├── conditional_ligand.py # ConditionalLigandTool (模板驱动配体生成)
│   ├── catalyst.py          # CatalystTool (结构化催化剂计算器，只算不判)
│   ├── think.py             # ThinkTool (虚拟推理空间，类似 Claude think tool)
│   ├── web_search.py        # WebSearchTool (免费层文献检索 Crossref/S2/PubChem)
│   └── fetch_url.py         # FetchUrlTool (URL 抓取 + HTML 清洗)
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

- https://figshare.com/articles/dataset/AiZynthFinder_a_fast_robust_and_flexible_open-source_software_for_retrosynthetic_planning/12334577
- https://zenodo.org/records/7341155
- https://zenodo.org/records/7797465

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
pip install -r requirements.txt
```

> 可选：在虚拟环境中安装以隔离依赖（`python3 -m venv .venv && source .venv/bin/activate`），但下面的示例均假设依赖已装好、直接用 `python3` 运行。

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
PYTHONPATH=. python3 -m retroagent.run.retro test-tools "CC(=O)Oc1ccccc1C(=O)O"
```

### 2. 逆合成规划

```bash
PYTHONPATH=. python3 -m retroagent.run.retro run "CC(=O)Oc1ccccc1C(=O)O"
```

### 3. 手性配体设计

```bash
PYTHONPATH=. python3 -m retroagent.run.retro run \
  "Point chirality ligand with P and O donor atoms" \
  --mode design
```

### 4. 保存轨迹

```bash
PYTHONPATH=. python3 -m retroagent.run.retro run "..." -o /tmp/traj.json
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
design_ligand ──▶ 模板库匹配骨架 + RDKit 取代基枚举（确定性）
    │
    ▼
analyze_chirality ──▶ 验证手性类型 / 立体中心 / R/S
    │
    ▼
classify_ligand ──▶ SMARTS 验证骨架 / 齿数 / 配位原子
    │
    ▼
[金属催化剂] design_catalyst ──▶ 计算配位数 / d电子 / labile位点 / 对称性（只算不判）
    │
    ▼
web_search ──▶ 查证文献先例（Crossref/S2/PubChem，免费）
    │
    ▼
Design Auditor (隔离 sub-agent) ──▶ 事实快照 + 文献对照 → ISSUE + FIX_SUGGESTIONS + VERDICT
    │
    ▼
LLM 根据审查反馈修正并提交
```

> 上下文压缩在每步前自动运行（>60K 字符时老 tool output 压成摘要），让 100 步长任务可持续。

---

## 📊 当前状态

| 阶段      | 状态 | 内容                                                                                                                          |
| --------- | ---- | ----------------------------------------------------------------------------------------------------------------------------- |
| Phase 1.1 | ✅   | 项目骨架 + RetroTool 协议                                                                                                     |
| Phase 1.2 | ✅   | SharedBlackboard 状态容器                                                                                                     |
| Phase 1.3 | ✅   | RetroEnvironment 工具分发器 + BashTool                                                                                        |
| Phase 1.4 | ✅   | 5 个核心 Tool (disconnect/propose/evaluate/stock/literature/condition)                                                        |
| Phase 1.5 | ✅   | RetroPlanner 控制循环 + PlannerConfig + System Template                                                                       |
| Phase 1.6 | ✅   | 集成测试 — aspirin 合成                                                                                                      |
| Phase 1.7 | ✅   | YAML 配置系统 + 模型路径平铺 + OpenAI client                                                                                  |
| Phase 1.8 | ✅   | 手性配体设计扩展：ChiralityTool / LigandCategoryTool / ConditionalLigandTool                                                  |
| Phase 1.9 | ✅   | 环断裂策略网络 + 模板库接入                                                                                                   |
| Phase 2   | ✅   | LLM 驱动端到端规划（逆合成 + 配体设计均已跑通）                                                                               |
| Phase 3   | ✅   | Loop Engineering: Think Tool + Dead-Loop Monitor + Enhanced Observation + Adaptive Scaling + Branch Tracking + Design Auditor |
| Phase 4   | ✅   | Claude 哲学重构: 隔离 Auditor sub-agent + design_catalyst 计算器 + classify_ligand SMARTS 修复 + 上下文压缩 |
| Phase 5   | ✅   | 模型侧优化: design_ligand 模板化 + 全免费层 web_search/fetch_url + 文献接地的审查 |
| Phase 6   | ⏳   | Benchmark 评估 + 更强 LLM (Auditor 用 Opus/GPT-4 级) |

---

## 🧠 关键设计决策

1. **Tool 不做策略决策**：`propose(use_fallback=True)` 由 LLM 决定是否调用，Tool 不自动 fallback
2. **Tool 诚实报告质量**：`disconnect` 返回 `matching` 标志 + 官能团分析，让 LLM 交叉验证
3. **ONNX 模型仅作为弱信号**：USPTO 训练数据偏向复杂药物分子，对简单目标预测不准。LLM 可通过化学知识绕过模型
4. **LLM 不必微调**：Outer Loop 只更新 Tool 内部模型参数，LLM 保持通用推理能力
5. **兼容非原生 tool-calling 的模型**：LLMClient 同时支持 OpenAI `tool_calls` 和文本 JSON 块解析，适配 DeepSeek 等模型
6. **Think Tool — 显式推理空间**：借鉴 Claude 的 extended thinking 设计。`think` 工具是纯文本、无副作用的虚拟工具，让模型在调用实际化学工具前"停下来思考"。思考内容以 `<thinking>` 标签注入对话历史。不计入 iteration count
7. **Dead-Loop Monitor — 防死循环安全网**：5 个纯启发式检测器（无 LLM 调用）：Tool-Type Cycling、Semantic Repeat、Stagnation Timer、Early Exit Hint、Evaluate Loop。步数提高到 100+ 后防止浪费
8. **Mode-Aware Auditing**：Retrosynthesis 模式有 Peer Reviewer（6 维度路线审核），Design 模式有 Design Auditor（6 维度催化剂设计审核：对称性、配位饱和、手性来源、辅助配体、文献可比性、氧化态）。Auditor 给出 ISSUE + FIX_SUGGESTIONS + AUDIT_VERDICT
9. **Adaptive Scaling**：RDKit 分析分子复杂度，自动调整 step_limit（简单≤30 / 中等≤100 / 复杂≤150）
10. **Branch Tracking Table**：动态注入分支探索状态表（Markdown 表格），防止长链路中丢失探索目标
11. **Tool 只计算，Model 才判断**（Claude 哲学核心）：`design_catalyst` 只报告事实（配位数、d 电子、labile 位点、对称性候选、氧化态合理性），**绝不**返回 REJECT/APPROVE。化学判断完全在 LLM (Auditor) 里
12. **独立视角用全新上下文 sub-agent**：Auditor / Peer Reviewer 不在主对话里切换角色（会自我锚定），而是跑在 2 条消息的隔离快照上下文——只看事实，不看主 agent 的推理过程。审查质量是系统瓶颈，值得这次 LLM 调用
13. **结构化表示解决表示问题**：金属配合物不用单一 SMILES（RDKit/LLM 都难写），改用结构化描述符（有机配体 SMILES + 金属/氧化态/几何 + 每配体齿数/供体原子/数量）。`design_catalyst` 据此计算事实
14. **上下文工程 > 加代码**：100 步长任务靠 `_compact_messages()` 支撑——老的 tool output 压成一行摘要，保护 thinking 块、审查 verdict、提交信号、最近 8 条工作窗口（幂等）
15. **确定性模板生成 + 外部知识获取**：`design_ligand` 用 12 个家族的真实代表性 SMILES 库 + RDKit 取代基枚举（确定性，不依赖 LLM 随机性）；`web_search` 用免费层（Crossref/Semantic Scholar/PubChem，无需 key）让 agent 能查证文献
16. **文献接地的审查**：审查前主动 web_search 拉相关文献，作为事实注入隔离快照。Auditor 同时看设计事实 + 真实先例，在一个隔离调用里判断（不让审查器自己调工具，保持单次调用）
17. **SMARTS 子结构匹配**：`classify_ligand` 用最小特征 SMARTS（any-bond `~` 匹配）而非完整 SMILES，BINAP 现在能匹配 BINAP；N,N'-二氧化物需 ≥2 个 N-oxide 才确认
18. **ZINC 局限性诚实标注**：ZINC 是虚拟筛选化合物库（17.4M 类药分子），不含 NaBH₄/Pd 催化剂/常用溶剂等试剂。`evaluate` 内置常用试剂 SMARTS 白名单，ZINC 查询前先识别，避免假阴性误导

---

## 📜 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
