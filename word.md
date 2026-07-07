# RetroAgent 工作文档：ChemMCP 借鉴与化学 Agent 路线图

> 本文档说明 [ChemMCP](https://github.com/OregonNLP/ChemMCP)（OSU NLP Group 出品的化学 MCP 工具包）的内容、与 RetroAgent 的关系，以及我们要借鉴/移植哪些部分、如何实现。

> 2026-07-07 更新：本文件由 `CHEMMCP_INTEGRATION.md` 重命名为 `word.md`，作为 RetroAgent 的显式工作文档持续维护。

---

## 1. ChemMCP 是什么

ChemMCP 是一个面向通用 LLM 的化学工具包，以 **MCP server**（Model Context Protocol，FastMCP 框架）形式提供 **36 个化学工具**。定位是"广而平"——让 GPT/Claude 这类通用模型不用微调就能做分子分析、性质预测、反应合成等任务。源自 OSU NLP 的 ChemToolAgent 工作。

设计特点：
- 每个工具继承统一的 `BaseTool` 抽象类，元数据（描述、输入输出、示例）规范化
- 支持 MCP server 模式（stdio/SSE）和直接 Python 调用两种用法
- ML 模型**懒加载**（首次调用才载入权重）
- 用 LiteLLM 做多 provider LLM 抽象

---

## 2. 工具清单（36 个，按类别）

### 通用（2）
| 工具 | 功能 | 依赖 |
|------|------|------|
| `WebSearch` | 网络搜索 | Tavily（需 key）|
| `PythonExecutor` | Jupyter 执行 Python（带 Docker）| jupyter/docker |

### 分子性质预测（6）— Uni-Mol 模型，本地权重
| 工具 | 预测目标 | 数据集 |
|------|---------|--------|
| `SolubilityPredictor` (esol) | 水溶性 logS | ESOL |
| `LogDPredictor` (lipo) | logD 分配系数 | Lipophilicity |
| `BbbpPredictor` (bbbp) | 血脑屏障透过 | BBBP |
| `ToxicityPredictor` (clintox) | 毒性概率 | ClinTox |
| `HivInhibitorPredictor` (hiv) | HIV 抑制剂活性 | HIV |
| `SideEffectPredictor` (sider) | 27 类副作用 | SIDER |

### 标识符转换（6）— PubChem 免费 API
`Name2Smiles`、`Iupac2Smiles`、`Smiles2Iupac`、`Smiles2Formula`、`Smiles2Cas`、`Smiles2Selfies`/`Selfies2Smiles`

### 分子分析（5）— RDKit 本地
`MoleculeSimilarity`（Tanimoto）、`MoleculeWeight`、`MoleculeAtomCount`、`FunctionalGroups`（55+ SMARTS 官能团）、`MoleculeVisualizer`

### 分子生成/修改（3）— ML 模型
`MoleculeCaptioner`/`MoleculeGenerator`（MolT5）、`MoleculeModifier`（synspace，提出化学可达的类似物）

### 可及性 / IP / 安全（3）
| 工具 | 功能 | 依赖 |
|------|------|------|
| `PatentCheck` | 是否专利/可购 | **molbloom（纯本地，几 MB）** |
| `MoleculePrice` | 商业可购 + 报价 | ChemSpace（需 key）|
| `SafetyCheck` | GHS 分类 + 管制品 | 本地管制品 CSV |

### 反应（2）— IBM RXN（需 key）
`Retrosynthesis`（单步逆合成）、`ForwardSynthesis`（正向产物预测）

### 信息检索（3）
`PubchemSearch`、`PubchemSearchQA`（LLM 增强）、`ReactionSmilesCheck`

---

## 3. 与 RetroAgent 的关系

### 真实重叠（不多）
| 能力 | RetroAgent | ChemMCP | 关系 |
|------|:---:|:---:|------|
| Web search | ✅ 免费层（Crossref/S2/PubChem）| ✅ Tavily | 重叠，保留我们的免费版 |
| 逆合成 | ✅ USPTO 模板 + ONNX | ✅ IBM RXN | 不同路线，可互补验证 |

### RetroAgent 的缺口（ChemMCP 有，我们没有）
| 缺口 | ChemMCP 工具 | 对 RetroAgent 的价值 |
|------|-------------|---------------------|
| **正向合成验证** | `ForwardSynthesis` (IBM RXN) | 🟢 高 — Auditor 可验证"前体真合成得出目标"|
| **性质预测** | Uni-Mol 套件 (6 个) | 🟡 中 — esol/lipo 对配体药代有用；其余偏药物发现 |
| **专利/可购检查** | `PatentCheck` (molbloom) | 🟢 高 — 比 ZINC 假阴性少，几 MB 本地 |
| **官能团识别** | `FunctionalGroups` (55+ SMARTS) | 🟡 中 — 现在 disconnect 附带，独立工具更干净 |
| **标识符转换** | Name/IUPAC/CAS↔SMILES | 🟡 中 — 模型用名字时有用，PubChem 免费 |
| **类似物修改** | `MoleculeModifier` (synspace) | 🟡 中 — 配体优化阶段 |
| **安全检查** | `SafetyCheck` | 🟡 低-中 — 试剂安全性 |

### 我们**不**需要的
- BBBP / HIV / SIDER — 药物发现专用，与逆合成/催化无关（虽然随 Uni-Mol 包一起来）
- MoleculePrice (ChemSpace) — 需 key，优先级低
- MolT5 captioner/generator — 与我们的 design_ligand 模板化路线不同

---

## 4. 借鉴方案与优先级

采用**第 4 条路径：借鉴思路 + 选择性移植**（不是整体跑 MCP server，也不盲目 pip 拉 chemmcp）。

### P0 — 零成本、纯本地、立刻有用
1. **`check_patent` 工具**（molbloom）— `pip install molbloom`，查分子是否可购/专利。替代 ZINC 对常见试剂的假阴性。
2. **`functional_groups` 工具** — 纯 RDKit SMARTS，55+ 官能团识别。独立于 disconnect。

### P1 — 免费 API、易实现
3. **`name_to_smiles` / `smiles_to_cas` 等转换工具** — PubChem PUG REST（web_search 已接 PubChem，封装成专门工具）。

### P2 — Uni-Mol 性质预测（⚠️ 环境受阻，已降级）
4. **Uni-Mol 性质预测 — 当前不可用**。两条路径都在本环境（Python 3.14）受阻：
   - **ChemMCP 路径**：需 `unicore`（Uni-Core C++ 扩展）— 不在 PyPI，GitHub 克隆超时
   - **unimol-tools 路径**：PyPI 可装，但只提供 backbone 权重（`mol_pre_*.pt`），**不含**微调任务模型（esol/bbbp 等）— 需自行训练
   - **deepchem 路径**：需 tensorflow + 训练，同样无现成预训练任务模型
   - **替代交付**：`molecule_properties` 工具用 RDKit 描述符（MW/MolLogP/TPSA/HBD-HBA/Lipinski/反应性警示）填补性质维度，确定性、即时、无需权重
   - **后续可行路径**：换 Python 3.10/3.11 环境装 unicore，或等 Uni-Core 出 py3.x wheel；届时把 ChemMCP 的 Zenodo 15299461 checkpoints（~2.6GB）放到 `models/unimol_checkpoints/`

### P3 — 需 key，按需
5. **`forward_synthesis` 工具** — IBM RXN。填补"前体验证"缺口。需 IBM RXN 免费账号 token。

### 暂不做
- MoleculePrice、MolT5、PythonExecutor、MoleculeModifier（synspace 较重）

---

## 5. 实现笔记

### 模型文件放置
- Uni-Mol 权重（未来）：`models/unimol_checkpoints/{esol,lipo,bbbp,clintox,hiv,sider}/checkpoint_best.pt`（已加 `.gitignore`）
- RDKit 描述符（已用）：无权重，即时计算

### 虚拟环境
- `.venv/`（Python 3.14.5）
- 已装：`molbloom`（专利检查）、`torch`+`unimol-tools`（后备，等 task checkpoint）、`huggingface_hub`

### 工具注册
所有新工具按现有协议实现，在 `retroagent/run/retro.py:_build_environment()` 注册。

### Claude 哲学约束
所有新工具**只计算/获取事实，不做判断**。性质描述符返回数值，专利检查返回 bool，官能团返回 present/absent 列表——好坏全由 LLM (Auditor) 判。

---

## 6. 状态

| 工具 | 状态 | 文件 | 来源 |
|------|:---:|------|------|
| `check_patent` (molbloom) | ✅ 已实现 | `retroagent/tools/patent.py` | ChemMCP PatentCheck |
| `functional_groups` (RDKit) | ✅ 已实现 | `retroagent/tools/functional_groups.py` | ChemMCP FunctionalGroups |
| `convert_identifier` (PubChem) | ✅ 已实现 | `retroagent/tools/identifiers.py` | ChemMCP Name/IUPAC/CAS 转换 |
| `molecule_properties` (RDKit) | ✅ 已实现 | `retroagent/tools/properties.py` | Uni-Mol 性质预测的确定性替代 |
| `map_reaction` (rxnmapper) | ✅ 已实现 | `retroagent/tools/reaction_mapper.py` | IBM rxnmapper（原子映射，Auditor 机理性审查用）|
| `examples` 元数据增强 | ✅ 已实现 | `environments/__init__.py` + system template | ChemMCP BaseTool 协议借鉴 |
| `predict_property` (Uni-Mol ML) | ❌ 放弃 | — | unicore 在 py3.14 装不了；6 模型里 4 个对催化化学无用 |
| `forward_synthesis` (IBM RXN) | ⏳ 待实现（需 key）| — | ChemMCP ForwardSynthesis |

---

## 7. 当前总目标：Chemical Research Agent

RetroAgent 的目标不只是复现已知逆合成路线，而是逐步发展成化学研究 agent：

1. **Synthesis Planner** — 生成、比较、递归展开多条合成路线。
2. **Discovery Mode** — 发掘 analog-supported / tool-supported / speculative 的新路线假设，而不是只找文献现成答案。
3. **Chemical Critic** — 独立审查机制、官能团兼容性、手性、选择性、条件现实性和安全风险。
4. **Experiment Designer** — 将路线假设转成可测试实验建议、对照实验和失败诊断路径。
5. **Research Memory** — 记录成功/失败路线、文献证据、用户偏好和实验反馈。

### 近期优先级

1. ✅ 修复官能团识别误判，并补测试。
2. ✅ 固化 `Molecule / Reaction / Route / Evidence / RiskFlag` 数据结构。
3. ✅ 将 blackboard 从列表升级为 AND-OR route graph。
4. ✅ 增加 `discovery` 模式骨架和 evidence-first 输出。
5. ✅ 建立小型 discovery benchmark，评估路线多样性、可行性、证据等级和风险标注。
6. ✅ 建立最小 ToolGraph，显式表达工具依赖、推荐后续工具和 safety level。
7. ✅ 建立结构化 `build_research_report` 工具，强制 discovery 输出路线假设、证据、风险和下一步实验。
8. ✅ 建立确定性 `chemical_critic` 工具，审查路线假设的证据、验证、库存、低分反应和实验计划缺口。
9. ✅ 建立 `design_experiments` 工具，把 critic issues 转换成可执行的验证实验、对照、成功标准和失败诊断。
10. ✅ 建立本地 `research_memory` 工具，记录/检索/总结 discovery 路线、critic、实验计划和结果标签。

### 2026-07-07 进展

- `functional_groups` 收紧羧酸、酯、醛、酮、醇、醚、胺等 SMARTS，避免 carboxylic acid/ester 被误判成 alcohol/aldehyde/ether。
- `disconnect` 的内置官能团摘要改为复用 `functional_groups` 的 broad summary，Planner 看到的分子事实与独立工具保持一致。
- 新增 `tests/test_functional_groups.py`：阿司匹林回归测试要求 present=`carboxylic_acid, ester`，absent=`phenol, alcohol, aldehyde, ketone, ether`。
- 新增 `retroagent/schemas.py`：定义 `EvidenceLevel / Evidence / RiskFlag / MoleculeNode / ReactionRecord / Route`，并提供 `build_reaction_record()` 与 `normalize_reaction_record()`。
- `propose` 与 `evaluate` 已接入共享 reaction schema；保留 `precursors` legacy alias，避免旧 prompt/轨迹立即失效。
- 新增 `tests/test_schemas.py`：覆盖 evidence/risk 序列化、canonical reaction 字段、legacy reaction payload 归一化。
- 新增 `retroagent/route_graph.py`：最小 AND-OR route graph。Molecule 是 OR 节点，Reaction 是 AND 节点；支持 target 插入、reaction 插入、canonical SMILES 去重、stock 标记、unresolved precursor 统计。
- `SharedBlackboard` 已维护 `route_graph`：初始化 retrosynthesis target 时创建 target molecule，收到 `propose` 输出时插入 reaction，收到 `check_stock` 输出时标记 stock。
- 新增 `tests/test_route_graph.py`：覆盖 reaction AND-OR 插入、canonical molecule 复用、stock 终止、blackboard 自动更新 route graph。
- 新增 `--mode discovery` 骨架：CLI help、blackboard route graph 初始化、Planner auto-tune/loop monitor、system prompt 和 instance prompt 均已支持 discovery。
- Discovery prompt 明确要求输出路线假设组合，并按 `known / analog-supported / tool-supported / speculative` 标证据等级，同时保留路线多样性和风险批判。
- 新增 `tests/test_discovery_mode.py`：覆盖 discovery 初始化 route graph，以及 prompt 渲染进入 `Discovery Strategy`。
- 新增 `benchmarks/discovery_small.jsonl` 与 `benchmarks/README.md`：首批 5 个非 aspirin discovery 任务（ibuprofen stereochemistry、lidocaine analog routes、naproxen stereochemical routes、warfarin coumarin routes、BINOL ligand routes）。
- 新增 `retroagent/benchmark.py` 与 CLI 命令 `benchmark-discovery`：支持 dry-run、limit、trajectory path、`results.jsonl` 输出、submission shape scoring，以及 trajectory tool-trace analysis。
- Trajectory analyzer 会从 Planner 轨迹中统计：tool sequence、proposal reaction count、reaction class/template/precursor diversity、evaluated route count、stock/literature hits、evidence/risk/validation tool usage、reviewer/auditor/warning messages、prompt/completion token totals。
- 新增 `tests/test_benchmarks.py`：校验 discovery benchmark JSONL 可解析、字段完整、SMILES 有效、目标非 aspirin、每个任务要求至少 3 条路线；同时覆盖 dry-run runner、submission 形状评分、synthetic trajectory analysis、runner 结果合并 trajectory metrics。
- Discovery benchmark 已有 pass/fail rubric：`shape_pass`、`route_diversity_pass`、`evidence_grounding_pass`、`validation_tool_pass`、`risk_review_pass`、`next_experiment_pass`、`overall_pass`。
- `benchmark-discovery` CLI summary 现在同时报告 `Rubric passed` 和 `Shape checks passed`；dry-run 不调用 LLM，只验证任务集和输出路径。
- 后续可以进一步接入人工/LLM 化学审查评分，特别是机制合理性、条件现实性和实验可行性。
- 新增 `retroagent/tool_graph.py`：记录工具 category、input/output types、requires、suggests_next、fallback_tools、safety_level 和 purpose。
- `RetroEnvironment.get_tool_graph_summary()` 与 Planner 模板变量已接入 ToolGraph；system prompt 现在包含 `ToolGraph Workflow Hints`，帮助模型按 `grounding -> disconnect -> propose -> evaluate/check_stock -> evidence/risk` 顺序行动。
- 新增 `tests/test_tool_graph.py`：覆盖 next-tool 查询、ToolGraph markdown 摘要、环境暴露摘要。
- 新增 research-agent 结构化对象：`HypothesisStatus / AuditIssue / ExperimentStep / ChemicalHypothesis`，并提供 `normalize_chemical_hypothesis()` 与 `build_research_report()`。
- 新增 `retroagent/tools/research_report.py`：`build_research_report` 只做 schema 归一化和完整性检查，不判断化学对错；用于 discovery final 前强制每条路线具备 claim、evidence、risk review、next experiment。
- `default.yaml` discovery strategy 已要求最终提交前调用 `build_research_report`，将自由文本路线组合压成可审计 JSON portfolio。
- ToolGraph 已加入 `build_research_report`，benchmark trajectory analyzer 会统计 `structured_report_count` 与 `complete_structured_report_count`，并把该工具纳入 risk/review 链路。
- 新增 `tests/test_research_report.py`；扩展 `tests/test_schemas.py`、`tests/test_tool_graph.py`、`tests/test_discovery_mode.py`、`tests/test_benchmarks.py` 覆盖结构化研究报告路径。
- 新增 `CriticReport` schema 与 `retroagent/tools/chemical_critic.py`：对 discovery hypotheses 做确定性审查，发现空 portfolio、speculative-only 证据、缺失 risk review、薄弱 next experiment、低 feasibility score、低频 template、未解决 precursor 等问题。
- `chemical_critic` 明确不证明路线化学正确，只输出可追踪 `AuditIssue` 和 `passed_checks`；LLM 需要把 warning/critical issue 修正，或在最终报告里标为未解决实验风险。
- `default.yaml` discovery strategy 已加入 `chemical_critic -> build_research_report -> final submission` 顺序，避免直接从自由文本路线跳到最终答案。
- ToolGraph 已加入 `chemical_critic`，并从 `map_reaction` / `check_patent` 等验证与风险工具推荐进入 critic；`build_research_report` 现在要求先经过 critic。
- Benchmark trajectory analyzer 已统计 `critic_report_count`、`critic_warning_count`、`critic_critical_count`，便于后续 discovery benchmark 区分“有路线”与“经过审查的路线”。
- 新增 `tests/test_chemical_critic.py`；扩展 schema、ToolGraph、discovery prompt、benchmark 测试覆盖 critic 路径。
- `chemical_critic` 已升级为 chemistry-aware deterministic critic：内部用 RDKit 解析 product/precursors，检查模板分类与产物官能团是否矛盾、未指定 stereocenter、polyfunctional chemoselectivity、反应性 precursor/safety alert、低置信 atom mapping、condition report 缺失等。
- `default.yaml` 已要求向 `chemical_critic` 传入 mapping reports 与 condition reports（若可用），使 final discovery portfolio 不只“有证据”，还经过机制/选择性/条件现实性审查。
- `chemical_critic` 已接收 serialized `route_graph`：审查 target 是否有 candidate reaction、未闭合 molecule node、反应类别/前体集合多样性、是否有 convergent step、以及 protection/deprotection-like loop 信号。
- 新增 route-level critic 测试：未解决前体 + 同质线性路线会触发 `route_graph / diversity / convergence` issue；库存闭合、多类别、多前体路线会通过 `route_graph_precursors_resolved / route_graph_route_diversity_present / route_graph_convergent_step_present`。
- 新增 `ExperimentPlan` schema 与 `retroagent/tools/experiment_designer.py`：`design_experiments` 将 `chemical_critic` 的 mechanism / stereochemistry / conditions / availability / route_graph / diversity / convergence / safety 等 issues 转换成结构化实验计划。
- `design_experiments` 输出 `steps`、`controls`、`success_criteria`、`failure_diagnostics`、`linked_issue_categories` 与 `ready_for_report`；仍然只提出实验，不声称实验已完成。
- Discovery prompt 与 ToolGraph 已更新为 `chemical_critic -> design_experiments -> build_research_report -> final submission`，使 speculative/new routes 必须带可测试的 next actions。
- Benchmark trajectory analyzer 已统计 `experiment_plan_count`、`experiment_step_count`、`experiment_control_count`，区分“写了 next experiment 字样”和“真正调用了实验设计工具”。
- 新增 `tests/test_experiment_designer.py`，扩展 schema、ToolGraph、discovery prompt、benchmark 测试覆盖 Experiment Designer 路径。
- 新增 `ResearchMemoryRecord` schema 与 `retroagent/tools/research_memory.py`：使用本地 append-only JSONL 记录 target、hypotheses、critic summary、experiment plan、route graph summary、outcome、tags、notes 和 timestamp。
- `research_memory` 支持 `record / search / summarize` 三种操作，默认路径为 `logs/research_memory.jsonl`；`logs/` 已加入 `.gitignore`，避免真实运行轨迹和记忆被误提交。
- Discovery prompt 现在要求开头先 `research_memory search/summarize`，最终 `build_research_report` 后再 `research_memory record`，形成研究闭环。
- ToolGraph 已加入 `research_memory`，并从 `search_literature` 与 `build_research_report` 连接到记忆节点。
- Benchmark trajectory analyzer 已统计 `memory_search_count` 与 `memory_record_count`，使 discovery benchmark 能观察 agent 是否复用和沉淀研究记忆。
- 新增 `tests/test_research_memory.py`；扩展 schema、ToolGraph、discovery prompt、benchmark 测试覆盖 Research Memory 路径。
- 验证命令：
  - `.venv/bin/python -m unittest discover -s tests -v`
  - `.venv/bin/python -m compileall retroagent tests`
  - `.venv/bin/python -m retroagent.run.retro run --help`
  - `.venv/bin/python -m retroagent.run.retro benchmark-discovery --dry-run --limit 2 --output-dir /tmp/retroagent_discovery_bench_test`
  - `PYTHONPATH=. .venv/bin/python -m retroagent.run.retro test-tools "CC(=O)Oc1ccccc1C(=O)O"`
