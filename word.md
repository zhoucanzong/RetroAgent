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
11. ✅ 建立 `score_routes` 工具，让 AND-OR RouteGraph 枚举并排序完整/部分路线。

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
- `RouteGraph` 已新增 `enumerate_routes()` 与 `scored_routes()`：从 target OR node 递归组合 reaction AND nodes，输出 solved/partial candidate routes，并按闭合程度、未解决前体、步数、反应类别多样性、收敛步骤和保护/脱保护负担评分。
- 新增 `retroagent/tools/route_scorer.py`：`score_routes` 可从 target、proposal reaction records 和 stock hits 构建路线图，返回 `routes`、`route_graph` 和 solved/partial/best-score summary。
- Discovery prompt 与 ToolGraph 已加入 `propose/evaluate/check_stock -> score_routes -> chemical_critic`，使 critic 和 final report 能基于整条路线排序，而不是孤立 reaction。
- Benchmark trajectory analyzer 已统计 `scored_route_count`、`solved_route_count`、`best_route_score`，用于衡量 agent 是否真正进行了路线级比较。
- 新增 `tests/test_route_scorer.py`；扩展 `tests/test_route_graph.py`、`tests/test_tool_graph.py`、`tests/test_discovery_mode.py`、`tests/test_benchmarks.py` 覆盖路线枚举/评分路径。
- 人工复杂 discovery 测试目标：`COc1cc(C(=O)N[C@H](C)c2ccncc2)cc(S(=O)(=O)N2CCOCC2)c1OC(F)(F)F`。测试显示模型能提出 late-stage amide coupling、sulfonamide-first、late-stage OCF3 等路线想法，但会绕过 `score_routes / chemical_critic / design_experiments / build_research_report / research_memory`，说明 prompt-only 约束不足。
- 修复真实运行暴露的模板 bug：system template 访问 `tool.examples` 前先判断 `is defined`，避免无 examples 的工具在 StrictUndefined 下导致 prompt 渲染失败。
- 修复 auto-tune 预算问题：复杂分子不再把用户显式传入的低 `--max-steps` 放大到 150；`step_limit < 100` 时保持用户上限。
- 新增 discovery workflow guard：当已有 proposal/evaluation 后，Planner 会动态提示下一步必须调用 `score_routes -> chemical_critic -> design_experiments -> build_research_report -> research_memory(record)`；如果 discovery 模式缺少这些关键工具却尝试 final submission，Planner 会阻止提交并要求补齐流程。
- 第二轮人工复杂 discovery 测试显示：仅靠动态 guard 仍不够，模型会继续调用 `check_patent / recommend_conditions` 等旁路工具而不进入 `score_routes`。
- 新增 discovery 执行层硬闸门：候选反应出现后，在完成 `score_routes -> chemical_critic -> design_experiments -> build_research_report -> research_memory(record)` 当前节点前，Planner 会阻止无关工具调用，并返回明确的 workflow blocked warning。
- 新增 discovery 自动路线评分：`propose` 或 stock 更新填充候选反应后，Planner 会自动插入一次 `score_routes`，避免模型因为忽略提示而停留在孤立反应层面。
- `SharedBlackboard` 已记录 `score_routes` 输出的 `routes` 与 route-level scores，使后续 critic/report/benchmark 能直接读取路线级排序。
- 新增测试覆盖：自动 `score_routes` 插入、候选路线后阻止旁路工具、blackboard 记录 route score。当前 `unittest` 全量为 59 项通过。
- 验证命令：
  - `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`
  - `.venv/bin/python -m unittest discover -s tests -v`
  - `.venv/bin/python -m compileall retroagent tests`
  - `.venv/bin/python -m retroagent.run.retro run --help`
  - `.venv/bin/python -m retroagent.run.retro benchmark-discovery --dry-run --limit 2 --output-dir /tmp/retroagent_discovery_bench_test`
  - `PYTHONPATH=. .venv/bin/python -m retroagent.run.retro test-tools "CC(=O)Oc1ccccc1C(=O)O"`

### 2026-07-11 DiscoveryController v2

- 新增 `retroagent/research_state.py`：使用 append-only `ResearchLedger` 保存研究事件，并可重放为 `ResearchState`。状态包含目标、子目标、竞争假设、路线生命周期、反应状态、证据、决策、执行记录、审批记录和失败动作指纹。
- 新增 `retroagent/research_feedback.py`：计算反馈可绑定具体 reaction/route；不可修复失败会将所有受影响路线标为 `failed`，可修复失败标为 `challenged` 并降低 feasibility/evidence strength、提高 uncertainty。
- 新增 `retroagent/agents/discovery_controller.py`：discovery 模式不再执行固定 `score_routes -> critic -> experiments -> report` 硬流水线，而执行 `OBSERVE -> DECIDE -> AUTHORIZE -> EXECUTE -> ASSIMILATE -> REPLAN`。Agent 每轮只提交一个带研究问题、理由、预期观测和证伪条件的结构化决策。
- Agent 可使用 `state` 动作显式提出 hypothesis 或设置 subgoal；这些动作只组织研究，不伪装成证据。工具、Python、系统命令和反馈结果才会产生 Evidence。
- Agent 可自主编写并执行临时 Python，或在 run workspace 中执行系统命令。每次执行记录代码/命令、cwd、stdout、stderr、返回码、超时和产物路径。
- 新增 `restricted / standard / full` 权限模式：在真正的 OS sandbox 接入前，standard 对所有临时 Python、系统命令、`bash` tool、联网、memory 写入、安装、项目写入、破坏性和特权操作请求审批；full 仅在用户显式提供非空 scope 时跳过逐项审批，空 scope 不再自动升级成 `*`。
- 当前执行隔离是路径与策略 guardrail，不是操作系统级安全沙箱；`code_execution` 表示任意 Python 能力，`system` 表示任意 shell 能力，两者都可能产生传递性的文件/网络效果，不伪装成细粒度隔离。不要把 full 模式交给不可信模型。
- 失败的工具/代码执行会生成 `action_failed + replan_required`，相同 action fingerprint 的原样重试会在执行前被拒绝；agent 必须改变参数、证据依据或研究分支。
- 重试所声明的 `new_evidence_ids` 必须在该动作失败之后产生并与当前 target 相关；旧证据、无关证据或单独的 `changed_basis=true` 不能绕过重复保护。
- 普通 route/reaction 注册事件不能把 `failed / superseded / completed` 终态恢复成 active；只有显式状态变更事件才能改变终态。
- Agent observation 会压缩大型工具结果，只保留 evidence 摘要、结果键和大小；完整原始结果仍保存在 append-only ledger 中。
- `score_routes` 的现有 `id/route_id` 输出和 `propose` reaction 输出已适配到 v2 研究状态；普通 retrosynthesis/design 仍由 `RetroPlanner` 执行。
- CLI `run --mode discovery` 已切换到 v2，并新增 `--permission-mode`、`--permission-scopes`、`--run-workspace`、`--ledger-output`。`benchmark-discovery --run` 也使用 v2 事件轨迹。
- 新增 `benchmarks/discovery_feedback.jsonl`、确定性 `FeedbackOracleTool` 和 `benchmark-discovery-feedback`。闭环通过要求同时满足 `failure_recorded / failed_route_abandoned / distinct_replan / evidence_cited / failed_action_not_repeated / failure_evidence_cited`。
- Feedback benchmark 校验 route ID 唯一、route A 初始 feasibility 最高、feedback reaction 属于 route A，且 oracle 只响应 route A；最终重新选择失败路线会导致 benchmark 失败。v2 event analyzer 已补齐旧 rubric 所需的 proposal/score/evidence/risk/validation 指标。
- Controller 已捕获 tool adapter 异常、非 dict 返回和 malformed state event，将其转为失败 Evidence 后继续 replan；fenced JSON 决策也可解析。
- `estimated_cost.time_minutes` 已进入确定性预算：ResearchState 累计已用预算，超出 `max_estimated_minutes` 的动作会在执行前拒绝；CLI 默认以配置的 wall-time 分钟值作为 estimated-time 上限。
- ResearchLedger 内部事件与外部 payload 深拷贝隔离；重复 evidence ID、语义缺字段等 replay 损坏会报告 event index 和最后有效 index。Evidence/Event ID 都支持非连续恢复，不会按数量覆盖旧记录。
- 最终独立 code review 在多轮反例修复后未发现残留 P0/P1；全量 `unittest` 为 116 项通过。
- 正式设计与逐文件实施计划位于 `docs/superpowers/specs/2026-07-11-discovery-controller-v2-design.md` 和 `docs/superpowers/plans/2026-07-11-discovery-controller-v2.md`。
- 验证命令：
  - `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`
  - `PYTHONPATH=. .venv/bin/python -m compileall -q retroagent tests`
  - `PYTHONPATH=. .venv/bin/python -m retroagent.run.retro run --help`
  - `PYTHONPATH=. .venv/bin/python -m retroagent.run.retro benchmark-discovery-feedback --dry-run --output-dir /tmp/retroagent_feedback_bench_test`

### 2026-07-16 最新论文严格盲测

- 选择 2026-07-15 在线发表的 Nature 论文 `Selectivity Emerges from Indiscriminate Photoreduction`（DOI: `10.1038/s41586-026-10897-7`）作为训练截止后盲题。
- 冻结输入为 styrene 与 `1-(2,2-dimethylcyclopropyl)propan-1-one` 的 perylene electrophotocatalysis 条件，只告诉 Agent styrene 更易还原约 0.30 V；论文标题、URL、标准产物和评分规则不进入 Agent prompt。
- 基线纯推理运行错误选择 H1（styrene reduction），暴露出 Agent 把初始还原电位排序直接等同于最终产物选择性，且没有主动取得区分 forward SET 与 post-SET branching 的证据。
- 新增 `benchmarks/blind_studies.jsonl`、`BlindStudyOracleTool` 和 `benchmark-discovery-blind`。Agent 可主动申请 transient absorption、electrochemical controls、post-SET barriers 和 structural analogue 测量；Oracle 只在调用后返回测量，不返回论文结论。
- Blind runner 使用显式 allowlist environment，只暴露 `blind_study_oracle / think / functional_groups / molecule_properties`；不注册 `research_memory`、文件读取、文献或网络工具，避免 Agent 读取版本化 case 中的 source、expected outcome 和 scorer。
- Blind scorer 要求至少 3 项主动测量、正确 outcome、完整 mechanism chain 和无矛盾结论。机理必须覆盖：两底物的 forward SET regime、ketyl beta-scission 与 BET 竞争、styrene radical anion 的 BET recycling、以及 post-SET branching 而非 redox ordering 控制产物。
- Source title、URL、expected outcome、success criteria 和 mechanism scoring chain 均不进入 Agent prompt/goal state；Agent 只看到原始问题、可申请测量及“至少申请 3 项”的过程约束。Scorer 另有否定/反向因果反例，`H2 is ruled out / H1 major / BET faster than beta-scission / styrene not recycled` 不能通过。
- 修复真实模型运行暴露的结构化协议问题：兼容扁平 tool action 和数值 `estimated_cost`；`run_python/run_command` 广告别名会归一化到对应 execution path；schema/policy rejection 会记录有界原始响应并在下一轮 observation 中带回拒绝动作和原因。
- 修复 open-ended discovery completion：无 route graph 时，goal-level evidence 可支持 `H1/H2/H3` 等命名假设；有路线任务仍要求 route/evidence 严格关联。
- 修复 stopping criteria 的可审计扩展写法：接受 `exact criterion: evidence detail`，仍拒绝缺项、任意意译或不匹配前缀。
- 最终独立运行使用 3 项隐藏测量，正确选择 H2：photoexcited perylene 使两底物都可快速 SET；ketyl beta-scission（5.5 kcal/mol）抢在 BET 前发生，open radical 加成 styrene 并闭环；styrene radical anion 则由 BET 回收，因此形成 `[3+2]` annulation，而非按 0.30 V 电位优势得到 H1。
- 正式结果：`exit_status=Submitted`，`outcome_pass=true`，6/6 mechanism groups 通过，`measurement_pass=true`，`contradiction_pass=true`，`blind_pass=true`；CLI 汇总为 `Blind passed: 1/1`。
- 验证命令：

  - `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`
  - `PYTHONPATH=. .venv/bin/python -m compileall -q retroagent tests`
  - `.venv/bin/python -m retroagent.run.retro benchmark-discovery-blind --dry-run --output-dir /tmp/retroagent_blind_dryrun`
  - `.venv/bin/python -m retroagent.run.retro benchmark-discovery-blind --run --max-cycles 16 --output-dir logs/discovery_blind_benchmark_verified_final4`

### 2026-07-22 第二篇论文严格盲测

- 选择 2026-07-13 发表的 Nature Chemistry 论文 `Electrophilic amination-induced 1,2-boronate migration for the modular synthesis of beta-amino boronic esters`（DOI: `10.1038/s41557-026-02206-6`）。该 case 与 BET case 不重复，测试时间顺序、中间体光谱归属、anti 立体化学和不相容试剂的自限机制。
- 冻结题面要求在 H1 协同 amination/migration、H2 快速 aziridinium 形成后慢速 backside 1,2-migration、H3 radical/ordinary aziridination 中选择；论文标题、DOI、H2、隐藏数值和评分 profile 不进入 Agent prompt。
- 新增 `aziridinium_migration` 独立方向性 scorer，并保留 `photoreduction_post_set` 原 scorer；新增 `--case-id` 以便单独运行一个 blind case。Scorer 覆盖真实输出中的括号 H2、跨句时间顺序、反事实 falsifier、`asynchronous`/`syn` 边界和 H1/H3 被排除时的否定语义。
- Prompt 增加通用 completion self-audit：停止前逐项核对题目要求与已记录证据，不得用无证据推测填补；同一测量中的多条必要观察必须带入结论；最终显式标记 `Uncertainty:` 和 `Falsification:`。
- 首轮运行选择 reaction-progress IR、spectral assignment 和 computed energy surface，正确推出 H2，但漏选唯一解释 product overreaction 的 stereochemical/reagent controls，并无证据猜测 DPPH 行为，严格失败。
- 后续独立运行开始主动选择 stereochemical/reagent controls。最终运行正确得到：1195 cm-1 transient 与 1193 cm-1 aziridinium 计算光谱匹配；transient 先于 product 积累；backside 1,2-migration 解释 anti stereoselectivity；DPPH 在 free amine 出现前被消耗；H1/H3 被排除，并给出 uncertainty 与 falsifier。
- 最终能力边界：Agent 再次遗漏 controls 中 `DPPH sparingly soluble` 这一条，未完整说明低有效浓度与时间分离共同抑制 overreaction。按当前 scorer 独立重评：`outcome_pass=true`，7 个 mechanism groups 中 6 个通过，`measurement_pass=true`，`contradiction_pass=true`，`blind_pass=false`。本轮停止继续调 prompt，保留失败作为“机制推理正确、证据摘要完整性仍不稳定”的真实结果。
- 最终轨迹位于忽略目录 `logs/discovery_blind_benchmark_boronate_final/`；实际只执行 `blind_study_oracle`，且 Agent-visible goal 未命中 source title、DOI、hidden numeric results、expected outcome 或 scoring profile。
- 验证命令：

  - `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`
  - `PYTHONPATH=. .venv/bin/python -m compileall -q retroagent tests`
  - `.venv/bin/python -m retroagent.run.retro benchmark-discovery-blind --dry-run --output-dir /tmp/retroagent_blind_two_case_dryrun`
  - `.venv/bin/python -m retroagent.run.retro benchmark-discovery-blind --run --case-id boronate-amination-2026-002 --max-cycles 16 --output-dir logs/discovery_blind_benchmark_boronate_final`

### 2026-08-28 Agent-native CLI Foundation

- 参考 OfficeCLI 的 Agent-first 设计，将 RetroAgent 从 Python 启动入口扩展为可安装的 `retroagent` CLI；新增 `pyproject.toml`，支持 `pip install -e .`、`pipx`/`uvx` 风格调用。wheel 显式包含默认 YAML 配置，包版本统一为 `0.2.0`，非源码安装可用 `RETROAGENT_MODELS_DIR` 指向外部模型权重目录。
- 新增 `.retroagent/` 项目工作区：版本化 manifest、append-only `ledger.jsonl`、`artifacts / cache / reports / runs / views / checkpoints` 分区。运行目录、egg-info 和构建产物已加入 `.gitignore`，不提交本地研究状态。
- 新增稳定资源地址 `molecule / hypothesis / route / reaction / evidence / experiment : id`，以及 `status / events / view / query / explain / validate / replay` 命令；所有新命令支持统一 `--json` 成功/错误 envelope 和机器可读错误码。
- 新增 schema 驱动的 `retroagent help --json`、`tool list --json`、Skill 导出和内置 MCP stdio bridge。CLI、`skills/retroagent/SKILL.md` 与 MCP tool description 使用同一个 capability registry，测试阻止漂移。
- 新增 `agent run / resume / stop / status / approve / reject`。Controller 的 approval request 现在保存完整 decision 和因果父事件；用户批准后，resume 会先精确执行一次原 pending action、记录 `approval_consumed`，再让 Agent 基于新 Evidence 继续规划，不会把批准泛化成长期权限。
- 新增项目化 `tool call`、`exec python`、`exec command`。standard 模式继续阻止高风险动作；用户可显式一次性 `--approve`，full 模式仍要求 manifest 中存在明确 scope。
- 新增原子 `batch`：先在内存账本逐项重放和语义校验，全部成功后使用 fsync + `os.replace` 替换 ledger；`--dry-run` 不写状态。原子性只覆盖数字研究状态，不宣称回滚真实实验或外部操作。
- 新增 `checkpoint create/list/diff/restore`。restore 前自动创建 backup checkpoint；`replay` 可从当前 ledger、checkpoint 或指定 event 截止点重建状态。
- 新增 HTML/SVG/PNG 研究视图：路线图、RDKit 分子图、单条 Evidence、Evidence matrix、单个 Experiment 和 Experiment timeline。视图默认写入项目 `views/`，可作为审查和 CI artifact。
- 新增 `tests/test_cli_foundation.py`，覆盖项目初始化、稳定寻址、查询、原子失败不落盘、检查点恢复、视图、JSON contract、权限、Skill/MCP schema 一致性，以及 approved action exactly-once resume。
- 当前验证：`PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_*.py'` 为 157 项通过；现有 DiscoveryController 和 ResearchLedger 定向回归 40 项通过。
- 验证命令：
  - `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_*.py'`
  - `PYTHONPATH=. .venv/bin/python -m compileall -q retroagent tests`
  - `PYTHONPATH=. .venv/bin/python -m retroagent.run.retro --help`
  - `PYTHONPATH=. .venv/bin/python -m retroagent.run.retro help --json`
  - `PYTHONPATH=. .venv/bin/python -m retroagent.run.retro project init /tmp/retroagent-cli-smoke --json`
  - `.venv/bin/pip install -e . --no-deps`

### 2026-08-29 Discovery Loop v3

- 新增 `select_experiment`：在多个可证伪假设之间计算先验熵、每个实验的期望后验熵和期望信息增益，再按时间、材料、计算成本与风险调整效用；输出完整排序而非只给一个黑箱选择。
- 新增 `update_hypotheses`：观测到实验结果后执行 Bayes 更新，记录归一化 posterior、likelihood ratio、posterior entropy 和 entropy reduction。Controller 只信任该内置工具产生的后验事件，普通工具不能伪造状态转换。
- 新增 `build_route_portfolio`：合并 template、template-free、mechanism、analogy 等来源；用 RDKit canonical structure 去重，保留来源支持、模型分歧，并以 reaction fingerprint 相对已知参考集计算 novelty/extrapolation distance。
- 新增 `validate_route_portfolio` 独立门控：不复用生成器评分，分别检查结构、目标覆盖、多来源支持、atom mapping、forward prediction、条件完整性和库存证据；矛盾为 `failed`，缺证据为 `needs-evidence`，全部必需检查通过才是 `passed`。
- ResearchLedger 新增 `experiment_selected / hypothesis_posterior_updated / route_portfolio_built / route_validated` 事件，并在 observation、project replay 和稳定 JSON state 中暴露，完整研究过程可确定性重放。
- DiscoveryController 对带 `goal.discovery_v3` 的任务执行硬停止契约：竞争假设数、主动实验选择、后验更新、最大后验熵、多来源路线、最低 novelty 与独立验证任一缺失都不允许提交。旧 discovery 与严格论文盲测默认不启用该契约，避免改变历史基准边界。
- Agent-native CLI 新增 `agent run --discovery-v3`、`--max-posterior-entropy` 和 `--min-route-novelty`；旧入口 `run --mode discovery` 同样支持。新增 `score-discovery-v3 TRAJECTORY --require-pass`，按事件而非最终答案关键词评分六项闭环能力。
- ToolGraph 已连接 `chemical_critic/design_experiments -> select_experiment -> update_hypotheses -> build_route_portfolio -> validate_route_portfolio -> report/memory`；四个新工具进入常规 Discovery 环境，但不进入现有 blind-study allowlist。
- 新增单元与完整脚本 Agent 轨迹测试，覆盖实验效用排序、概率归一化、结构等价路线去重、novelty 估计、独立证据缺失/矛盾/通过三态、事件回放、CLI completion contract 和 v3 trajectory scorer。

### 2026-09-26 RetroChimera 多来源共识融合

- 研究并引用 Microsoft [RetroChimera](https://github.com/microsoft/retrochimera) 及论文 `Chemist-aligned retrosynthesis by ensembling diverse inductive bias models`。核心启发不是简单增加一个模型，而是在 canonical reactant-set 层聚合具有互补归纳偏置的来源，并使用 source-by-rank 权重求和提升跨来源共识候选。
- 新增可选 `retrochimera_predict` 适配器：兼容 RetroChimera 1.3+，通过 `RETROCHIMERA_MODEL_DIR` 或参数加载 checkpoint，支持 `consensus_only`、并行 submodel 调用和 1-10 个候选；输出转换为 RetroAgent reaction contract，同时保留 constituent model ranks、metadata、ensemble score 与 probability。
- `pyproject.toml` 新增 `retrochimera` optional extra，默认安装不引入其 PyTorch/Syntheseus 依赖，模型和 checkpoint 不在本仓库分发。
- 新增模型无关 `ensemble_disconnections`：接受 template、template-free、数据库或人工来源的有序候选；用 RDKit canonical SMILES 把无序 reactant set 去重，按校准的来源 × 排名权重求和，支持 consensus-only、最小来源数、softmax probability 和完整 provenance。没有校准 profile 时明确标记 reciprocal-rank fallback。
- rank-weight profile 强制非负、随排名单调不增且凸，避免任意权重破坏排序含义；超过 10 个输出会产生严格过滤警告。ensemble score 被显式标注为排序信号，不得作为 feasibility evidence。
- `build_route_portfolio` 现在能从单个 ensemble candidate 中展开 constituent sources 与 individual scores；修复了多来源候选被压成同一个最高分、导致 model disagreement 错算为 0 的问题。
- ToolGraph 更新为 `disconnect/propose/retrochimera_predict -> ensemble_disconnections -> build_route_portfolio -> validate_route_portfolio`。DiscoveryController 被明确要求优先组合不同归纳偏置，并继续执行 mapping、forward prediction、conditions、stock 与独立验证门控。
- Discovery benchmark 轨迹分析新增 `retrochimera_call_count / retrochimera_candidate_count / ensemble_call_count / ensemble_candidate_count / consensus_candidate_count`，可以区分“工具已注册”和“Agent 实际利用了跨来源共识”。

### 2026-09-27 RetroChimera 真实神经推理

- 建立隔离的 `.venv/retrochimera` CPU 环境：Python 3.11.15、RetroChimera 1.3.0、Syntheseus 0.9.0、PyTorch 2.2.2、Lightning 2.2.2、RDKit 2023.9.6；依赖固定在 `requirements-retrochimera-cpu.txt`，`uv pip check` 通过。
- 官方 Figshare 下载返回 HTTP 403，改用社区 `Ian-L/backwards-models-v2` 的 Pistachio safetensors 转存，固定 revision `30cc1962fa8de3decdc32f624b9c2ece324d2b45`。逐文件 SHA-256 校验通过，重建 inference-only Lightning checkpoint，保存来源记录。校验保证与固定镜像一致，尚未独立验证与原始 Microsoft 下载包逐字节等价。
- 新增 `scripts/prepare_retrochimera_mirror.py` 和 `scripts/smoke_retrochimera.py`。模型位于忽略目录 `models/retrochimera/`，运行轨迹位于 `logs/retrochimera_smoke/`。复现步骤详见 `docs/retrochimera-cpu.md`。
- 适配器增加按 checkpoint 子模型名称传递 `submodel_kwargs`，参数进入模型缓存键；真实调用禁用 autograd。修正发布版 1.3.0 尚无 `canonicalization_processes` 参数的兼容问题，保留其默认实现。
- 通过正式 `RetroChimeraPredictionTool` 跑官方目标 `Oc1ccc(OCc2ccccc2)c(Br)c1`：beam size 20、augmentation size 10、CPU 4 线程、模板 worker 2 个。两子模型完整加载并推理，冷启动加预测耗时 172.95 秒，返回 5 条有效 SMILES 候选，进程退出码 0。
- 首位前体 `BrCc1ccccc1` + `Oc1ccc(O)c(Br)c1`，两个子模型均排第 1，集成归一化概率 0.957151；这只证明真实推理与协议接通，不能当作反应成功率或盲测泛化结论。
- 完整预测、模型版本、耗时及来源保存在 `logs/retrochimera_smoke/result.json`；完整日志为 `full_inference.log`。上游退出时报告一个 semaphore 清理警告，已核对无残留推理 worker。
- 适配器定向回归 8 项、路线组合 8 项、DiscoveryController 31 项通过；脚本编译及 diff 检查通过。另有 Transformer 单模型真实试跑约 1.84 秒，使用 reduced beam/augmentation，仅用于排查运行兼容性。

### 2026-09-27 Agent 可自主调用的神经模型脚手架

- 明确架构：RetroAgent 是研究决策主体，NeuralLoc、R-SMILES 2 Transformer、RetroChimera 是可选能力；ToolGraph 只提供建议，Agent 自主选择调用、比较、验证或放弃候选，不强制三模型流水线。
- 新增 `neuralloc_predict` 与 `transformer_predict`，保留集成入口 `retrochimera_predict`。单模型可以接受集成根目录并按 manifest 定位组件，或直接接受组件目录；`model_kwargs` 只传给该组件。三个入口共用适配器的规范化、缓存、记录和结果协议。
- 结果增加可直接用于融合的 `ensemble_source`、实际 checkpoint 路径 `source_ids`、明确的 rank base 和生效推理选项。融合器拒绝重叠来源（重复/改名调用，以及集成模型与其自身组件再次投票），并拒绝不同目标分子的候选混入。
- 规范化反应物时清除 atom-map 编号，保留立体化学；内容哈希反应 ID 避免不同目标的第一个候选都写入同一 `rxn_0001`。
- 同一 Agent 环境的三个工具共享一个活动模型配置缓存；切换时释放旧模板 worker，重复调用复用已加载模型。CPU 默认模板 worker 数为 2；模型日志及 worker stdout 转入 stderr，修复真实 CLI 输出被上游日志污染的问题。
- 降低 RDKit 最低版本到已实测的 2023.9.6，使完整 RetroAgent 可安装到 Python 3.11 神经运行环境；CPU requirements 加入 PyTables，用于现有模板库加载。原 Python 3.14 环境保持兼容。
- 真实模型验证：NeuralLoc 独立入口返回 5 项，冷启动加推理 199.02 秒；Transformer 独立入口默认参数返回 5 项，5.51 秒。两份输出原样用于融合，得到 5 项候选，首位具有两个来源支持。保存于 `neuralloc_standalone.json / transformer_standalone.json / standalone_fusion.json`。
- 神经环境的正式 CLI 已暴露 32 个工具，`transformer_predict --json` 真实执行、纯 JSON 解析和证据写入通过。脚本模型 Controller 测试验证可自主选择两个独立工具而不被要求调用集成；统计中预测/融合不再计作化学验证。
- 原开发环境和神经运行环境各自通过 192 项回归；更新 README 中英文架构、模型入口说明、能力状态，以及 `docs/retrochimera-cpu.md` 的复现命令。尚未把软件通过或模型共识当成真实新路线可行性的证明。
