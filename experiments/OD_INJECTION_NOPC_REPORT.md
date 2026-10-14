# 3D-LLaVA OD 注入实验报告（EXP-OD6：去掉点云——GT bbox 文本独存的点云边际价值测量）

> 日期：2026-10-13 ~ 10-14　|　前置：[EXP-OD1 全量](OD_INJECTION_EXP1_REPORT.md)、[OD2 topk](OD_INJECTION_TOPK_REPORT.md)、[OD3 names](OD_INJECTION_NAMES_REPORT.md)、[OD4 rel](OD_INJECTION_REL_REPORT.md)、[OD5 GT](OD_INJECTION_GT_REPORT.md)

---

## 1. 摘要

**实验问题**：OD 注入系列证明了收益 = 物体清单 + 选择，检测器精度无关（OD5）。那反过来的问题——**点云本身还贡献什么？** 本实验在 OD5 基础上把点云整个去掉：LLM 输入只有 GT bbox 文本 + 问题，无 `<pc>` token、无点云张量（代码级真移除，`--no-pc` 门控）。若 OD6 ≈ OD5 → 点云塔对 VQA 无边际贡献；若明显低 → 量化点云贡献并定位其作用题型。

**结论：点云对 VQA 有真实但类型化集中的边际贡献，sqa3d 上几乎可弃、scanqa 上 −2~3 分**——

- **sqa3d：点云可弃**。OD6 vs 完整多模态 baseline 配对平手（−0.40, p=0.61），vs OD5（GT+点云）−1.19（p=0.0897）也不显著——**GT 框列表 + 语言先验即可打平整个点云编码器**
- **scanqa：点云值 −2~3 EM**。vs baseline −1.99（p=0.0002）、vs OD5 −2.95（p<1e-4）、vs names −2.78（p<1e-4），全部显著；CIDEr 掉 −9.6（83.48 vs 93.11）
- **掉点结构（与预注册的"颜色题崩塌"一致，但方向在分桶上反转）**：scanqa **非空间桶**掉最狠（vs OD5 −3.81, p<1e-4）——无空间关键词的题大量是颜色/外观题，RGB 只存在于点云；**空间桶**掉得少（−2.09）——空间关系可从框坐标推导，OD 文本部分替代了几何感知
- **题型分解（sqa3d）**：How-many 计数题 OD6 = 54.41，比 baseline **+5.38**（OD5 55.05）——框列表可直接数，无点云也保住 OD 注入的全部计数增益；What 识别题 −2.6、Can-see 可见性题 −3.3（vs baseline）——外观与视点信息是点云独占
- **架构事实**：ref-seg（mask 从点云特征解码）与 scan2cap（`<loc>` 指定目标物体）**无点云即任务不成立**，本实验仅覆盖 VQA 两任务

**对工程路线的含义**：点云塔的价值集中在外观（颜色/材质）与视点可见性信息；几何/计数/存在性信息已被 bbox 文本覆盖。若下游只要 VQA 类能力，"检测器 → 文本清单 → 纯 LLM"管线理论上可省掉整个 3D 编码器（sqa3d 实证平手），但 open-ended QA（scanqa/CIDEr）与全部几何任务仍需要点云。

## 2. 方法

- **数据**：`*_with_od_gt_nopc.json` ×2 = OD5 注入文件剥掉 human 轮开头 5 字节 `<pc>\n`，其余逐字节不变（GT 块、问题、gpt 轮、scene_id、task_type 全同）——OD6 vs OD5 的输入差异**只有点云**（训练集构成差异除外，见 §7）。scanqa 26,515 + sqa3d 26,623 = **53,138** 条（8 文件中仅这 2 个可纯文本训练：nr3d/sr3d/scanrefer/multi3drefer 答案为 `[SEG].` 需 mask 监督，scan2cap/nr3d_caption 用 `<loc>` 指定目标）；token p50=814 / p95=1840 / max=3338
- **代码**（全部 flag 门控，关闭时行为不变）：`train.py` 加 `--no_pc`（dataset 跳过点云加载、`has_image=False` 走纯文本 mask 路径、collator 因样本无 `scene_id` 键自动跳过点云块、`is_multimodal` 假图垫片被门控）；`llava_llama.py` 的 `forward_mask_decoder_train` 加 None 守卫；`model_scanqa/sqa3d.py` 加 `--no-pc`（od 块置于 prompt 首部 = 训练格式、跳过点云块、generate 走 `embed_tokens` 纯文本分支）。模型侧本就有纯文本分支（`llava_arch.py:269/321`），未改模型
- **审计**：mask 边界完美（受监督 span = 答案+`</s>`、masked 尾 = `ASSISTANT:`、400 抽样 0 全掩码）；带点云默认路径复测未破坏

## 3. 运行环境

- 训练：**GPU 0,3**（1,2 他人占用）、累积 8 → 全局 batch 32 = OD5 同优化规模；**4h17m**（53k 条、无 tower 前向）
- 评估：GPU 0,3 两分片；答案数 sqa3d 3519 / scanqa 4675 全对、0 OOM、ALL_EVAL_DONE

## 4. 七方对比（VQA；‡=最低。ref-seg/scan2cap 无点云不成立，全系列沿用 OD5 行）

| 指标 | baseline | OD1 全量 | OD2 topk | OD3 names | OD4 rel | OD5 GT | **OD6 GT-nopc** |
|---|---|---|---|---|---|---|---|
| sqa3d EM1 | 53.99 | 54.25 | 54.36 | 54.53 | 53.40‡ | **54.79** | 53.59 |
| sqa3d EM1_refined | 56.66 | 57.00 | 56.98 | 57.12 | 55.78‡ | **57.80** | 56.15 |
| scanqa EM1 | 25.48 | 26.40 | 26.07 | 26.27 | 26.42 | **26.44** | 23.49‡ |
| scanqa EM1_refined | 41.20 | 42.20 | 42.03 | 41.39 | **43.23** | 42.57 | 38.91‡ |
| scanqa CIDEr | 93.11 | 91.24 | 93.16 | **93.46** | 87.49 | 91.78 | 83.48‡ |

## 5. 配对检验（EM1，McNemar 精确）

| 对比（sqa3d） | Δ | p | 对比（scanqa） | Δ | p |
|---|---|---|---|---|---|
| OD6 vs baseline | −0.40 | 0.61 | OD6 vs baseline | −1.99 | **0.0002** |
| OD6 vs OD1(全量) | −0.65 | 0.37 | OD6 vs OD1 | −2.91 | **<1e-4** |
| OD6 vs OD5 | −1.19 | 0.0897 | OD6 vs OD5 | −2.95 | **<1e-4** |
| OD6 vs names | −0.94 | 0.20 | OD6 vs names | −2.78 | **<1e-4** |

分桶（scanqa EM1，vs OD5 / vs baseline）：spatial −2.09 (p=0.0066) / −1.79 (p=0.026)；**nonspatial −3.81 (p<1e-4) / −2.18 (p=0.0032)**——非空间（颜色/外观题为主）受创更重。CIDEr 分桶（vs baseline）：spatial 76.71 vs 85.49（−8.8）、nonspatial 89.65 vs 100.08（−10.4）——长答案流畅度全域依赖点云。

## 6. 题型分解（sqa3d EM1；n=1147/652/465/338/351/566）

| 题型 | baseline | OD5 GT | **OD6** | OD6−base | OD6−OD5 |
|---|---|---|---|---|---|
| type2 How-many（计数） | 49.03 | 55.05 | **54.41** | **+5.38** | −0.64 |
| type5 Are/If/Where | 53.18 | 53.18 | **54.06** | +0.88 | +0.88 |
| type1 Is（是否） | 68.71 | 69.33 | 68.25 | −0.46 | −1.08 |
| type4 Which（辨别） | 45.87 | 47.86 | 45.87 | 0.00 | −1.99 |
| type0 What（识别） | 45.16 | 44.64 | 42.55 | −2.61 | −2.09 |
| type3 Can-see（可见性） | 72.19 | 70.71 | 68.93 | −3.26 | −1.78 |

**读法**：计数题把 OD 注入增益完整保住（框列表可数）；识别与可见性题贡献了主要掉点（外观/视点是点云独占信息）；是否题不动（语言先验主导）。

## 7. 局限与混杂

- **任务混合混杂（最大混杂，与点云缺失同向）**：OD5 训 293k 多任务（含 caption/ref-seg 迁移），OD6 训 53k 纯 VQA。scanqa −2.95 中有多少来自多任务迁移损失无法分离；但 sqa3d 的平手（p=0.61）说明该混杂不足以在所有 VQA 上制造显著差距。混对齐控制（53k 纯 VQA + 点云）未做，需要可补
- 15 条 scene0154_00（GT 空）样本退化为纯问题输入，与 OD5 处理对称
- eval prompt 无 `<pc>` 后 od 块置于句首，与训练格式逐字节同构；但与 OD5 的注入位置（`<pc>` 后）在文本版式上略有差异
- ref-seg/scan2cap/nr3d/sr3d 未评估：前两者架构不可能，后两者本就无独立 val 集

## 8. 产物

- 数据：`scripts/generate_with_od_gt_nopc_data.py`、`playground/data/train_info/{scanqa,sqa3d}_train_3d_llava_with_od_gt_nopc.json`
- 代码：`train.py`/`llava_llama.py`/`model_scanqa.py`/`model_sqa3d.py` 的 flag 改动
- 脚本：`scripts/train/finetune-3d-llava-lora-with-od-gt-nopc.sh`、`scripts/eval/multigpu_eval_all_with_od_gt_nopc.sh`
- checkpoint：`checkpoints/finetune-3d-llava-lora-with-od-gt-nopc/`（LoRA）；预测+指标：`playground/predictions/finetune-3d-llava-lora-with-od-gt-nopc/`
- 分析：`scripts/eval/spatial_split_eval.py`（第 7 run，`runs_for` 过滤 + FULLCHECK_REF 真值回填）、`scripts/eval/spatial_split_paired.py`（4 新对）；日志 `logs/spatial_split_nopc.log`、`logs/spatial_split_paired_nopc.log`
