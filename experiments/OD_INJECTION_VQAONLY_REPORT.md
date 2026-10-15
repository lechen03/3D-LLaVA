# 3D-LLaVA OD 注入实验报告（EXP-OD8 —— vqaonly 两格：带点云、仅 VQA 数据训练，匹配 2×2 收官）

> 日期：2026-10-14　|　前置：[EXP-OD1 全量](OD_INJECTION_EXP1_REPORT.md)、[OD2 topk](OD_INJECTION_TOPK_REPORT.md)、[OD3 names](OD_INJECTION_NAMES_REPORT.md)、[OD4 rel](OD_INJECTION_REL_REPORT.md)、[OD5 GT](OD_INJECTION_GT_REPORT.md)、[OD6 去点云](OD_INJECTION_NOPC_REPORT.md)、[OD7 仅问题](OD_INJECTION_TEXTONLY_REPORT.md)

---

## 1. 摘要

**实验问题**：baseline（点云、无 OD）与 OD5（点云+GT OD）都训自 293k 多任务混合，OD6/OD7 训自 53k 纯 VQA——此前所有跨格对比都被任务混合混杂。本实验补上两个**带点云、同样只用 53k VQA 数据**的格子（vqaonly = pc+无 OD；with_od_gt_vqaonly = pc+GT OD），把 pc×OD 四格全部放到同一份数据、同一混合上，并顺带测出多任务迁移效应本身。

**结论：匹配混合下 OD 边际归零、点云边际减半——两信息源的冗余替代是对称的；多任务迁移小且任务不对称（sqa3d≈0、scanqa 约 +1~1.6）**——

- **OD 边际（有点云时）**：sqa3d +0.68（p=0.25）、scanqa +0.28（p=0.54）——**均不显著**；OD5-vs-baseline 的 +0.80/+0.96（293k）在匹配混合下没有复现出显著值。OD 文本的显著因果价值（无点云时 +2.10/+2.98）**只在点云缺席时兑现**
- **点云边际**：无 OD 时 +2.78（p<1e-4）/ +4.09（p<1e-4）；有 OD 时缩到 +1.36（p=0.0096）/ +1.39（p=0.0005）——**减半但保持显著**
- **冗余替代对称**：OD 边际（+2.10/+2.98 → +0.68/+0.28）与 pc 边际（+2.78/+4.09 → +1.36/+1.39）互相在场时都大幅缩水——OD7 报告的"高冗余信息编码"结论在完全匹配条件下成立
- **多任务迁移**：vqaonly vs baseline sqa3d −0.28（p=0.71，零迁移）、scanqa −0.88（p=0.087）；with_od_gt_vqaonly vs OD5 sqa3d −0.17（p=0.83）、scanqa −1.56（p=0.0018，集中在非空间桶 +2.36）——caption/ref-seg 多任务对 scanqa 有 ~1~1.6 分正迁移、对 sqa3d 无。**OD6/OD7 报告的混杂担忧：sqa3d 上可完全解除，scanqa 上此前跨格掉点约 1~1.6 分来自去迁移而非去信息**
- **点云 vs GT 框文本（同数据直接对打）**：sqa3d +0.68（p=0.28，平手——OD6 报告"sqa3d 点云可弃"在匹配条件下成立）；scanqa +1.11（p=0.023，点云略胜但远小于其对 OD7 的 +4.09）
- CIDEr：vqaonly 90.56（vs baseline 93.11，−2.6 = 迁移损失）；with_od_gt_vqaonly 88.37（OD 块再 −2.2，named 格式代价复现）

**对工程路线的含义**：53k 纯 VQA 训练即可达到 293k 多任务的 sqa3d 水平（54.28~54.96 vs 53.99~54.79 全在噪声内）；scanqa 差距 ~1~1.6 分是多任务迁移而非数据量。若目标是 VQA 单任务，训练混合可以砍掉 82% 数据（ref-seg/caption 样本）几乎无损——但 scanqa 的开放式问答偏好（CIDEr −2.6）会受影响。

## 2. 方法

- **数据（零生成）**：vqaonly = 原始 `{scanqa,sqa3d}_train_3d_llava.json`（`<pc>\n`+问题，53,138）；with_od_gt_vqaonly = OD5 的 `{scanqa,sqa3d}_train_3d_llava_with_od_gt.json`（`<pc>\n<od>…</od>\n`+问题）
- **零代码改动**：训练去掉 `--no_pc` 走默认点云路径（OD6 期间已验证该路径完好）；eval 一无 `--no-pc`/`--od-file`、eval 二加 `--od-file od_gt_val.json`（均原始路径）
- 匹配 2×2 四格（textonly / with_od_gt_nopc / vqaonly / with_od_gt_vqaonly）训练配置全同：GPU 0,3、全局 batch 32、1 epoch、LoRA r32/α64、lr 2e-4——**任意两格之间唯一差异是输入模态/OD 文本**

## 3. 运行环境

- 训练串跑：vqaonly **1h36m**（3.4s/step，短序列）、with_od_gt_vqaonly **4h57m**（10.7s/step，OD 块使序列 ~700-1800 token）
- 评估串跑：各 ~1h（带点云+beam5）；答案数 3519/4675 ×2 全对、0 OOM、双 ALL_EVAL_DONE

## 4. 匹配 2×2（EM1；全 53k 纯 VQA 训练；括号 = 格间配对差）

| | − OD | + GT OD | OD 边际 |
|---|---|---|---|
| **− 点云** | textonly 51.49 / 20.51 | OD6 53.59 / 23.49 | +2.10 (p=0.0005) / +2.98 (p<1e-4) |
| **+ 点云** | vqaonly 54.28 / 24.60 | with_od_gt_vqaonly 54.96 / 24.88 | **+0.68 (p=0.25) / +0.28 (p=0.54)** |
| **pc 边际** | +2.78 (p<1e-4) / +4.09 (p<1e-4) | +1.36 (p=0.0096) / +1.39 (p=0.0005) | |

参照行（293k 多任务）：baseline 53.99 / 25.48；OD5 54.79 / 26.44。

scanqa 分桶：pc 边际（有 OD 侧）spatial −1.32 (p=0.033) / nonspatial −1.46 (p=0.0048) 均匀；迁移（OD 格）集中 nonspatial +2.36 (p=0.0009)。

## 5. 配对检验（EM1，McNemar 精确）

| 对比（sqa3d） | Δ | p | 对比（scanqa） | Δ | p |
|---|---|---|---|---|---|
| with_od_gt_vqaonly vs vqaonly（OD 边际·匹配） | −0.68 | 0.25 | 同左 | −0.28 | 0.54 |
| with_od_gt_vqaonly vs with_od_gt_nopc（pc 边际·有 OD） | −1.36 | **0.0096** | 同左 | −1.39 | **0.0005** |
| vqaonly vs textonly（pc 边际·无 OD） | −2.78 | **<1e-4** | 同左 | −4.09 | **<1e-4** |
| vqaonly vs baseline（多任务迁移） | −0.28 | 0.71 | 同左 | −0.88 | 0.087 |
| with_od_gt_vqaonly vs with_od_gt（迁移·OD 格） | −0.17 | 0.83 | 同左 | −1.56 | **0.0018** |
| vqaonly vs with_od_gt_nopc（pc vs 框文本） | +0.68 | 0.28 | 同左 | +1.11 | **0.023** |

## 6. 判读

- **预登记 (a) 命中**：vqaonly ≈ baseline（sqa3d p=0.71、scanqa p=0.087 边缘）——多任务迁移对 sqa3d ≈0；scanqa 的 +0.88 边缘显著，OD 格的 −1.56 显著——**迁移小而真实，且只在 scanqa**
- **预登记 (d) 命中**：(with_od_gt_vqaonly, vqaonly) = +0.68/+0.28，均小于 293k 的 +0.80/+0.96 且不显著——293k 多任务混合没有稀释 OD 增益（两边都小），**OD 边际在有点云时本来就接近零**
- **与 OD7 冗余结论互证**：匹配 2×2 的两个边际都随另一因子在场缩水（对称替代），且四格里"全有"（54.96/24.88）对"只有点云"（54.28/24.60）的增益在噪声内——**对 VQA 而言，点云与 GT 框文本几乎可互换，叠加无额外收益**

## 7. 局限

- 匹配四格均 1 epoch LoRA、53k 数据——绝对值可能低于充分训练；但格间比较内部一致
- with_od_gt_vqaonly 的 od 块是 GT（18 类、无外观信息）——用更丰富的文本先验（含颜色/材质的 caption 化描述）能否替代点云的非空间桶优势未测
- scanqa 迁移效应（+0.9~1.6）与 pc 边际（+1.4~4.1）同向叠加于同一参照系（baseline），已在配对中分离但样本间不独立

## 8. 产物

- 脚本：`scripts/train/finetune-3d-llava-lora-{vqaonly,with-od-gt-vqaonly}.sh`、`scripts/eval/multigpu_eval_all_{vqaonly,with_od_gt_vqaonly}.sh`
- checkpoint：`checkpoints/finetune-3d-llava-lora-{vqaonly,with-od-gt-vqaonly}/`；预测+指标：`playground/predictions/` 同名
- 分析：两脚本第 9/10 run（`vqaonly`、`with_od_gt_vqaonly`，`runs_for` 过滤 + FULLCHECK_REF 真值回填）；日志 `logs/spatial_split_vqaonly.log`、`logs/spatial_split_paired_vqaonly.log`
