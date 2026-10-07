# 3D-LLaVA OD 注入实验报告（EXP-OD4：显式空间关系版——几何推断文本化）

> 日期：2026-10-05 ~ 10-07　|　前置：[EXP-OD1 全量](OD_INJECTION_EXP1_REPORT.md)、[EXP-OD2 topk](OD_INJECTION_TOPK_REPORT.md)、[EXP-OD3 names](OD_INJECTION_NAMES_REPORT.md)、[空间分桶](OD_INJECTION_SPATIAL_SPLIT.md)

---

## 1. 摘要

**实验问题**：前序实验证明 OD 注入的增益始终不在空间关系题上（LLM 需从裸坐标自行推关系，且帧不对齐）。若用**纯几何脚本**把成对空间关系显式写进 OD 文本（观察者立于场景中心、看向被指向物体，三轴关系：left/right、in front of/behind of、above of/below of），能否撬动空间题增益？

**结论（预登记判读落定）**：

- **H1 拒绝**：空间题增益未抬升——scanqa 空间桶 +0.64（p=0.39，不如 topk 的 +0.81）；sqa3d 空间桶反而 **−0.85**
- **H2 成立且加强**：文本化关系无法替代几何对齐，且冗长关系文本**有毒性**——sqa3d 全线跌破基线（EM1 53.40 vs 基线 53.99，五模型最低）、scanqa CIDEr **−5.6pt**（87.49）、caption −2.7pt（78.57）、ref-seg 两任务双最低
- **H3 部分成立**：scanqa 非空间桶增益保留（+1.24, p=0.082；结构与全量版同构——增益本质仍是存在性 grounding）
- **H4 拒绝**：caption 未受益反而最差（关系文本与 caption 语言同构的假设不成立——模型模仿了关系文本的**风格**而非利用其**内容**）

唯一亮点：scanqa EM1 26.42 / EM1_refined 43.23 均为五模型最高（vs 基线 +0.94, p=0.061 / +2.03），但同场 Bleu_1 42.15 最低、CIDEr 大跌——**风格极化**：答案变长变详（宽容匹配受益、精确生成指标受害），与 EXP-OD1 的 EM↑/CIDEr↓ trade-off 同源但严重得多。

**系列终局定论（五变体 + 两轮后分析）**：OD 注入的全部价值 = **物体清单（names 已证）+ 问题条件选择（topk 已证）**；逐框坐标（EXP-OD3）与文本化关系（本实验）均为零或负贡献。部署矩阵不变：VQA→names、ref-seg/caption→topk。

## 2. 方法

- 格式（用户 spec）：`object: painting, position_xyz: (...), angle_z: ..., scale_xyz: (...), spatial relationship: above of table 1, behind of table 2, ...`；重复类名编号（table 1/2）
- 推断（`scripts/od_spatial_relations.py`，纯几何零 LLM）：观察者 = 检测框中心 xy 包围盒中心；**关系锚定被指向物体 B 的视线**（"A left of B" = 观察 B 时 A 在左；互易对自洽 A left of B ⇔ B right of A）；三轴死区：横向 0.15m / 深度 0.25m / 高度 0.15m
- 覆盖：每对象 **K=6 近邻** + 确定性降级链 (6,4,3,2,1,0)，字符预算保险丝（真实 tokenizer 标定：tokens≈chars×0.40；预算 6200 字符）；K=0 终档保巨型场景（53 框 base 行本身超限）
- 数据：293,582 条 = 293,379 注入 + 203 空；token **p50=2033 / p95=2618 / max=3598**、含系统提示 **0 截断**（逐项校验过 810→0）
- 合成算例断言（方向 + 互易）全过；真实场景目检合理

## 3. 运行环境偏差（必须注明）

| 项 | 本实验 | 前序实验 |
|---|---|---|
| 训练 GPU | **3 卡（1,2,3）**——GPU 0 先后被他人任务（shangquan 8.5h+ / liuquan）占用 | 4 卡 |
| 全局 batch | **24**（2×3×4） | 32 |
| 训练时长 | **34.8h**（token ×1.8 且少一卡） | 6h~14.4h |
| 评估 | 3 卡分片 3.5h | 48m~1.5h |

batch 24 是一个混杂因素（优化规模 −25%），但本实验的退化模式（风格指标崩、sqa3d 跌破基线）幅度大且跨 5 任务一致，无法仅用 batch 偏差解释；scanqa EM1（+0.94, p=0.061）与全量版（batch 32, +0.92, p=0.037）几乎相同也说明优化规模影响有限。

## 4. 五方对比（粗体=最佳，‡=最低）

| 任务 | 指标 | baseline | OD1 全量 | OD2 topk | OD3 names | **OD4 rel** |
|---|---|---|---|---|---|---|
| sqa3d | EM1 | 53.99 | 54.25 | 54.36 | **54.53** | 53.40‡ |
| sqa3d | EM1_refined | 56.66 | 57.00 | 56.98 | **57.12** | 55.78‡ |
| scanqa | EM1 | 25.48 | 26.40 | 26.07 | 26.27 | **26.42** |
| scanqa | EM1_refined | 41.20 | 42.20 | 42.03 | 41.39 | **43.23** |
| scanqa | CIDEr | 93.11 | 91.24 | 93.16 | **93.46** | 87.49‡ |
| scan2cap | CIDEr@.25 | 81.25 | 81.20 | **81.71** | 80.77 | 78.57‡ |
| scanrefer | mIoU | 41.10 | 40.94 | **41.49** | 40.81 | 40.20‡ |
| multi3drefer | mIoU | 46.79 | 46.90 | 47.09 | **47.12** | 46.46‡ |

## 5. 配对检验与分桶（EM1，McNemar 精确双侧）

scanqa（空间 2340 / 非空间 2335）：

| 对比 | 空间题 | 非空间题 | 全量 |
|---|---|---|---|
| baseline → rel | +0.64 (p=0.39) | **+1.24 (p=0.082)** | +0.94 (p=0.061) |
| 全量 → rel | +0.38 (p=0.61) | −0.34 (p=0.65) | +0.02 (p=1.0，平手) |

sqa3d：baseline → rel 全量 **−0.60**（空间桶 −0.85/非空间 +8.08，n=99）——五模型中唯一总量为负。

scanqa 分桶 CIDEr：rel 非空间桶 92（其余四模型 100-102）——CIDEr 崩溃两桶皆有、非空间更重。

## 6. 解读

1. **为什么显式关系没用**：空间关系题的答案取决于**点云坐标系**里的真实方位（3D-LLaVA 的 vqa 帧与我们 OD 帧差每场景常量平移、观察者位置是虚构假设）——文本里的 "left of" 以虚构观察者为参照，与题目所需的参照系不匹配；且 topk 已隐式证明模型不需要显式关系（选择机制即可温和抬升空间桶）
2. **为什么变毒**：p50 2033 token 的关系化文本占据了训练信号的主导（风格模仿：EM1_refined ↑、Bleu/CIDEr ↓），挤占了任务本身的学习；与 EXP-OD1 的 CIDEr trade-off 同源、剂量放大
3. **与系列结论闭环**：存在性清单（names）是唯一有效成分；任何「更丰富的几何/关系文本」尝试（全量坐标、显式关系）都零或负收益——**外部检测先验对 3D-LLaVA 的价值上限就是语义清单 + 检索选择**

## 7. Caveats

- batch 24 / 3 卡偏差（见 §3）；单次运行单 seed
- 关系推断的观察者是假设（场景中心、无朝向先验）；死区/K 参数未消融
- 关系仅 K=6 近邻覆盖（超大场景 K=0）——「全对关系」在 token 预算下不可行，结论限于近邻关系文本化
- scanqa EM1 +0.94（p=0.061）与全量版统计上无法区分（互比 p=1.0）

## 8. 复现

```bash
python scripts/od_spatial_relations.py                                # 自测（合成断言 + 真实场景目检）
python scripts/generate_with_od_rel_data.py                           # 数据（含标定审计）
bash scripts/train/finetune-3d-llava-lora-with-od-rel.sh              # 3 卡 batch 24, ~35h
CUDA_VISIBLE_DEVICES=1,2,3 bash scripts/eval/multigpu_eval_all_with_od_rel.sh   # runner 已尊重继承 GPU 集
python scripts/eval/spatial_split_eval.py                             # 五方分桶（full 自检门槛）
/home/chenle/miniconda3/envs/llava/bin/python scripts/eval/spatial_split_paired.py
```

## 9. 产物

`scripts/od_spatial_relations.py`（推断模块）、`generate_with_od_rel_data.py`、`finetune-3d-llava-lora-with-od-rel.sh`、`multigpu_eval_all_with_od_rel.sh`（GPU 集已参数化）、`od_exp4{,_val}_rel.json`、`*_with_od_rel.json` ×8、checkpoint、预测+metrics、五方分桶数据（`spatial_split/parsed_metrics.json`）、日志 `logs/{generate_od_rel,finetune_with_od_rel,eval_with_od_rel,spatial_split_rel}.log`

---

*事故记录：①首启训练 OOM——GPU 0 被他人任务占用（改 3 卡 batch 24）；②首启评估分片缺失——runner 内写死的 `export CUDA_VISIBLE_DEVICES=0,1,2,3` 覆盖继承环境，GPU-0 分片与他人任务撞车（改参数化并固化教训）。*
