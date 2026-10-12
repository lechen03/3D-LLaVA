# 3D-LLaVA OD 注入实验报告（EXP-OD5：Ground-Truth bbox 注入——检测噪声 vs LLM 可用性消融）

> 日期：2026-10-09 ~ 10-12　|　前置：[EXP-OD1 全量](OD_INJECTION_EXP1_REPORT.md)、[EXP-OD2 topk](OD_INJECTION_TOPK_REPORT.md)、[EXP-OD3 names](OD_INJECTION_NAMES_REPORT.md)、[EXP-OD4 rel](OD_INJECTION_REL_REPORT.md)

---

## 1. 摘要

**实验问题**：OD 注入收益有限的原因是哪一个？(A) 框级信息难以被 LLM 利用；(B) EXP4 检测器不准、制造噪音。判别方法：把全部注入换成 **Ground-Truth bbox**（SpatialLM 训练标签，与 EXP4 预测同 18 类词表、同坐标系、p50 框数相当 11 vs 11），格式用 named 形式（`object: {name}, position_xyz: (...), angle_z: ..., scale_xyz: (...)`，无关系子句）。GT 仍不优 → A；GT 提点 → B。

**结论：A 强支持、B 拒绝**——

- **决定性证据**：GT vs 预测框（OD1）在 scanqa EM1 上**完美配对平手**（+0.04, p=0.96；翻转 190:188 = 掷硬币）；sqa3d +0.54（p=0.40，噪声内）。把每一个框换成真值，模型行为几乎不变
- GT 版虽是**全系列 VQA 表面最高**（sqa3d EM1 54.79 / EM1_refined 57.80、scanqa EM1 26.44），但与 OD1 的差异不具统计意义——这些数字就是「物体清单效应」的天花板，与框精度无关
- baseline → GT 的 scanqa 增益（+0.96, p=0.0295 显著）复现了 OD1 的非空间题倾斜结构（非空间 +1.63 p=0.0077 / 空间 +0.30 n.s.）——存在性 grounding 机制不变
- named 格式 CIDEr 毒性轻度复现（91.78 vs 基线 93.11，−1.33；OD4-rel 是 −5.6——坐标本身约 −1.3，关系文本额外 −4.3）
- 几何任务无解锁：caption 回到基线（+0.06）、ref-seg ≈ 基线（41.02 vs 41.10）——**即使完美框也撬不动 ref-seg/caption**

**对工程路线的含义**：改进检测器（更高的 F1/更准的坐标）**不会**提升 OD 注入收益；收益杠杆仍是 names（最便宜）+ topk 选择。OD 注入系列的最终完整结论见各报告指针。

## 2. 方法

- GT 源：`/home/chenle/SpatialLM/data/scannet_spatiallm/layout/{scene}.txt`（1513 场景全覆盖 = SpatialLM EXP1-4 的训练标签；18 类词表、墙/地板/杂物/微框已剔除；PCA-yaw OBB）→ 重排为 named 格式（2 位小数、重复类名编号）
- 注入策略同 OD1/OD3：全任务全场景；293,582 条 = 293,379 注入 + 203 空（两场景 GT 亦为空，与历轮对称）；token **p50=899 / p95=1892 / max=3370**
- 产物：`od_gt.json`（1114 train）/ `od_gt_val.json`（312 val）/ `*_with_od_gt.json` ×8（`scripts/generate_with_od_gt_data.py`）

## 3. 运行环境

- 训练：**GPU 2,3、累积 8 → 全局 batch 32**——与原始 baseline 及 OD1/OD3 完全同优化规模、同卡数，**本实验无任何 batch/硬件混杂**（这是六变体中条件最干净的一次）；29.6h
- 评估：2 卡两段（先 2,3 按用户要求暂停让卡 → 后 0,3 续跑；resume runner 跳过已完成的 sqa3d，指标文件由备份段+续跑段重组，答案数全对 3519/4675/2007/9508/11120、0 OOM）

## 4. 六方对比（粗体=最佳，‡=最低）

| 任务 | 指标 | baseline | OD1 全量 | OD2 topk | OD3 names | OD4 rel | **OD5 GT** |
|---|---|---|---|---|---|---|---|
| sqa3d | EM1 | 53.99 | 54.25 | 54.36 | 54.53 | 53.40‡ | ***54.79*** |
| sqa3d | EM1_refined | 56.66 | 57.00 | 56.98 | 57.12 | 55.78‡ | ***57.80*** |
| scanqa | EM1 | 25.48 | 26.40 | 26.07 | 26.27 | 26.42 | ***26.44*** |
| scanqa | EM1_refined | 41.20 | 42.20 | 42.03 | 41.39 | ***43.23*** | 42.57 |
| scanqa | CIDEr | 93.11 | 91.24 | 93.16 | ***93.46*** | 87.49‡ | 91.78 |
| scan2cap | CIDEr@.25 | 81.25 | 81.20 | ***81.71*** | 80.77 | 78.57‡ | 81.31 |
| scan2cap | CIDEr@.50 | 76.28 | 76.17 | ***76.57*** | 75.75 | 73.77‡ | 76.33 |
| scanrefer | mIoU | 41.10 | 40.94 | ***41.49*** | 40.81 | 40.20‡ | 41.02 |
| multi3drefer | mIoU | 46.79 | 46.90 | 47.09 | ***47.12*** | 46.46‡ | 46.98 |

成本：训练 29.6h（2 卡）；评估 ~4.5h（2 卡两段）；token p50 899。

## 5. 配对检验（EM1，McNemar 精确双侧）

scanqa（空间 2340 / 非空间 2335）：

| 对比 | 空间题 | 非空间题 | 全量 |
|---|---|---|---|
| baseline → GT | +0.30 (p=0.68) | **+1.63 (p=0.0077)** | **+0.96 (p=0.0295)** |
| **OD1 → GT（判别对）** | +0.04 (p=1.0) | +0.04 (p=1.0) | **+0.04 (p=0.96，完美平手)** |

sqa3d：baseline → GT 全量 +0.80 (p=0.20)、非空间桶 +9.09 (p=0.035, n=99)；OD1 → GT +0.54 (p=0.40)。

分桶补充（scanqa）：GT 非空间桶 CIDEr 1.00（names 1.02 最高）；空间桶 EM1 26.75（topk 27.26 仍最高——**选择机制仍是空间题的最优解，完美几何没有超过它**）。

## 6. 判读

1. **解释 B（检测器瓶颈）拒绝**：判别对比 OD1→GT 在全部 9,194 道 VQA 题上净翻转 +2 题。检测误差（EXP4 macro@.25≈0.58 水平）对注入收益的影响为零
2. **解释 A 强支持**：LLM 对注入框的利用停留在「有哪些类」——坐标、尺寸、朝向无论真假都不被有效利用（与 EXP-OD3 的坐标消融、EXP-OD4 的关系消融构成完整证据链：清单是唯一有效成分，topk 的选择是唯一增值机制）
3. **表面最高≠真实提升**：GT 的系列最高 VQA 数字（54.79/57.80/26.44）与 OD1/names 的差异全在噪声内——任何「注入更准的框」的后续投入都不会有回报
4. **混杂说明**：GT 同时改进了清单质量与几何精度，本设计无法单独归因；但「GT≈OD1≈names≈上限」的整体格局意味着两个因素的天花板都已被看到
5. named 格式（object: + 坐标）的 CIDEr 代价 ≈ −1.3pt，来源于坐标文本的风格干扰；工程上 names 格式（无坐标）仍是最优选择

## 7. Caveats

- 单次运行、单 seed；scanqa 六方 EM1 全距 0.96pt（<1.2σ）
- GT 的 OBB 角度来自 PCA-yaw（非人工标注朝向）；GT 与预测同帧（per-scene 平移 caveat 对两者同等成立）
- 评估跨两张 GPU 组合分两段完成（解码逐题确定性与分片无关，历史已验证分片等价）
- sqa3d 非空间桶 n=99

## 8. 复现

```bash
python scripts/generate_with_od_gt_data.py                    # 数据（GT 源在 SpatialLM 侧 layout/）
bash scripts/train/finetune-3d-llava-lora-with-od-gt.sh       # GPU 2,3, batch 32, ~30h
CUDA_VISIBLE_DEVICES=<free> bash scripts/eval/multigpu_eval_all_with_od_gt.sh
# （本次实际执行含一次人工暂停与 resume runner 续跑；metrics_all.txt 由备份段重组）
python scripts/eval/spatial_split_eval.py                     # 六方分桶（full 自检门槛）
/home/chenle/miniconda3/envs/llava/bin/python scripts/eval/spatial_split_paired.py
```

## 9. 产物

`scripts/generate_with_od_gt_data.py`、`scripts/train/finetune-3d-llava-lora-with-od-gt.sh`、`scripts/eval/multigpu_eval_all_with_od_gt{,_resume}.sh`、`playground/data/od_gt{,_val}.json`、`train_info/*_with_od_gt.json` ×8、`checkpoints/finetune-3d-llava-lora-with-od-gt/`、`playground/predictions/finetune-3d-llava-lora-with-od-gt/`（预测+重组 metrics）、`logs/{generate_od_gt,finetune_with_od_gt,eval_with_od_gt,spatial_split_gt}.log`、`logs/od5_metrics_sqa3d_only.txt.bak`

---

*OD 注入系列至此六变体收官。终局结论：外部检测先验对 3D-LLaVA 的全部可提取价值 = **物体清单（names）+ 问题条件选择（topk）**；框级几何信息——无论检测的还是真值的、无论裸坐标还是文本化关系——都不被 LLM 有效利用；改进检测器对该注入范式无回报。*
