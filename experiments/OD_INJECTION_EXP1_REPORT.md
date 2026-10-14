# 3D-LLaVA OD 注入实验报告（EXP-OD1：EXP4 检测先验注入微调）

> 日期：2026-09-21 ~ 09-22　|　本仓库：3D-LLaVA（OD 注入管线代码与数据所在）
> 上游：SpatialLM EXP4 模型（`saves/scannet_exp4/checkpoint-2500`，18 类检测 macro@.25 = 0.5825）

---

## 1. 摘要

**实验问题**：给 3D-LLaVA（点云+LLM：VQA / dense caption / referring segmentation）的每条训练对话注入一份外部 3D 物体检测结构化文本（来自 SpatialLM EXP4），能否提升其多任务表现？

**做法**：human 轮 `<pc>\n` 后插入 `<od>\n{检测文本}\n</od>\n`（纯文本、loss-mask、仅作条件），单变量对照微调（唯一差异 = 8 个训练 JSON 换 `_with_od` 版）。

**结果**：**VQA 温和受益（scanqa EM1 +0.92pt / EM1_refined +1.01pt，sqa3d +0.3pt），dense caption 与 referring segmentation 持平（±0.8pt 内）**；scanqa 呈 EM↑/CIDEr↓ 的风格 trade-off。代价：训练时长 ×2.3（14.3h vs 6h，OD 使序列 ~100→p50 1129 token）、推理 prompt ~10×。

## 2. 数据制备

| 步骤 | 内容 | 产物 |
|---|---|---|
| OD 生成（train） | 8 个原始 train_info JSON 的场景并集 **1114 个**（全在我们 train split、无 val 泄漏），EXP4 ckpt-2500 + seed 42，4 卡 45 分钟 | `playground/data/od_exp4.json`（14,152 框，恰好 18 类，2 空场景） |
| OD 生成（val） | 312 个 val 场景用现成 `pred_val_exp4_best/` | `playground/data/od_exp4_val.json`（312/312 非空，中位 11 框/场景） |
| 注入 | `<pc>\n` 后插 `<od>\n…\n</od>\n`；坐标舍入 **2 位小数**；293,582 条中 293,440 带注入、203 条（空检测场景）保持原文；gpt 轮与其余字段逐字节不变 | `train_info/*_with_od.json` ×8（`scripts/generate_with_od_data.py`） |

**可行性论证**（探索确认）：`<pc>`→`<image>\n` 重锚保留注入块；`model_max_length=4096` 右侧截断不伤开头；OD 块在 human 轮 → `IGNORE_INDEX`（只做条件不监督）；token 审计 93k 条实测 **max 2696 < 4096**（2 位小数 ~47 token/行，全精度 ~41 行即爆——舍入必要）。

## 3. 训练

- 配置：`scripts/train/finetune-3d-llava-lora-with-od.sh`，与原脚本唯一差异 = data_path；**4 卡 + 梯度累积 8→4**（全局 batch 保持 32，与 2 卡基线同优化规模，纯并行加速）
- 9408 步 / 1 epoch / **14.3 小时**（基线 2 卡 6h；OD 使每步 2.3s→5.5s，4 卡抢回 ~1.65×）
- loss 5.40 → 末段均值 0.716，正常收敛
- 产物：`checkpoints/finetune-3d-llava-lora-with-od/`

## 4. 评估

### 4.1 注入实现（eval 侧）

5 个 `llava/eval/model_*.py` 各加 `--od-file` 参数（不传则行为与原版逐字节一致）+ prompt 注入一行；runner `scripts/eval/multigpu_eval_all_with_od.sh`（4 卡 5 任务，每任务后自动跑指标并记录答案数自检）。输出 prompt 经抽查与训练格式逐字节一致。

### 4.2 事故与修复：eval OOM 杀分片

首轮评估 10 次 OOM **杀死分片进程**（sqa3d 只剩 452/3519、scanqa 642/4675，指标作废）。根因：长 OD prompt 的 KV cache + beam=5 逼近上限后的**显存碎片化**（报错特征：只需 40–60MiB 但 1.4–1.8GB reserved-unallocated）。修复：① 逐题 `torch.cuda.empty_cache()`；② generate 包 OOM 重试环（清缓存原参重试 → 降 beam=1 重试）；③ 答案数自检。重跑 **0 OOM、5 任务 100% 完成**。

### 4.3 结果（312 val 场景，与基线同解码设置）

| 任务 | 指标 | 基线 | with-od | Δ |
|---|---|---|---|---|
| sqa3d (VQA) | EM1 / EM1_refined | 0.5399 / 0.5666 | **0.5425 / 0.5700** | +0.26 / +0.34pt |
| scanqa (VQA) | EM1 / EM1_refined | 0.2548 / 0.4120 | **0.2640 / 0.4220** | **+0.92 / +1.01pt** |
| scanqa | CIDEr / Bleu_1 | **0.9311 / 0.4569** | 0.9124 / 0.4432 | −1.87 / −1.37pt |
| scan2cap (caption) | CIDEr@0.25 / @0.50 | 0.8125 / 0.7628 | 0.8120 / 0.7617 | 持平 |
| scanrefer (ref seg) | mIoU / Acc@.25 / @.50 | 0.4110 / 0.5971 / 0.3943 | 0.4094 / 0.5898 / 0.3944 | −0.16 / −0.73 / +0.01pt |
| multi3drefer (ref seg) | mIoU / Acc@.25 / @.50 | 0.4679 / 0.6569 / 0.4530 | 0.4690 / 0.6489 / 0.4576 | +0.11 / −0.80 / +0.46pt |

### 4.4 解读

1. **VQA 受益方向一致**：scanqa ~43/4675 题翻转（+0.92pt EM1）；sqa3d +0.26pt 属噪声边缘。检测先验对"物体是什么/在哪"类问题有真实但轻微的帮助
2. **EM↑ / CIDEr↓ trade-off**：答案更常精确命中，但措辞与参考答案相似度略降——疑似 bbox 文本风格让回答更"名词化"
3. **Caption / ref-seg 持平**：描述与分割任务本就以点云特征为主；ref-seg 的 OD 坐标与其输入帧完全不对齐（见 caveat），持平符合预期
4. 成本收益：+1pt scanqa EM 的代价是训练 ×2.4 时长、推理 prompt ×10 token——若只服务 VQA 场景可考虑任务选择性注入

## 5. Caveat

- **坐标未对齐**（已决策保留）：OD 坐标 = axis-align 旋转 + min 重锚；3D-LLaVA 的 vqa/caption 帧与之差每场景常量平移（公式 `their_x = our_x − extent_x/2`，同 y，z 一致），refer_seg 帧完全不同且带随机旋转。LLM 只见点云 token 不见坐标，起作用的是类别/尺寸/相对布局（frame-invariant）
- OD 来自 EXP4 在 **train 场景**上的输出（模型见过，质量高于 val 口径）；val 侧用真 val 预测，两侧质量不对称
- 单次运行、未做多种子重复；显著性未做检验（scanqa +0.92pt 量级 ~1.6σ 边缘）
- beam 降级重试在本轮 0 次触发，未引入解码偏差

## 6. 复现命令

```bash
# 数据（在本仓库根目录下）
python scripts/generate_with_od_data.py                       # train 注入
# od_exp4_val.json 由 SpatialLM 侧 data/scannet_spatiallm/pred_val_exp4_best/ 舍入生成

# 训练（tmux）
bash scripts/train/finetune-3d-llava-lora-with-od.sh          # 4 卡，~14h

# 评估（tmux；含 OOM 装甲与指标汇总）
bash scripts/eval/multigpu_eval_all_with_od.sh
# 基线指标（CPU 秒级，用现成预测）：见 playground/predictions/baseline_metrics/

## 7. 产物

| 路径 | 内容 |
|---|---|
| `playground/data/od_exp4{,_val}.json` | train 1114 / val 312 场景 OD 文本 |
| `playground/data/train_info/*_with_od.json` | 注入后训练数据 ×8 |
| `scripts/generate_with_od_data.py` | 注入脚本 |
| `scripts/train/finetune-3d-llava-lora-with-od.sh` | 训练脚本 |
| `llava/eval/model_*.py`（5 个） | +--od-file 参数 + OOM 重试装甲 |
| `scripts/eval/multigpu_eval_all_with_od.sh` | 评估 runner |
| `checkpoints/finetune-3d-llava-lora-with-od/` | 训练模型 |
| `playground/predictions/finetune-3d-llava-lora-with-od/` | 预测 + metrics_all.txt |
| `playground/predictions/baseline_metrics/` | 基线 5 任务指标 |
| `logs/{finetune_with_od,eval_with_od}.log` | 训练/评估日志 |

---

*后续候选（未做）：坐标对齐版 OD ablation（公式已备）；按任务选择性注入（仅 VQA）；OD 截断至与问题相关的 top-K 框以压缩 prompt；多种子重复验证 scanqa 增益显著性。*

*已完成后续分析：[空间关系问题分桶](OD_INJECTION_SPATIAL_SPLIT.md)（2026-09-28）——scanqa EM1 全量增益 +0.92pt 配对显著（McNemar p=0.037），但**全部来自非空间关系问题**（+1.58pt, p=0.010；空间题 +0.26pt, p=0.73）→ OD 先验是存在性/类别 grounding，不是空间关系推理。*

*已完成后续实验：[EXP-OD2 选择性注入](OD_INJECTION_TOPK_REPORT.md)（2026-09-29）——RAG 类级 top-3 检索注入（OD 文本 ÷3.4、训练 −15%）：13 项主指标 10 项三列最佳，**EM↑/CIDEr↓ trade-off 消除**，ref-seg/caption 首次全面超基线；增益分布与全量版互补（topk 偏空间题、全量偏非空间题）。*

*已完成后续实验：[EXP-OD3 仅类名版](OD_INJECTION_NAMES_REPORT.md)（2026-09-29）——坐标消融：OD 块只留去重类名（token p50 1129→64、训练 14.4h→6.7h）。**VQA 上与全量版配对平手（p=0.81）且 sqa3d 四模型最强、CIDEr 最高——坐标是死重、物体清单即全部有效成分**；ref-seg 跌回基线下（有效成分是 topk 的「选择」）；caption 略降（几何对 dense caption 有值）。*

*已完成后续实验：[EXP-OD4 显式空间关系版](OD_INJECTION_REL_REPORT.md)（2026-10-07）——纯几何推断的观察者视角成对关系文本化：**H1 拒绝**（空间题未抬升，scanqa 空间桶 +0.64 n.s.、sqa3d 反跌 −0.6）、**冗长关系文本有毒**（CIDEr −5.6pt、caption/ref-seg 全线最低、sqa3d 跌破基线）；唯一亮点 scanqa EM1/EM1_refined 双最高但伴随严重风格极化。**系列终局：OD 先验价值上限 = 物体清单 + 检索选择，任何更丰富的几何/关系文本化均零或负贡献。***

*已完成后续实验：[EXP-OD5 Ground-Truth bbox 版](OD_INJECTION_GT_REPORT.md)（2026-10-12）——判别消融定论：GT 与预测框在 scanqa EM1 完美配对平手（+0.04, p=0.96）→ **检测器精度不是瓶颈、框级信息难以被 LLM 利用**；GT 的系列最高 VQA 数字（sqa3d 54.79）即清单效应天花板。系列六变体收官：价值 = names 清单 + topk 选择，改进检测器对此范式无回报。*

*已完成后续实验：[EXP-OD6 去点云版](OD_INJECTION_NOPC_REPORT.md)（2026-10-14）——反向消融（OD5 去掉点云，仅 GT bbox 文本 + 问题，代码级 `--no-pc`）：**sqa3d 与完整多模态 baseline 配对平手（−0.40, p=0.61）——GT 框列表 + 语言先验可打平整个 3D 编码器**；scanqa −2~3 EM 显著、CIDEr −9.6。掉点类型化集中：非空间/颜色外观题（RGB 只在点云里）与 What 识别/Can-see 可见性题崩、How-many 计数题保住 OD 注入全部增益（+5.4 vs baseline，框列表可数）。ref-seg/scan2cap 无点云即任务不成立。*
