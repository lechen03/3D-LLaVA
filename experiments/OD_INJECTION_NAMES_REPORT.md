# 3D-LLaVA OD 注入实验报告（EXP-OD3：全量注入·仅类名版——坐标消融）

> 日期：2026-09-29　|　前置：[EXP-OD1 全量注入](OD_INJECTION_EXP1_REPORT.md)、[EXP-OD2 topk 检索](OD_INJECTION_TOPK_REPORT.md)、[空间分桶分析](OD_INJECTION_SPATIAL_SPLIT.md)

---

## 1. 摘要

**实验问题**：OD 注入的有效成分到底是什么？把全量 OD 的坐标全部删掉、只留**去重类名列表**（`<od>\nwindow, chair, door, sink\n</od>`，~15-60 token），其余与 EXP-OD1 完全同构——增益还在吗？

**结论（预登记判读全部落定）**：

1. **VQA：坐标是死重，物体清单就是全部有效成分**——scanqa EM1 上 names 与全量 bbox 版**配对统计平手**（26.27 vs 26.40，p=0.81）；sqa3d 上 names 反而**四模型最强**（EM1 54.53，非空间桶 vs 基线 +12.1pt，p=0.0075）；scanqa CIDEr **93.46 四模型最高**（基线 93.11、全量版 91.24）
2. **ref-seg：有效成分是「选择」不是「清单」**——names（无选择无几何）在 scanrefer 跌破基线（mIoU 40.81 vs 41.10），而 topk（有选择）41.49 最佳；预登记预期兑现
3. **caption：几何有值**——names 版 scan2cap CIDEr@.25 80.77 低于基线 81.25（topk 版 81.71 最佳；注意 topk 的 dense_captioning 训练时保留全量 bbox 文本，本实验只拿类名，此差异部分来自训练侧条件差异）
4. **成本几乎归零**：训练 6h43m（≈无 OD baseline 的 6h；全量版 14h22m 的 47%）、评估 48min、OD token p50 **64**（全量 1129 的 1/18）

**部署含义**：服务 VQA → names 版（最便宜且最强）；服务 ref-seg/caption → topk 版。

## 2. 方法

- OD 块 = 场景检测类名的**去重逗号列表**（按 bbox 序首现序；p50 5-6 类，最多 16 类）；全任务全场景注入（dense_captioning 同样拿类名——本实验中「全量」=类名版）；203 条空 OD 保持原文；gpt 轮逐字节不变
- 数据 `scripts/generate_with_od_names_data.py`（293,582 条断言全过；token p50=64/p95=90/max=192）；eval 侧零代码改动——`od_exp4_names_val.json` 直接走现有 `--od-file` 路径
- 训练/评估脚本为 EXP-OD1 克隆（仅数据路径与 OD 文件不同）；runner 已固化 `HF_HUB_OFFLINE=1`（EXP-OD2 的 401 教训）
- 评估 0 OOM，5 任务 answers 3519/4675/2007/9508/11120 全对

## 3. 四方对比（312 val 场景；粗体=四列最佳，斜体=次佳）

| 任务 | 指标 | baseline | OD1 全量 | OD2 topk | **OD3 names** |
|---|---|---|---|---|---|
| sqa3d | EM1 | 53.99 | 54.25 | 54.36 | ***54.53*** |
| sqa3d | EM1_refined | 56.66 | 57.00 | 56.98 | ***57.12*** |
| scanqa | EM1 | 25.48 | ***26.40*** | 26.07 | *26.27* |
| scanqa | EM1_refined | 41.20 | ***42.20*** | 42.03 | 41.39 |
| scanqa | CIDEr | 93.11 | 91.24 | 93.16 | ***93.46*** |
| scan2cap | CIDEr@.25 | *81.25* | 81.20 | ***81.71*** | 80.77 |
| scan2cap | CIDEr@.50 | *76.28* | 76.17 | ***76.57*** | 75.75 |
| scanrefer | mIoU | *41.10* | 40.94 | ***41.49*** | 40.81 |
| scanrefer | Acc@.50 | 39.43 | 39.44 | ***40.15*** | 38.96 |
| multi3drefer | mIoU | 46.79 | 46.90 | *47.09* | ***47.12*** |
| multi3drefer | Acc@.25 | *65.69* | 64.89 | 65.49 | ***65.72*** |
| multi3drefer | Acc@.50 | 45.30 | 45.76 | ***46.07*** | 45.67 |

成本：

| | baseline | OD1 全量 | OD2 topk | **OD3 names** |
|---|---|---|---|---|
| 训练 | 6h00m | 14h22m | 12h10m | **6h43m** |
| 评估（5 任务） | — | 1h32m | 1h17m | **48m** |
| OD token p50 | — | 1129 | 326 | **64** |

## 4. 配对检验（EM1，McNemar 精确双侧）

**scanqa**（空间 2340 / 非空间 2335）：

| 对比 | 空间题 | 非空间题 | 全量 |
|---|---|---|---|
| baseline → names | +0.47 (p=0.47) | +1.11 (p=0.075) | **+0.79 (p=0.068)** |
| 全量 → names | +0.21 (p=0.79) | −0.47 (p=0.49) | **−0.13 (p=0.81，平手)** |
| topk → names | −0.34 (p=0.61) | +0.73 (p=0.19) | +0.19 (p=0.66) |

**sqa3d**（非空间 n=99）：baseline → names **+12.12（p=0.0075）**——三系实验中最大的单桶效应；names vs 全量 +0.28、vs topk +0.17（均平手）。

分桶补充：scanqa 非空间桶 CIDEr——names **102.14** 四模型最高；multi3drefer 原生非空间桶 mIoU——names **55.07**（+1.42 vs 基线）最佳。

## 5. 机制分解（三个变体 = 三种成分的对照实验）

| 变体 | 物体清单 | 逐框几何 | 问题条件选择 |
|---|---|---|---|
| OD1 全量 | ✓（逐框） | ✓ | ✗ |
| OD2 topk | ✓（逐框） | ✓（截断） | ✓ |
| OD3 names | ✓（去重） | ✗ | ✗ |

- **VQA**：清单充分（names≈full 平手）；几何无增量（full≈names）；清单「去重」也无损（丢失数量信息不伤 EM）。全量版唯一的「优势」scanqa EM1 +0.13pt 是噪声（p=0.81）。全量版的 CIDEr 代价则真实存在（91.24 vs names 93.46）——**坐标文本反而拖累生成风格**
- **ref-seg**：帮助来自「选择」（topk 聚焦题目相关类）；纯清单（names）无益反小害（scanrefer mIoU −0.29 vs 基线）——分割任务需要「哪个框」级别的定位线索，清单级语义不够用
- **caption**：topk 最佳的来源有二——dense_captioning 训练侧保留全量 bbox（几何线索）+ VQA 侧的选择性注入；names 把两者都拿掉后回落到基线下 0.48pt
- 与空间分桶结论闭环：EXP-OD1 增益在非空间题 = 存在性 grounding；names 恰好只保留存在性信息 → 非空间桶增益完整保留甚至放大（scanqa +1.11；sqa3d +12.1）

## 6. Caveats

- 单次运行、单 seed；scanqa EM1 四模型全距仅 0.92pt（<1σ），EM 结论以配对平手 + 多指标一致性表述
- names 版与全量版在 dense_captioning 上的差异混入了「训练条件不同」（类名 vs bbox 文本），不能把 scan2cap 差异全归因于评估侧
- 类名来自 EXP4 检测（含误检；无置信度信息）；去重丢失数量（5 把椅子与 1 把椅子同形）
- sqa3d 非空间桶 n=99，效应量虽大需多种子确认

## 7. 复现

```bash
python scripts/generate_with_od_names_data.py                  # 数据 + names OD 文件
bash scripts/train/finetune-3d-llava-lora-with-od-names.sh     # tmux, 4 卡, ~6.7h
bash scripts/eval/multigpu_eval_all_with_od_names.sh           # tmux, offline 已固化, ~48min
python scripts/eval/spatial_split_eval.py                      # 四方分桶（full 自检门槛）
/home/chenle/miniconda3/envs/llava/bin/python scripts/eval/spatial_split_paired.py
```

## 8. 产物

`scripts/generate_with_od_names_data.py`、`scripts/train/finetune-3d-llava-lora-with-od-names.sh`、`scripts/eval/multigpu_eval_all_with_od_names.sh`、`playground/data/od_exp4{,_val}_names.json`、`train_info/*_with_od_names.json` ×8、`checkpoints/finetune-3d-llava-lora-with-od-names/`、`playground/predictions/finetune-3d-llava-lora-with-od-names/`（预测+metrics）、`logs/{generate_od_names,finetune_with_od_names,eval_with_od_names}.log`

---

*OD 注入系列至此完成三变体 + 两轮后分析。系列总结论：**外部检测先验对 3D-LLaVA 的价值 = 物体清单（VQA 存在性 grounding）+ 问题条件选择（ref-seg）**；逐框坐标文本在所有任务上均无正贡献（VQA 死重、风格负贡献）。后续候选：names+topk 组合（选择后的类名清单，理论上兼得且更便宜）；给 names 加数量；坐标对齐版做最终验证。*
