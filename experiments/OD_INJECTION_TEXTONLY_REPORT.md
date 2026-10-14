# 3D-LLaVA OD 注入实验报告（EXP-OD7：仅问题文本——语言先验地板，2×2 因子收官）

> 日期：2026-10-14　|　前置：[EXP-OD1 全量](OD_INJECTION_EXP1_REPORT.md)、[OD2 topk](OD_INJECTION_TOPK_REPORT.md)、[OD3 names](OD_INJECTION_NAMES_REPORT.md)、[OD4 rel](OD_INJECTION_REL_REPORT.md)、[OD5 GT](OD_INJECTION_GT_REPORT.md)、[OD6 去点云](OD_INJECTION_NOPC_REPORT.md)

---

## 1. 摘要

**实验问题**：OD6（GT bbox 文本、无点云）已证明框列表+语言先验 ≈ 完整多模态 baseline（sqa3d p=0.61）。那语言先验**本身**值多少？OD 文本在无点云时的**因果**贡献又是多少？本实验把输入剥到只剩问题（无点云、无 OD），补全 2×2 因子设计的最后一格：

| EM1 | + GT OD | − OD |
|---|---|---|
| **+ 点云** | OD5：sqa3d 54.79 / scanqa 26.44 | baseline：53.99 / 25.48 |
| **− 点云** | OD6：53.59 / 23.49 | **OD7：51.49 / 20.51** |

**结论：语言先验是真实地板而非全部；OD 文本在无点云时因果贡献显著且类型化集中——**

- **OD7 vs OD6**（全系列最干净对比：同 2 文件、同 53k、同混合，唯一差异 = od 块）：sqa3d **+2.10**（p=0.0005）、scanqa **+2.98**（p<1e-4）——OD 文本因果价值显著
- **OD7 vs baseline**（点云总贡献，无 OD 侧）：sqa3d **−2.50**（p=0.0011）、scanqa **−4.96**（p<1e-4）——点云在没有任何文本辅助时贡献最大
- **因子替代结构**：每个因子的效应在另一因子缺席时更大（OD 效应：有点云 +0.80 vs 无点云 +2.10/scanqa +0.96 vs +2.98；点云效应：有 OD −1.20/−2.95 vs 无 OD −2.50/−4.96）——**点云与 OD 文本是高冗余的信息编码**，各含对方的大部分有效信息（几何/存在性）
- **题型分解（预注册验证）**：How-many 计数题 OD6−OD7 = **+5.59**（54.41→48.82，跌回 baseline 49.03 之下）——OD6 的计数增益 **100% 来自数框列表**；预注册的"Is 是否题不降"**预测失手**（65.64，−2.61 vs OD6）：存在性问题（"is there a X"）确实需要某种场景证据（清单或点云任一）；唯一超 baseline 的题型是 Which 辨别（48.15 vs 45.87，纯语言区分）
- **CIDEr 阶梯单调**：93.11（baseline）→ 91.78（OD5）→ 83.48（OD6）→ **77.75（OD7）**——开放式长答案的流畅性同时依赖两种信息源

**对工程路线的含义**：纯语言先验地板 = sqa3d 51.5 / scanqa 20.5（微调后）。任何 3D 管线的净贡献上限 = 距此地板的差值；检测器→文本清单路线实际交付了其中一半以上（无点云时 +2~3），且是唯一可解释的成分（可数的框）。

## 2. 方法

- **数据**：`*_textonly.json` ×2 = OD6 `_nopc` 文件剥掉开头 `<od>\n…\n</od>\n` 块；**全量对拍审计**（非抽样）：OD7 human 轮 == 原始 `_3d_llava.json` 轮去 `<pc>\n` 前缀，逐字节相等；53,138 条（scanqa 26,515 + sqa3d 26,623；15 条 scene0154_00 空 GT 本就无块，原样通过）；token p50=32 / max=100
- **零代码改动**：OD6 的 `--no-pc`/`no_pc` 基础设施原样复用；eval **不传 `--od-file`** → od_map 空 → 不注入 → prompt = 问题+后缀，与训练格式逐字节同构
- 训练/评估仅覆盖 2 个 VQA 任务（其余 6 文件 `[SEG].`/`<loc>` 依赖点云，OD6 已定论；ref-seg/scan2cap 无点云即任务不成立）

## 3. 运行环境

- 训练：**GPU 0,3**（1,2 他人占用）、全局 batch 32 = 全系列同规模、**54 分钟**（纯问题序列极短，1.92s/it vs OD6 9.2s/it）
- 评估：GPU 0,3 两分片（生成 4-5.7 it/s，比 OD6 快一倍）；答案数 3519/4675 全对、0 OOM、ALL_EVAL_DONE

## 4. 2×2 因子表（EM1；括号 = 与左/上邻格的配对差）

| 指标 | OD7（全无） | OD6（+OD） | baseline（+pc） | OD5（+pc+OD） |
|---|---|---|---|---|
| sqa3d EM1 | 51.49 | 53.59 (+2.10, p=0.0005) | 53.99 (+2.50, p=0.0011) | 54.79 (+0.80 vs base) |
| sqa3d EM1_refined | 53.99 | 56.15 | 56.66 | 57.80 |
| scanqa EM1 | 20.51 | 23.49 (+2.98, p<1e-4) | 25.48 (+4.96, p<1e-4) | 26.44 (+0.96, p=0.0295) |
| scanqa EM1_refined | 34.05 | 38.91 | 41.20 | 42.57 |
| scanqa CIDEr | 77.75 | 83.48 | 93.11 | 91.78 |

**读法**：边际效应随另一因子在场而缩水——信息冗余。OD 文本与点云各自单独的贡献（+2.1/+2.5 与 +3.0/+5.0）都大于两者叠加的净增益（OD5 vs baseline 仅 +0.8/+1.0）。

## 5. 配对检验（EM1，McNemar 精确；全为 OD7 为 base 侧）

| 对比（sqa3d） | Δ | p | 对比（scanqa） | Δ | p |
|---|---|---|---|---|---|
| OD7 vs baseline | −2.50 | **0.0011** | OD7 vs baseline | −4.96 | **<1e-4** |
| OD7 vs OD6 | −2.10 | **0.0005** | OD7 vs OD6 | −2.98 | **<1e-4** |
| OD7 vs OD5 | −3.30 | **<1e-4** | OD7 vs OD5 | −5.93 | **<1e-4** |
| OD7 vs names | −3.04 | **0.0001** | OD7 vs names | −5.75 | **<1e-4** |

分桶结构：scanqa 上 OD 效应（OD6−OD7）spatial +4.23 / nonspatial +1.71——**无点云时 OD 文本主要贡献空间题**（计数+关系可从框推导）；而 OD6 报告中 pc 效应主要在非空间桶（颜色/外观）——两种信息源恰好互补覆盖两类题。sqa3d 非空间桶 n=99 无统计力。

## 6. 题型分解（sqa3d EM1；n=1147/652/465/338/351/566）

| 题型 | OD7 | baseline | OD6 | OD6−OD7（OD 文本因果贡献） |
|---|---|---|---|---|
| type2 How-many（计数） | 48.82 | 49.03 | 54.41 | **+5.59**（计数增益 100% 来自框列表） |
| type0 What（识别） | 39.76 | 45.16 | 42.55 | +2.79（外观仍主要靠点云） |
| type1 Is（是否存在） | 65.64 | 68.71 | 68.25 | +2.61（**预注册预测失手**：存在性题需要场景证据） |
| type5 Are/If/Where | 51.94 | 53.18 | 54.06 | +2.12 |
| type4 Which（辨别） | **48.15** | 45.87 | 45.87 | −2.28（纯语言辨别，OD7 最强） |
| type3 Can-see（可见性） | 70.41 | 72.19 | 68.93 | −1.48 |

## 7. 局限

- 任务混合混杂（承 OD6）：OD5 训 293k 多任务 vs OD7/OD6 训 53k 纯 VQA——但 OD7 vs OD6 同数据同混合，**该对比无混杂**；涉及 OD5/baseline 的对比（+pc 格）含此混杂
- OD7 是"微调后的语言先验"（1 epoch LoRA 在 53k VQA 上）——不是零样本先验；但其相对位次不受此影响
- eval 无 od 注入的 prompt 与训练格式逐字节同构（问题+后缀），无版式差异
- 15 条空 GT 样本与 OD6 对称处理

## 8. 产物

- 数据：`scripts/generate_q_only_data.py`、`playground/data/train_info/{scanqa,sqa3d}_train_3d_llava_textonly.json`
- 脚本：`scripts/train/finetune-3d-llava-lora-textonly.sh`、`scripts/eval/multigpu_eval_all_textonly.sh`（无 `--od-file`）
- checkpoint：`checkpoints/finetune-3d-llava-lora-textonly/`（LoRA）；预测+指标：`playground/predictions/finetune-3d-llava-lora-textonly/`
- 分析：两脚本第 8 run（`textonly`，`runs_for` 过滤 + FULLCHECK_REF 真值回填）；日志 `logs/spatial_split_textonly.log`、`logs/spatial_split_paired_textonly.log`
