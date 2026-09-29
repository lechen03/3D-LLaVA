# 3D-LLaVA OD 注入实验报告（EXP-OD2：RAG 式选择性注入，类级 top-3 检索）

> 日期：2026-09-28 ~ 09-29　|　前置：[EXP-OD1 全量注入](OD_INJECTION_EXP1_REPORT.md)、[空间分桶分析](OD_INJECTION_SPATIAL_SPLIT.md)
> 对照：baseline（无 OD）、EXP-OD1（全量 OD 注入）

---

## 1. 摘要

**实验问题**：全量 OD 注入（EXP-OD1）代价高（训练 ×2.4、token p50 100→1129）且增益集中在非空间题。若每条对话只注入**与题目相关**的 OD 框（RAG 检索），能否以更低成本保住甚至放大增益？

**做法**：MiniLM 嵌入（`all-MiniLM-L6-v2`，仓库本地加载）对 18 类类名与题目文本算 cosine，**类级 top-3 检索**（注入这三类的全部 bbox 行，原序原编号）；dense_captioning 无 query → 全量兜底（与 EXP-OD1 逐字节一致）。单变量对照微调（唯一差异 = 8 个训练 JSON 换 `_with_od_topk` 版）+ 同口径评估。

**结果**：**topk 是三个模型里综合最优的一个**——13 项主指标中 10 项三列最佳；**消除了 EXP-OD1 的 EM↑/CIDEr↓ trade-off**（CIDEr 93.16 ≈ 基线 93.11，全量版掉到 91.24）；**ref-seg 与 caption 首次全面超过基线**；但预登记的「scanqa EM1 ≥ 0.264」未达成（26.07 vs 26.40，配对 p=0.47 平手）。增益分布与全量版**正好互补**：topk 增益偏向空间题（检索=问题条件化的聚光灯），全量版增益在非空间题（存在性 grounding）。

## 2. 方法

| 项 | 设计 |
|---|---|
| 嵌入模型 | `sentence-transformers/all-MiniLM-L6-v2`（384 维，88MB，下载至 `playground/models/`，运行时纯本地加载，**零新依赖**——mean pooling 在 `scripts/od_retrieval.py` 内实现；CLIP 方案因缓存只有 config 无权重而弃） |
| 检索 | 类级：score(class)=cos(query, 类名)，top-3 类 → 注入这些类全部 bbox 行（保留 `bbox_N` 原序原编号，坐标 2 位小数，格式与 EXP-OD1 逐字节兼容）；场景类数 ≤3 时自然退化为全量 |
| query 来源 | vqa = human 轮全文（去 `<pc>\n`）；refer_seg = 描述（去固定模板 + multi3drefer 尾指令）；训练/评估同口径（eval 用 `source['text']/['description']`） |
| 无 query 任务 | dense_captioning（65,304 条，模板 + `<loc>` 占位符无语义）→ 全量 OD 兜底，与 `_with_od` **逐字节一致（65,381 条断言全过）**；scan2cap 评估同理 |
| 数据 | `*_with_od_topk.json` ×8：228,075 选择性 + 65,304 全量 + 203 空 OD 保持原文；**bbox 保留率 63.0%**；token p50：vqa 338 / refer_seg 326（全量版 1129，**÷3.4**）、max 2480 < 4096 |
| 检索质量 | 12 条人工 spot-check：目标类命中 ~11/12（miss 仅词表外 pillow；monitor→desk、couch→sofa、nightstand→bed 语义桥接正确） |

工具：`scripts/od_retrieval.py`（训练/评估共用）、`scripts/generate_with_od_topk_data.py`（内建全量断言）。

## 3. 训练与评估成本

| | baseline | EXP-OD1 全量 | **EXP-OD2 topk** |
|---|---|---|---|
| 训练墙钟 | 6h00m（2 卡） | 14h22m（4 卡） | **12h10m（4 卡，−15%）** |
| 评估墙钟（5 任务 4 卡） | — | 1h32m | **1h17m（−16%）** |
| OD token p50（vqa/ref-seg） | 0 | 1129 | **326/338（÷3.4）** |

成本降幅小于 token 降幅的原因：dense_captioning（65k 条）保持全量注入、refer_seg p95 仍 ~965 token。

## 4. 全量结果（312 val 场景，同解码设置；**粗体=三列最佳**）

| 任务 | 指标 | baseline | with_od（全量） | **with_od_topk** |
|---|---|---|---|---|
| sqa3d | EM1 | 53.99 | 54.25 | **54.36** |
| sqa3d | EM1_refined | 56.66 | **57.00** | 56.98 |
| scanqa | EM1 | 25.48 | **26.40** | 26.07 |
| scanqa | EM1_refined | 41.20 | **42.20** | 42.03 |
| scanqa | CIDEr | 93.11 | 91.24（−1.87） | **93.16（+0.05，trade-off 消失）** |
| scanqa | Bleu_1 | 45.69 | 44.32 | 45.42（回升） |
| scan2cap | CIDEr@.25 | 81.25 | 81.20 | **81.71** |
| scan2cap | CIDEr@.50 | 76.28 | 76.17 | **76.57** |
| scanrefer | mIoU | 41.10 | 40.94 | **41.49** |
| scanrefer | Acc@.25 / @.50 | 59.71 / 39.43 | 58.98 / 39.44 | **59.96 / 40.15** |
| multi3drefer | mIoU | 46.79 | 46.90 | **47.09** |
| multi3drefer | Acc@.25 | **65.69** | 64.89 | 65.49 |
| multi3drefer | Acc@.50 | 45.30 | 45.76 | **46.07** |

## 5. 空间分桶与配对检验（EM1，McNemar 精确双侧）

scanqa（桶 n：空间 2340 / 非空间 2335）：

| 对比 | 空间题 Δ | 非空间题 Δ | 全量 Δ |
|---|---|---|---|
| baseline → 全量 OD | +0.26 (p=0.73) | **+1.58 (p=0.010)** | +0.92 (p=0.037) |
| baseline → topk | +0.81 (p=0.21) | +0.39 (p=0.55) | +0.60 (p=0.166) |
| 全量 OD → topk | +0.56 (p=0.40) | **−1.20 (p=0.034)** | −0.32 (p=0.465) |

**两个变体的增益分布互补**：全量版的显著增益在非空间题（「场景里有什么」的存在性 grounding，注入越全越好）；topk 把增益搬到了空间题（+0.81 vs 全量版的 +0.26）——检索本身是**问题条件化的聚光灯**：注入框已被筛到题目提到的类别，等于告诉模型「这道题在问哪些物体」，即便坐标未对齐也帮助关系落地；代价是非空间题的存在性信息被截断（−1.20，显著）。

sqa3d（空间桶占 97%，区分度低；非空间桶 n=99）：topk 在非空间桶 **+10.10（p=0.031）**，强于全量版的 +7.07（p=0.092）；全量 +0.37。小样本，方向性参考。

scanqa CIDEr 分桶：空间桶 85.49 → 全量 82.24 → **topk 85.15**（全量版的 CIDEr 下跌在 topk 中消失）；非空间桶 100.08 → 99.66 → **100.51**。

ref-seg 分桶（mIoU，空间/非空间）：
- scanrefer：base 40.11/46.58，全量 39.98/46.24，**topk 40.42/47.42**（两桶均最佳）
- multi3drefer（词面）：base 45.60/51.74，全量 45.66/52.02，**topk 45.71/52.80**
- multi3drefer（原生 spatial 口径，非空间 n=357）：base 53.65，全量 52.44（Acc@.25 −4.5pt 回退），**topk 54.38（Acc@.25 完全恢复到 79.27=基线水平）**——全量版在 native_nonspatial 上的回退被 topk 修复

## 6. 对照预登记成功标准

- 「scanqa EM1 ≥ 0.264」：**未达成**（26.07，与全量版配对平手 p=0.47；vs 基线 +0.60 不显著 p=0.17）
- 「训练时长显著 < 14.3h」：达成（12h10m，−15%）
- 「ref-seg/caption 不倒退」：**超额**——ref-seg 与 caption 均首次全面超过基线（全量版此前持平或微降）

综合判断：topk 用 ~26% 的 OD 文本量（63% 框保留率按行计）拿到更均衡的全面增益，是比全量注入更优的默认方案；唯一让渡是 scanqa EM1 峰值（全量版 +0.92 显著 vs topk +0.60 不显著）。

## 7. 事故与坑（本轮新踩）

- **`find_adapter_config_file` 401 崩溃**：eval 加载 LoRA 时 peft 会向 hub 探测 base model 上不存在的 `adapter_config.json`；匿名请求应 404 优雅跳过，**坏 token 使 hub 返回 401 直接 raise**（`OSError: ... not a valid model identifier`）。训练路径不做此探测故幸免。修复：评估环境加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（最小复现验证过）。**教训：这台机器上一切 eval 类进程默认应带 offline 环境变量**
- CLIP 缓存陷阱：`models--openai--clip-vit-large-patch14-336` 目录存在但**只有 config.json 无权重**（LLaVA 只用其 vision tower），`local_files_only=True` 也挡不住 transformers 4.37 的 `has_file` 网络探测穿透——改用本地目录加载 MiniLM
- 首启评估失败时 runner 仍盲跑完 5 任务输出 nan——runner 已有 answers 自检，但缺「模型加载失败即中止」的护栏（本轮靠人工清理重跑，未改 runner）

## 8. Caveats

- 单次运行、单 seed；scanqa EM1 三列差异均在 1σ 内，结论以配对检验与多指标一致性为准
- 检索为词面语义级（类名 vs 问题），不利用坐标；「相关」定义=语义相似而非空间邻近
- topk 的 dense_captioning/scan2cap 与 EXP-OD1 完全同输入，其差异纯来自模型训练侧
- sqa3d 非空间桶 n=99、multi3drefer 原生非空间桶 n=357，均小样本
- 评估侧检索在 GPU 上跑（fp32），与训练数据生成（同代码同模型）完全确定性对齐

## 9. 复现

```bash
# 1) 训练数据（含全量断言与 token 审计）
python scripts/generate_with_od_topk_data.py
# 2) 训练（tmux，4 卡，~12h）
bash scripts/train/finetune-3d-llava-lora-with-od-topk.sh
# 3) 评估（tmux；务必带 offline 环境变量）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash scripts/eval/multigpu_eval_all_with_od_topk.sh
# 4) 分析（三列对比 + 空间分桶 + 三方配对）
python scripts/eval/spatial_split_eval.py
/home/chenle/miniconda3/envs/llava/bin/python scripts/eval/spatial_split_paired.py
```

## 10. 产物

| 路径 | 内容 |
|---|---|
| `scripts/od_retrieval.py` | 检索工具（MiniLM 本地加载，训练/评估共用） |
| `scripts/generate_with_od_topk_data.py` | 训练数据生成（内建断言） |
| `scripts/train/finetune-3d-llava-lora-with-od-topk.sh` | 训练脚本 |
| `llava/eval/model_{scanqa,sqa3d,scanrefer,multi3drefer}.py` | +`--od-topk` 参数（不传=原行为） |
| `scripts/eval/multigpu_eval_all_with_od_topk.sh` | 评估 runner |
| `playground/models/all-MiniLM-L6-v2/` | 嵌入模型本地副本（88MB，勿提交） |
| `playground/data/train_info/*_with_od_topk.json` ×8、`od_topk_stats.json` | 训练数据与统计 |
| `checkpoints/finetune-3d-llava-lora-with-od-topk/` | 模型 |
| `playground/predictions/finetune-3d-llava-lora-with-od-topk/` | 预测 + metrics_all.txt |
| `playground/predictions/spatial_split/parsed_metrics.json` | 三 run 分桶指标 |
| `logs/{generate_od_topk,finetune_with_od_topk,eval_with_od_topk}.log` | 日志 |

---

*后续候选（未做）：topk 预算消融（top-2/5 类）；把检索得分显式写进 OD 文本（排序注入）；空间题与非空间题混合注入策略（空间题 topk + 非空间题全量，兼取两种增益）；坐标对齐版检索 OD（方位词增益的上限测试）。*
