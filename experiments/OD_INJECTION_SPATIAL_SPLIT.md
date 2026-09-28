# OD 注入实验·空间关系问题分桶分析（EXP-OD1 后续）

> 日期：2026-09-28　|　前置实验：[OD_INJECTION_EXP1_REPORT.md](OD_INJECTION_EXP1_REPORT.md)
> 方法：纯后处理——复用两套已有预测（baseline / with-od）分桶重算指标，**零推理**

---

## 1. 问题与结论

**问题**：OD 注入在 scanqa/sqa3d 上的 EM 增益，是否集中在「带空间关系的问题」（left, right, up, down, front, beside 等方向/关系词）？

**结论：不是——增益恰好集中在 *非* 空间关系问题上。**

- scanqa EM1 全量 +0.92pt 经逐题配对 McNemar 检验为显著（p=0.037，405 次翻转 224:181），但拆开后：**非空间题 +1.58pt（p=0.010，显著），空间题 +0.26pt（p=0.73，无增益）**
- 解读：EXP4 检测先验提供的是「场景里**有什么**」的 grounding（类别/存在性先验），而非「X 在 Y 的哪个方向」的空间关系推理能力
- 这与坐标 caveat 一致：OD 坐标与 3D-LLaVA 输入帧不对齐（vqa 帧差每场景常量平移、refer_seg 完全不对齐），方向词增益本就无从借力
- scanqa 的 CIDEr 下降同样集中在空间题桶（−3.25pt vs 非空间桶 −0.42pt）
- ref-seg 两任务两桶全部持平（±0.4pt）；sqa3d 两桶均在噪声内（97% 的题因情境前导语含方向词被归入空间桶，非空间桶仅 99 条）

## 2. 口径

- **分桶信号**（输入侧）：sqa3d/scanqa = 问题全文（sqa3d 含情境前导语）；scanrefer/multi3drefer = referring `description`；scan2cap 无输入文本 → 按 **GT 参考描述**分桶（输出侧口径，与其它任务不同）
- **关键词**（严格档，`\b` 边界忽略大小写，用户选定不含 on/in/at 等泛介词）：left, leftmost, right, rightmost, up, upper, down, lower, front, back, rear, behind, beside, alongside, above, below, under, underneath, beneath, between, among, middle, center, centre, central, adjacent, against, facing, toward, towards, near, nearby, nearest, closest, furthest, farthest, farther, further, atop, next to, close to, on top of, far from
- **指标复算机制**：sqa3d/scanqa 子集化 result+annotation 两文件（eval 为位置 zip + 逐行 qid 断言）后原样调 `eval_sqa3d/eval_scanqa`；refer_seg 两任务内联均值（与 `eval_refer_seg` 数学等价）；scan2cap 子集化 result+corpus（否则缺失键被回填零分）
- **管线自检（硬门槛）**：每个任务的 full 桶先重算并逐值比对 `metrics_all.txt` / `baseline_metrics` 参考值——5 任务 × 2 run 全部一致后才采信分桶数字
- **配对检验**：用 eval 脚本自带 `clean_answer/answer_match` 逐题重算 EM1，McNemar 精确二项检验（双侧）
- multi3drefer 另报数据集原生 `spatial` 布尔口径（比词面宽：10763 vs 8957 命中）

## 3. 分桶结果

桶大小（spatial / nonspatial）：scanqa 2340/2335、sqa3d 3420/99、scanrefer 8055/1453、multi3drefer 8957/2163（词面）/ 10763/357（原生）、scan2cap 2003/4。

### scanqa（EM 为 %；p 为配对 McNemar 精确双侧）

| 指标 | 空间题 base | 空间题 od | Δ | 非空间题 base | 非空间题 od | Δ |
|---|---|---|---|---|---|---|
| EM1 | 26.45 | 26.71 | **+0.26** (p=0.73) | 24.50 | 26.08 | **+1.58** (p=0.010) |
| EM1_refined | 42.95 | 43.12 | +0.17 | 39.44 | 41.28 | +1.84 |
| CIDEr | 85.49 | 82.24 | −3.25 | 100.08 | 99.66 | −0.42 |
| Bleu_1 | 42.28 | 40.39 | −1.89 | 48.21 | 47.25 | −0.96 |

全量参考：EM1 25.48→26.40（+0.92, p=0.037, b/c=224/181）；EM1_refined 41.20→42.20；CIDEr 93.11→91.24。

### sqa3d

| 指标 | 空间题 (n=3420) base | od | Δ | 非空间题 (n=99) base | od | Δ |
|---|---|---|---|---|---|---|
| EM1 | 54.30 | 54.36 | +0.06 (p=0.96) | 43.43 | 50.51 | +7.07 (p=0.092) |
| EM1_refined | 57.02 | 57.13 | +0.11 | 44.44 | 52.53 | +8.09 |

非空间桶方向与 scanqa 一致（非空间题受益），但 n=99（σ≈5pt）只能算边缘证据。sqa3d 情境前导语（"I am standing... on my right facing..."）几乎总含方向词，97.2% 被归入空间桶，该任务分桶区分度天然不足。

### referring segmentation

| 任务 / 桶 | mIoU base→od (Δ) | Acc@.25 (Δ) | Acc@.50 (Δ) |
|---|---|---|---|
| scanrefer 空间题 (8055) | 40.11→39.98 (−0.13) | 58.63→57.90 (−0.73) | 38.18→38.11 (−0.07) |
| scanrefer 非空间题 (1453) | 46.58→46.24 (−0.34) | 65.66→64.97 (−0.69) | 46.39→46.80 (+0.41) |
| multi3drefer 空间题 (8957) | 45.60→45.66 (+0.06) | 63.69→63.03 (−0.66) | 43.60→43.88 (+0.28) |
| multi3drefer 非空间题 (2163) | 51.74→52.02 (+0.28) | 73.97→72.58 (−1.39) | 52.33→53.54 (+1.21) |
| multi3drefer 原生空间 (10763) | 46.57→46.72 (+0.15) | 65.24→64.56 (−0.68) | 44.96→45.48 (+0.52) |
| multi3drefer 原生非空间 (357) | 53.65→52.44 (−1.21) | 79.27→74.79 (−4.48) | 55.46→54.06 (−1.40) |

全桶持平（原生非空间桶 Acc@.25 −4.5pt 约 2σ，n=357，量级小且是次要口径，仅记录）。基线侧规律本身成立：**空间关系描述显著更难**（scanrefer mIoU 低 6.5pt、multi3drefer 低 6.1pt），OD 未改变这一点。

### scan2cap（GT 描述分桶，输出侧口径）

| 桶 | CIDEr@.25 base→od | CIDEr@.50 |
|---|---|---|
| 空间描述 (2003/2007) | 81.25→81.21 (−0.04) | 76.27→76.17 (−0.10) |
| 非空间描述 (4/2007) | n=4，无统计意义 | — |

ScanNet dense caption 的 GT 几乎必含相对方位描述（2003/2007 命中），该任务无法做有效分桶。

## 4. 解读与后续

1. **OD 先验 = 存在性/类别 grounding，不是空间推理**：VQA 增益集中在「场景里有什么」型问题上；方向词问题的答案需要帧对齐的几何关系，未对齐的 bbox 文本给不了
2. 若想提升空间关系题：坐标对齐版 OD（vqa 帧公式已备）或把 OD 文本升级为显式相对方位描述（如 "chair: left of desk"），是比加大剂量更对症的方向
3. `right` 歧义敏感度（仅由 'right' 触发空间分类的条目）：sqa3d 230/3420、scanqa 321/2340、scanrefer 978/8055、multi3drefer 1044/8957、scan2cap 17/2003——即便全部误判也不改变各桶结论
4. Caveats：词面启发式分桶；scanqa CIDEr 在子集内重算 IDF（两 run 对称）；单次 run、单 seed；EM1_refined 与 EM1 同方向

## 5. 复现

```bash
# 分桶指标（含 full 桶自检；~10 分钟，scanqa/scan2cap 指标走 Java）
python scripts/eval/spatial_split_eval.py          # 产物 playground/predictions/spatial_split/
# 逐题配对 McNemar（llava env）
/home/chenle/miniconda3/envs/llava/bin/python scripts/eval/spatial_split_paired.py
# 汇总数字：playground/predictions/spatial_split/parsed_metrics.json，日志 logs/spatial_split.log
```
