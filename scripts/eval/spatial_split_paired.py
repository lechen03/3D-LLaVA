#!/usr/bin/env python3
"""Paired (McNemar) per-question EM analysis for the spatial split.

For sqa3d / scanqa: recompute per-question EM1 with the eval scripts' own
clean_answer/answer_match, pair baseline vs with-od on the SAME questions
(merge files are line-aligned with the annotation file), and report flip
counts per spatial bucket with an exact two-sided binomial McNemar p-value.

Run with the llava env python (module imports pycocoevalcap):
  /home/chenle/miniconda3/envs/llava/bin/python scripts/eval/spatial_split_paired.py
"""

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "llava/eval"))

from eval_scanqa import clean_answer, answer_match  # noqa: E402

RUNS = {
    "baseline": REPO / "playground/predictions/finetune-3d-llava-lora",
    "with_od": REPO / "playground/predictions/finetune-3d-llava-lora-with-od",
    "with_od_topk": REPO / "playground/predictions/finetune-3d-llava-lora-with-od-topk",
}
EVAL_INFO = REPO / "playground/data/eval_info"

sys.path.insert(0, str(REPO / "scripts/eval"))
from spatial_split_eval import SPATIAL_RE, is_spatial  # noqa: E402


def mcnemar_exact_p(b, c):
    """Two-sided exact binomial test on discordant pairs (b+c trials, p=.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * p)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def em1_flags(preds, gts):
    flags = []
    for pred, gt in zip(preds, gts):
        pred_answer = clean_answer(pred["text"])
        ref = [clean_answer(g) for g in gt["text"]]
        em1, _ = answer_match(pred_answer, ref)
        flags.append(em1)
    return flags


def analyze(task, ann_file, run_task_dir):
    ann = read_jsonl(ann_file)
    questions = {
        "sqa3d": EVAL_INFO / "sqa3d/sqa3d_test_question.json",
        "scanqa": EVAL_INFO / "scanqa/scanqa_val_question.jsonl",
    }[task]
    qtexts = [q["text"] for q in read_jsonl(questions)]
    assert len(qtexts) == len(ann)
    flags = [is_spatial(t) for t in qtexts]

    run_em = {}
    for run, run_dir in RUNS.items():
        preds = read_jsonl(run_dir / run_task_dir / "merge.jsonl")
        assert len(preds) == len(ann)
        run_em[run] = em1_flags(preds, ann)

    for run_a, run_b in [("baseline", "with_od"), ("baseline", "with_od_topk"),
                          ("with_od", "with_od_topk")]:
        print(f"\n===== {task} EM1 paired ({run_a} vs {run_b}) =====")
        print(f"{'bucket':12s} {'n':>5s} {'base':>7s} {'od':>7s} {'delta':>7s} "
              f"{'b(od+base-)':>12s} {'c(od-base+)':>12s} {'p_exact':>9s}")
        for bucket, want in [("spatial", True), ("nonspatial", False), ("full", None)]:
            idx = [i for i in range(len(ann)) if want is None or flags[i] == want]
            base = [run_em[run_a][i] for i in idx]
            od = [run_em[run_b][i] for i in idx]
            acc_b = sum(base) / len(idx)
            acc_o = sum(od) / len(idx)
            b = sum(1 for x, y in zip(od, base) if x and not y)
            c = sum(1 for x, y in zip(od, base) if y and not x)
            p = mcnemar_exact_p(b, c)
            print(f"{bucket:12s} {len(idx):5d} {acc_b:7.4f} {acc_o:7.4f} "
                  f"{(acc_o - acc_b) * 100:+7.2f} {b:12d} {c:12d} {p:9.4f}")


if __name__ == "__main__":
    analyze("sqa3d", EVAL_INFO / "sqa3d/sqa3d_test_answer.json", "sqa3d")
    analyze("scanqa", EVAL_INFO / "scanqa/scanqa_val_answer.jsonl", "scanqa")
