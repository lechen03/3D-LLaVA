#!/usr/bin/env python3
"""Post-hoc spatial-relation split analysis for the OD injection experiment.

Splits each eval task's val questions into two buckets by whether the input
text contains directional/relational keywords (left, right, up, down, front,
beside, ... -- strict tier, no bare on/in/at), then recomputes baseline vs
with-od metrics per bucket from the EXISTING prediction files (no inference).

Bucket signal per task (input side unless noted):
  sqa3d / scanqa        question `text` (includes sqa3d situation preamble)
  referseg_scanrefer    `description`
  multi3drefer          `description`  (+ dataset-native `spatial` bool as a
                        second, broader split)
  scan2cap              GT captions (output-side split: the input prompt is a
                        constant template with no per-instance text)

Alignment facts (verified): both runs' merge.jsonl are line-aligned with the
source question files.  eval_sqa3d/eval_scanqa zip result+annotation
positionally (assert len + per-line question_id), so subsets must cut BOTH
files in the same order.  eval_refer_seg just averages iou/tp25/tp50 over
result lines (computed inline, mathematically identical).  eval_scan2cap
back-fills corpus keys missing from the result with zero scores, so subsets
must cut the corpus dict too.

A "full" bucket (all lines) is always computed first and compared against the
reference metrics recorded in metrics_all.txt / baseline_metrics -- if that
check fails the script aborts before reporting any subset numbers.

Usage (from repo root):
  python scripts/eval/spatial_split_eval.py            # full check + split
  python scripts/eval/spatial_split_eval.py --tasks sqa3d scanqa
"""

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LLAVA_PY = "/home/chenle/miniconda3/envs/llava/bin/python"
OUT_DIR = REPO / "playground/predictions/spatial_split"

RUNS = {
    "baseline": REPO / "playground/predictions/finetune-3d-llava-lora",
    "with_od": REPO / "playground/predictions/finetune-3d-llava-lora-with-od",
}

EVAL_INFO = REPO / "playground/data/eval_info"

# Strict directional/relational keyword tier (user-approved; excludes bare
# on/in/at/inside/around).  Word-boundary, case-insensitive.
SPATIAL_RE = re.compile(
    r"\b(?:left|leftmost|right|rightmost|up|upper|down|lower|front|back|rear|"
    r"behind|beside|alongside|above|below|under|underneath|beneath|between|"
    r"among|middle|center|centre|central|adjacent|against|facing|toward|"
    r"towards|near|nearby|nearest|closest|furthest|farthest|farther|further|"
    r"atop|next to|close to|on top of|far from)\b",
    re.IGNORECASE,
)
RIGHT_RE = re.compile(r"\bright\b", re.IGNORECASE)


def is_spatial(text):
    return bool(SPATIAL_RE.search(text))


def right_only_trigger(text):
    """Spatial flag flips to False if all 'right' mentions are removed."""
    return is_spatial(text) and not SPATIAL_RE.search(RIGHT_RE.sub("", text))


# Reference full-set metrics (metrics_all.txt / baseline_metrics) for the
# harness self-check.
FULLCHECK_REF = {
    ("sqa3d", "with_od"): {
        "[sqa3d] EM1": 0.5424836601307189,
        "[sqa3d] EM1_refined": 0.5700483091787439,
    },
    ("sqa3d", "baseline"): {
        "[sqa3d] EM1": 0.5399261153736857,
        "[sqa3d] EM1_refined": 0.5666382495026996,
    },
    ("scanqa", "with_od"): {
        "[scanqa] EM1": 0.2639572192513369,
        "[scanqa] EM1_refined": 0.42203208556149735,
        "[scanqa] Bleu_1": 0.44315822719628106,
        "[scanqa] CIDEr": 0.9124145123343258,
    },
    ("scanqa", "baseline"): {
        "[scanqa] EM1": 0.25475935828877005,
        "[scanqa] EM1_refined": 0.41197860962566846,
        "[scanqa] Bleu_1": 0.4569000012733735,
        "[scanqa] CIDEr": 0.9311336906052258,
    },
    ("scan2cap", "with_od"): {
        "[scan2cap] CIDEr@0.25": 0.8120326390607033,
        "[scan2cap] CIDEr@0.50": 0.7617320765698825,
    },
    ("scan2cap", "baseline"): {
        "[scan2cap] CIDEr@0.25": 0.8124614688242546,
        "[scan2cap] CIDEr@0.50": 0.7627650756902353,
    },
    # referseg numbers were only recorded at 4 decimals -> compare rounded
    ("referseg_scanrefer", "with_od"): (0.4094, 0.5898, 0.3944),
    ("referseg_scanrefer", "baseline"): (0.4110, 0.5971, 0.3943),
    ("multi3drefer", "with_od"): (0.4690, 0.6489, 0.4576),
    ("multi3drefer", "baseline"): (0.4679, 0.6569, 0.4530),
}

MULTI3D_PROMPT_PREFIX = (
    " Please output the segmentation mask according to the following description. \n"
)
MULTI3D_PROMPT_SUFFIX = (
    "\nThere may be no corresponding object, or there may be one or more objects."
)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def read_raw_lines(path):
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_eval(cmd, timeout=600):
    """Run a metric script; return its final printed metrics dict."""
    import os

    env = dict(os.environ, PYTHONPATH=str(REPO))
    proc = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:])
        sys.stderr.write(proc.stderr[-3000:])
        raise RuntimeError(f"metric command failed: {' '.join(cmd)}")
    metric_line = None
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith("{'["):
            metric_line = s
    if metric_line is None:
        raise RuntimeError(f"no metrics dict found in output of: {' '.join(cmd)}")
    return ast.literal_eval(metric_line)


def rel(path):
    return str(Path(path).relative_to(REPO))


def check_full(task, run, metrics):
    """Compare full-bucket metrics against the recorded reference values."""
    ref = FULLCHECK_REF[(task, run)]
    if isinstance(ref, tuple):  # referseg triple, 4-decimal
        got = tuple(round(float(v), 4) for v in metrics)
        if got != tuple(round(float(v), 4) for v in ref):
            raise AssertionError(f"{task}/{run} full-check: got {got}, want {ref}")
        return
    for key, want in ref.items():
        got = metrics.get(key)
        if got is None or abs(got - want) > 1e-9:
            raise AssertionError(f"{task}/{run} full-check {key}: got {got}, want {want}")


def buckets_from_flags(flags):
    """Return list of (bucket_name, index_list); 'full' first for self-check."""
    n = len(flags)
    idx = list(range(n))
    sp = [i for i in idx if flags[i]]
    non = [i for i in idx if not flags[i]]
    assert len(sp) + len(non) == n
    return [
        ("full", idx),
        ("spatial", sp),
        ("nonspatial", non),
    ]


def task_vqa(task, question_file, ann_file, eval_module):
    """sqa3d / scanqa: subset result+annotation (positional zip in the eval)."""
    questions = read_jsonl(question_file)
    ann_raw = read_raw_lines(ann_file)
    answers = [json.loads(line) for line in ann_raw]
    assert len(questions) == len(answers), f"{task}: question/answer count mismatch"
    flags = [is_spatial(q["text"]) for q in questions]
    n_right_only = sum(right_only_trigger(q["text"]) for q in questions)

    out = {"n_right_only_spatial": n_right_only}
    for run, run_dir in RUNS.items():
        pred_raw = read_raw_lines(run_dir / task / "merge.jsonl")
        preds = [json.loads(line) for line in pred_raw]
        assert len(preds) == len(questions), f"{task}/{run}: merge line count"
        for i, q in enumerate(questions):
            assert (
                preds[i]["question_id"] == q["question_id"] == answers[i]["question_id"]
            ), f"{task}/{run}: misaligned at line {i}"
        for bucket, sel in buckets_from_flags(flags):
            sub_res = OUT_DIR / task / run / f"{bucket}_result.jsonl"
            sub_ann = OUT_DIR / task / run / f"{bucket}_ann.jsonl"
            write_lines(sub_res, [pred_raw[i] for i in sel])
            write_lines(sub_ann, [ann_raw[i] for i in sel])
            metrics = run_eval(
                [
                    LLAVA_PY,
                    f"llava/eval/{eval_module}.py",
                    "--annotation-file",
                    rel(sub_ann),
                    "--result-file",
                    rel(sub_res),
                ]
            )
            metrics["n"] = len(sel)
            if bucket == "full":
                check_full(task, run, metrics)
            out[(run, bucket)] = metrics
            print(f"[{task}] {run}/{bucket} n={len(sel)} done", flush=True)
    return out


def seg_means(records):
    return {
        "mIoU": float(np.mean([r["iou"] for r in records])),
        "Acc25": float(np.mean([r["tp25"] for r in records])),
        "Acc50": float(np.mean([r["tp50"] for r in records])),
    }


def extract_description_from_prompt(prompt, with_suffix):
    """Pull the referring description back out of a stored eval prompt.

    Handles both baseline prompts and with-od prompts (skip the <od> block).
    """
    p = prompt
    assert p.startswith("<image>\n"), p[:40]
    p = p[len("<image>\n") :]
    if p.startswith("<od>\n"):
        end = p.index("\n</od>\n") + len("\n</od>\n")
        p = p[end:]
    assert p.startswith(MULTI3D_PROMPT_PREFIX), p[:80]
    p = p[len(MULTI3D_PROMPT_PREFIX) :]
    if with_suffix:
        assert p.endswith(MULTI3D_PROMPT_SUFFIX), p[-80:]
        p = p[: -len(MULTI3D_PROMPT_SUFFIX)]
    return p


def task_referseg(task, question_file, with_suffix, native_spatial=False):
    """referseg_scanrefer / multi3drefer: inline means (== eval_refer_seg)."""
    questions = json.load(open(question_file))
    flags = [is_spatial(q["description"]) for q in questions]
    native = [bool(q["spatial"]) for q in questions] if native_spatial else None
    n_right_only = sum(right_only_trigger(q["description"]) for q in questions)

    out = {"n_right_only_spatial": n_right_only}
    for run, run_dir in RUNS.items():
        preds = read_jsonl(run_dir / task / "merge.jsonl")
        assert len(preds) == len(questions), f"{task}/{run}: merge line count"
        for i, q in enumerate(questions):
            if "scene_id" in preds[i]:
                assert preds[i]["scene_id"] == q["scene_id"], (
                    f"{task}/{run}: scene misaligned at line {i}"
                )
            desc = extract_description_from_prompt(preds[i]["prompt"], with_suffix)
            assert desc == q["description"], f"{task}/{run}: desc mismatch at line {i}"
        for bucket, sel in buckets_from_flags(flags):
            recs = [preds[i] for i in sel]
            metrics = seg_means(recs)
            metrics["n"] = len(sel)
            if bucket == "full":
                check_full(
                    task,
                    run,
                    (metrics["mIoU"], metrics["Acc25"], metrics["Acc50"]),
                )
            out[(run, bucket)] = metrics
        if native_spatial:  # dataset-native `spatial` bool, second split
            for bucket, sel in buckets_from_flags(native)[1:]:
                metrics = seg_means([preds[i] for i in sel])
                metrics["n"] = len(sel)
                out[(run, "native_" + bucket)] = metrics
        print(f"[{task}] {run} done", flush=True)
    out["n_native_spatial"] = int(sum(native)) if native_spatial else None
    out["n_lexical_spatial"] = int(sum(flags))
    return out


def task_scan2cap(question_file, corpus_file, pred_attrs, gt_attrs):
    """scan2cap: split by GT captions (output-side); cut corpus + result."""
    corpus = json.load(open(corpus_file))
    key_flags = {k: is_spatial(" ".join(v)) for k, v in corpus.items()}
    n_right_only = sum(right_only_trigger(" ".join(v)) for v in corpus.values())

    out = {"n_right_only_spatial": n_right_only}
    for run, run_dir in RUNS.items():
        pred_raw = read_raw_lines(run_dir / "densecap_scanrefer" / "merge.jsonl")
        preds = [json.loads(line) for line in pred_raw]
        keys = [f"{p['scene_id']}|{p['gt_id']}" for p in preds]
        for k in keys:
            assert k in key_flags, f"scan2cap/{run}: pred key {k} not in corpus"
        line_flags = [key_flags[k] for k in keys]

        for bucket, sel in buckets_from_flags(line_flags):
            sub_res = OUT_DIR / "scan2cap" / run / f"{bucket}_result.jsonl"
            sub_corpus = OUT_DIR / "scan2cap" / run / f"{bucket}_corpus.json"
            write_lines(sub_res, [pred_raw[i] for i in sel])
            # cut the corpus over ALL its keys (incl. the 61 keys with no
            # prediction) so that n_spatial + n_nonspatial == len(corpus)
            keep = {
                k: v
                for k, v in corpus.items()
                if bucket == "full" or key_flags[k] == (bucket == "spatial")
            }
            sub_corpus.write_text(json.dumps(keep))
            metrics = run_eval(
                [
                    LLAVA_PY,
                    "llava/eval/eval_scan2cap.py",
                    "--pred-instance-attribute-file",
                    rel(pred_attrs),
                    "--gt-instance-attribute-file",
                    rel(gt_attrs),
                    "--annotation-file",
                    rel(sub_corpus),
                    "--result-file",
                    rel(sub_res),
                ]
            )
            metrics["n_result"] = len(sel)
            metrics["n_corpus"] = len(keep)
            if bucket == "full":
                check_full(task="scan2cap", run=run, metrics=metrics)
            out[(run, bucket)] = metrics
            print(f"[scan2cap] {run}/{bucket} n={len(sel)} done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=["sqa3d", "scanqa", "scan2cap", "referseg_scanrefer", "multi3drefer"],
    )
    args = ap.parse_args()

    results = {}
    if "sqa3d" in args.tasks:
        results["sqa3d"] = task_vqa(
            "sqa3d",
            EVAL_INFO / "sqa3d/sqa3d_test_question.json",
            EVAL_INFO / "sqa3d/sqa3d_test_answer.json",
            "eval_sqa3d",
        )
    if "scanqa" in args.tasks:
        results["scanqa"] = task_vqa(
            "scanqa",
            EVAL_INFO / "scanqa/scanqa_val_question.jsonl",
            EVAL_INFO / "scanqa/scanqa_val_answer.jsonl",
            "eval_scanqa",
        )
    if "scan2cap" in args.tasks:
        results["scan2cap"] = task_scan2cap(
            EVAL_INFO / "densecap_scanrefer/scan2cap_mask3d_val.json",
            EVAL_INFO / "densecap_scanrefer/scan2cap_val_corpus.json",
            EVAL_INFO / "densecap_scanrefer/scannet_mask3d_val_attributes.pt",
            EVAL_INFO / "densecap_scanrefer/scannet_val_attributes.pt",
        )
    if "referseg_scanrefer" in args.tasks:
        results["referseg_scanrefer"] = task_referseg(
            "referseg_scanrefer",
            EVAL_INFO / "referseg_scanrefer/ScanRefer_filtered_val.json",
            with_suffix=False,
        )
    if "multi3drefer" in args.tasks:
        results["multi3drefer"] = task_referseg(
            "multi3drefer",
            EVAL_INFO / "multi3drefer/multi3drefer_val.json",
            with_suffix=True,
            native_spatial=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump = {}
    for task, task_res in results.items():
        dump[task] = {
            f"{run}|{bucket}": m
            for k, m in task_res.items()
            if isinstance(k, tuple)
            for run, bucket in [k]
        }
        dump[task]["meta"] = {k: v for k, v in task_res.items() if not isinstance(k, tuple)}
    (OUT_DIR / "parsed_metrics.json").write_text(json.dumps(dump, indent=2))
    print("\n=== parsed_metrics.json written ===")
    for task, task_res in results.items():
        print(f"\n===== {task} =====")
        meta = {k: v for k, v in task_res.items() if not isinstance(k, tuple)}
        print("meta:", meta)
        for k, m in task_res.items():
            if isinstance(k, tuple):
                head = {kk: round(v, 4) for kk, v in m.items() if kk not in ("n", "n_result", "n_corpus")}
                print(f"{k[0]:9s} {k[1]:16s} {head}")
    print("\nALL_OK")


if __name__ == "__main__":
    main()
