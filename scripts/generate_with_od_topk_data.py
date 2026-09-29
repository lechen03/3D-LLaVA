"""Generate `_with_od_topk` variants of the 3D-LLaVA train_info JSONs (EXP-OD2).

Like generate_with_od_data.py (EXP-OD1, full-scene injection), but the OD
block is selected PER CONVERSATION by RAG-style class retrieval: the classes
mentioned-ish in the conversation's own question / referring description are
found via MiniLM cosine similarity (top-3 classes, scripts/od_retrieval.py),
and only those classes' bbox lines are injected.

- vqa / refer_seg conversations: selective injection (query extracted by
  byte-exact prefix stripping from the human turn).
- dense_captioning conversations: NO usable query (fixed template + <loc>
  placeholder) -> full-scene OD, byte-identical to EXP-OD1's `_with_od`
  output (asserted below).
- Empty-OD scenes keep their original turn (same policy as EXP-OD1).
- Only conversations[0]["value"] is modified; every other field is untouched.
- Source files are never modified. Outputs: `<name>_3d_llava_with_od_topk.json`.

Usage (from repo root, llava env or any env with torch+transformers):
    python scripts/generate_with_od_topk_data.py [--top-c 3] [--skip-audit]
"""

import argparse
import copy
import glob
import json
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_INFO = os.path.join(REPO, "playground", "data", "train_info")
OD_FILE = os.path.join(REPO, "playground", "data", "od_exp4.json")

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from od_retrieval import OdRetriever, parse_od_lines, round_od  # noqa: E402

PC = "<pc>\n"
REFSEG_PREFIX = " Please output the segmentation mask according to the following description. \n"
REFSEG_SUFFIX = "\nThere may be no corresponding object, or there may be one or more objects."


def extract_query(task_type, turn):
    """Query text for retrieval, or None when the task has no query."""
    rest = turn[len(PC):]
    if task_type == "vqa":
        return rest.lstrip()
    if task_type == "refer_seg":
        assert rest.startswith(REFSEG_PREFIX), f"unexpected refer_seg template: {rest[:80]!r}"
        desc = rest[len(REFSEG_PREFIX):]
        if desc.endswith(REFSEG_SUFFIX):
            desc = desc[: -len(REFSEG_SUFFIX)]
        return desc.strip()
    return None  # dense_captioning: fixed template, no query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-c", type=int, default=3)
    ap.add_argument("--skip-audit", action="store_true", help="skip the token audit")
    args = ap.parse_args()

    od = json.load(open(OD_FILE))
    retriever = OdRetriever()

    # --- pre-parse scenes and embed the closed class vocabulary once ---
    scene_lines = {k: parse_od_lines(v) for k, v in od.items() if v.strip()}
    all_classes = sorted({c for lines in scene_lines.values() for _, c, _ in lines})
    cls_emb = {c: e for c, e in zip(all_classes, retriever.embed(all_classes))}
    print(f"scenes with OD: {len(scene_lines)}, class vocab: {len(all_classes)} {all_classes}")

    # --- pass 1: collect every unique query across all files ---
    files = sorted(
        f for f in glob.glob(os.path.join(TRAIN_INFO, "*_3d_llava.json"))
        if "_with_click" not in f and "_with_od" not in f
    )
    src_data, queries = {}, set()
    for src in files:
        data = json.load(open(src))
        src_data[src] = data
        for d in data:
            q = extract_query(d["task_type"], d["conversations"][0]["value"])
            if q:
                queries.add(q)
    uniq = sorted(queries)
    print(f"unique queries to embed: {len(uniq)}")
    q_emb = {q: e for q, e in zip(uniq, retriever.embed(uniq))}

    def select_text(scene_id, query):
        """Selected OD text for one conversation, or None to keep the original."""
        lines = scene_lines.get(scene_id)
        if not lines:
            return None
        if query is None:  # dense_captioning fallback: full scene, EXP-OD1-identical
            return round_od(od[scene_id].strip())
        classes = sorted({c for _, c, _ in lines})
        scores = {c: float(np.dot(cls_emb[c], q_emb[query])) for c in classes}
        keep = set(sorted(classes, key=lambda c: -scores[c])[: args.top_c])
        return "\n".join(round_od(l) for i, c, l in lines if c in keep)

    # --- pass 2: write outputs ---
    stats = {}
    total = n_od = n_empty = n_full = n_selective = 0
    for src in files:
        base = os.path.basename(src)
        data = src_data[src]
        out = []
        file_stats = {"n": len(data), "od": 0, "empty": 0, "full": 0, "selective": 0,
                      "lines_kept": 0, "lines_total": 0}
        for d in data:
            d2 = copy.deepcopy(d)
            turn = d2["conversations"][0]["value"]
            assert turn.startswith(PC), f"{base}: entry {d2['id']} lacks <pc>\\n prefix"
            query = extract_query(d2["task_type"], turn)
            lines = scene_lines.get(d2["scene_id"], [])
            text = select_text(d2["scene_id"], query)
            if text is None:
                file_stats["empty"] += 1
            else:
                d2["conversations"][0]["value"] = "<pc>\n<od>\n" + text + "\n</od>\n" + turn[len(PC):]
                file_stats["od"] += 1
                file_stats["full" if query is None else "selective"] += 1
                file_stats["lines_total"] += len(lines)
                file_stats["lines_kept"] += len(parse_od_lines(text))
            out.append(d2)
        dst = src.replace("_3d_llava.json", f"_3d_llava_with_od_topk{'' if args.top_c == 3 else str(args.top_c)}.json")
        with open(dst, "w") as f:
            json.dump(out, f)
        stats[base] = file_stats
        total += file_stats["n"]; n_od += file_stats["od"]; n_empty += file_stats["empty"]
        n_full += file_stats["full"]; n_selective += file_stats["selective"]
        print(f"{os.path.basename(dst)}: {file_stats['n']} entries, "
              f"{file_stats['od']} with od ({file_stats['selective']} selective + {file_stats['full']} full), "
              f"{file_stats['empty']} kept original")

    kept = sum(s["lines_kept"] for s in stats.values())
    tot = sum(s["lines_total"] for s in stats.values())
    print(f"\nTOTAL {total} entries | with-od {n_od} (selective {n_selective} + full {n_full}) | "
          f"empty {n_empty} | bbox-line retention {kept}/{tot} = {kept / max(tot, 1):.1%}")
    json.dump(stats, open(os.path.join(REPO, "playground", "data", "od_topk_stats.json"), "w"), indent=2)

    # --- verification: dense_captioning entries byte-identical to EXP-OD1 ---
    n_checked = 0
    for src in files:
        base = os.path.basename(src)
        if stats[base]["full"] == 0:
            continue
        with_od = json.load(open(src.replace("_3d_llava.json", "_3d_llava_with_od.json")))
        topk = json.load(open(src.replace("_3d_llava.json", "_3d_llava_with_od_topk.json")))
        for a, b in zip(with_od, topk):
            if a["task_type"] == "dense_captioning":
                assert a["conversations"][0]["value"] == b["conversations"][0]["value"], \
                    f"{base}: dense_captioning mismatch at id {a['id']}"
                n_checked += 1
    print(f"dense_captioning byte-identity vs _with_od: {n_checked} entries OK")

    # --- token audit on the selectively-injected tasks ---
    if not args.skip_audit:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            os.path.join(REPO, "checkpoints", "finetune-3d-llava-lora"))
        per_task = {}
        for src in files:
            base = os.path.basename(src)
            topk = json.load(open(src.replace("_3d_llava.json", "_3d_llava_with_od_topk.json")))
            for d in topk:
                if d["task_type"] == "dense_captioning":
                    continue
                per_task.setdefault(d["task_type"], []).append(
                    d["conversations"][0]["value"] + d["conversations"][1]["value"])
        for task, texts in per_task.items():
            lens = []
            for i in range(0, len(texts), 512):
                lens += [len(x) for x in tok(texts[i:i + 512], add_special_tokens=False)["input_ids"]]
            lens = np.array(lens)
            print(f"token audit [{task}] n={len(lens)} p50={int(np.percentile(lens, 50))} "
                  f"p95={int(np.percentile(lens, 95))} max={lens.max()}")
            assert lens.max() < 4096, f"{task}: exceeds model_max_length"


if __name__ == "__main__":
    main()
