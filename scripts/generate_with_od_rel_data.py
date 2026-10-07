"""Generate `_with_od_rel` variants of the 3D-LLaVA train_info JSONs (EXP-OD4).

Full-scene injection like EXP-OD1, but each object line is reformatted with
explicit viewer-centered spatial relationships inferred purely geometrically
(scripts/od_spatial_relations.py -- no LLM):

    <pc>\\n<od>\\nobject: painting, position_xyz: (...), angle_z: ..., scale_xyz: (...), spatial relationship: above of table 1, ...\\nobject: table 1, ...\\n</od>\\n{original remainder}

Also writes eval-side files `od_exp4_rel.json` / `od_exp4_rel_val.json`
(consumable by the existing --od-file mechanism, zero eval-code changes).

Same policies as EXP-OD1/OD3: every conversation of every task gets the
block; 2 empty-OD scenes keep their original turn; only
conversations[0]["value"] is modified.

Usage (from repo root):
    python scripts/generate_with_od_rel_data.py [--skip-audit]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_INFO = os.path.join(REPO, "playground", "data", "train_info")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from od_spatial_relations import relational_od_text  # noqa: E402

PC = "<pc>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-audit", action="store_true", help="skip the token audit")
    args = ap.parse_args()

    # --- relational OD files (train + val) for the eval-side --od-file path ---
    cache = {}
    for src_name, dst_name in [
        ("od_exp4.json", "od_exp4_rel.json"),
        ("od_exp4_val.json", "od_exp4_rel_val.json"),
    ]:
        od = json.load(open(os.path.join(REPO, "playground", "data", src_name)))
        rel = {}
        for scene, text in od.items():
            if text not in cache:  # identical scene text across train/val reused
                cache[text] = relational_od_text(text)
            rel[scene] = cache[text]
        with open(os.path.join(REPO, "playground", "data", dst_name), "w") as f:
            json.dump(rel, f)
        sizes = [len(v) for v in rel.values() if v]
        print(f"{dst_name}: {len(rel)} scenes, chars/scene p50={int(np.percentile(sizes, 50))} max={max(sizes)}")

    od = json.load(open(os.path.join(REPO, "playground", "data", "od_exp4_rel.json")))

    total = n_od = n_empty = 0
    files = sorted(
        f for f in glob.glob(os.path.join(TRAIN_INFO, "*_3d_llava.json"))
        if "_with_click" not in f and "_with_od" not in f
    )
    for src in files:
        data = json.load(open(src))
        n_file_od = n_file_empty = 0
        for d in data:
            turn = d["conversations"][0]["value"]
            assert turn.startswith(PC), f"{os.path.basename(src)}: entry {d['id']} lacks <pc>\\n"
            block = od.get(d["scene_id"], "").strip()
            if block:
                d["conversations"][0]["value"] = (
                    "<pc>\n<od>\n" + block + "\n</od>\n" + turn[len(PC):]
                )
                n_file_od += 1
            else:
                n_file_empty += 1  # keep original turn
        dst = src.replace("_3d_llava.json", "_3d_llava_with_od_rel.json")
        with open(dst, "w") as f:
            json.dump(data, f)
        total += len(data); n_od += n_file_od; n_empty += n_file_empty
        print(f"{os.path.basename(dst)}: {len(data)} entries, {n_file_od} with od, {n_file_empty} kept original")

    print(f"\nTOTAL {total} entries | with-od {n_od} | empty {n_empty}")

    if not args.skip_audit:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            os.path.join(REPO, "checkpoints", "finetune-3d-llava-lora"))
        lens = []
        for src in files:
            data = json.load(open(src.replace("_3d_llava.json", "_3d_llava_with_od_rel.json")))
            texts = [d["conversations"][0]["value"] + d["conversations"][1]["value"] for d in data]
            for i in range(0, len(texts), 512):
                lens += [len(x) for x in tok(texts[i:i + 512], add_special_tokens=False)["input_ids"]]
        lens = np.array(lens)
        print(f"token audit [all 293k] p50={int(np.percentile(lens, 50))} "
              f"p95={int(np.percentile(lens, 95))} max={lens.max()}")
        assert lens.max() < 4096, f"max {lens.max()} exceeds model_max_length"
        d = json.load(open(files[0].replace("_3d_llava.json", "_3d_llava_with_od_rel.json")))[0]
        head = d["conversations"][0]["value"][:200].replace("\n", "\\n")
        print(f"spot-check: {head}")


if __name__ == "__main__":
    main()
