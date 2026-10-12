"""Generate `_with_od_gt` variants of the 3D-LLaVA train_info JSONs (EXP-OD5).

Diagnostic ablation: inject GROUND-TRUTH bounding boxes (the SpatialLM
training labels, 18-class vocab, same frame as the EXP4 predictions) instead
of detector output, in the named format WITHOUT relations:

    <pc>\\n<od>\\nobject: painting, position_xyz: (0.68, 1.86, 1.80), angle_z: -1.40, scale_xyz: (0.72, 0.06, 0.70)\\nobject: table 1, ...\\n</od>\\n{remainder}

Reading: GT still not better than EXP-OD1 -> box-level info is hard for the
LLM to use (explanation A); GT clearly better -> the detector is the
limiting factor (explanation B).

GT source: /home/chenle/SpatialLM/data/scannet_spatiallm/layout/{scene}.txt
(all 1513 scenes; bbox_N=Bbox(class, cx, cy, cz, angle_z, sx, sy, sz);
wall/floor/unlabeled/tiny already excluded; produced by prepare_scannet.py).

Same policies as EXP-OD1/OD3: every conversation of every task gets the
block (GT is non-empty everywhere); only conversations[0]["value"] changes.

Usage (from repo root):
    python scripts/generate_with_od_gt_data.py [--skip-audit]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_INFO = os.path.join(REPO, "playground", "data", "train_info")
LAYOUT_DIR = "/home/chenle/SpatialLM/data/scannet_spatiallm/layout"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from od_spatial_relations import assign_names, parse_boxes  # noqa: E402

PC = "<pc>\n"


def gt_named_text(layout_text):
    """Reformat GT layout lines to the named per-object format (no relations)."""
    boxes = parse_boxes(layout_text)
    if not boxes:
        return ""
    names = assign_names(boxes)
    lines = []
    for name, (_, _, cx, cy, cz, ang, sx, sy, sz) in zip(names, boxes):
        lines.append(
            f"object: {name}, position_xyz: ({cx:.2f}, {cy:.2f}, {cz:.2f}), "
            f"angle_z: {ang:.2f}, scale_xyz: ({sx:.2f}, {sy:.2f}, {sz:.2f})"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-audit", action="store_true", help="skip the token audit")
    args = ap.parse_args()

    # --- GT od files keyed like the EXP4 ones (train 1114 / val 312) ---
    od_gt = {}
    for src_name, dst_name, scene_src in [
        ("od_exp4.json", "od_gt.json", "od_exp4.json"),
        ("od_exp4_val.json", "od_gt_val.json", "od_exp4_val.json"),
    ]:
        scenes = json.load(open(os.path.join(REPO, "playground", "data", scene_src)))
        out = {}
        for scene in scenes:
            layout_path = os.path.join(LAYOUT_DIR, f"{scene}.txt")
            assert os.path.exists(layout_path), f"missing layout for {scene}"
            out[scene] = gt_named_text(open(layout_path).read())
        with open(os.path.join(REPO, "playground", "data", dst_name), "w") as f:
            json.dump(out, f)
        od_gt[dst_name] = out
        counts = [len(parse_boxes(open(os.path.join(LAYOUT_DIR, f"{s}.txt")).read())) for s in scenes]
        empty = sum(1 for s in scenes if not out[s])
        print(f"{dst_name}: {len(out)} scenes ({empty} empty), boxes/scene "
              f"p50={int(np.percentile(counts, 50))} max={max(counts)}")

    od = od_gt["od_gt.json"]

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
        dst = src.replace("_3d_llava.json", "_3d_llava_with_od_gt.json")
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
            data = json.load(open(src.replace("_3d_llava.json", "_3d_llava_with_od_gt.json")))
            texts = [d["conversations"][0]["value"] + d["conversations"][1]["value"] for d in data]
            for i in range(0, len(texts), 512):
                lens += [len(x) for x in tok(texts[i:i + 512], add_special_tokens=False)["input_ids"]]
        lens = np.array(lens)
        print(f"token audit [all 293k] p50={int(np.percentile(lens, 50))} "
              f"p95={int(np.percentile(lens, 95))} max={lens.max()}")
        assert lens.max() < 4096, f"max {lens.max()} exceeds model_max_length"
        d = json.load(open(files[0].replace("_3d_llava.json", "_3d_llava_with_od_gt.json")))[0]
        head = d["conversations"][0]["value"][:200].replace("\n", "\\n")
        print(f"spot-check: {head}")


if __name__ == "__main__":
    main()
