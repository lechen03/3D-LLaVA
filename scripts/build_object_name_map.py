"""Map `object_id` -> object name for the 3D-LLaVA referring-expression datasets.

`object_id` in nr3d / multi3drefer / scanrefer is NOT a name lookup key — it is
the per-point *instance index* inside a scene's instance mask
(`raw_data['instance_gt'] == object_id`), so the same integer means a different
object in every scene. The object *name* is recovered from that instance's
dominant `semantic_gt200` class, which this repo stores as the positional index
into the ScanNet200 `CLASS_LABELS_200` list (verified: scene0525_00 oid=9 ->
class 34 -> 'plant', matching the description "The plant ...").

This script:
  1. Collects the union of (scene_id, object_id) pairs across the requested
     datasets. Handles both schemas:
       - train / nr3d / scanrefer: `object_id` (int)
       - train multi3drefer:       `object_id: [int, ...]`
       - multi3drefer_val:         `object_ids: [int, ...]`
       - ScanRefer_filtered_val:   `object_id: "int"` (string)
     Empty-target entries (`[]`) contribute no pairs.
  2. Loads each scene .pth exactly once and, in one vectorised pass, computes
     the majority `semantic_gt200` class for every instance in that scene.
  3. Writes a per-dataset mapping JSON (+ a combined `all_*`) of the form
     {"_meta": {...}, "scenes": {scene_id: {"9": "plant", ...}}}.
  4. For datasets that ship a native `object_name` (the val files), also writes a
     `<ds>_native_vs_pth.json` cross-check comparing the native label against the
     derived ScanNet200 name.

Usage:
  python scripts/build_object_name_map.py                          # all datasets
  python scripts/build_object_name_map.py --datasets multi3dref_val scanrefer_val
"""

import argparse
import json
import pathlib
import re
from collections import Counter

import numpy as np
import torch

# ---------------------------------------------------------------------------
# ScanNet200 label table — positional index into this tuple == the value stored
# in each point's `semantic_gt200`. Sourced verbatim from the official ScanNet
# BenchmarkScripts/ScanNet200/scannet200_constants.py (CLASS_LABELS_200).
# ---------------------------------------------------------------------------
CLASS_LABELS_200 = (
    'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf',
    'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window',
    'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair',
    'coffee table', 'box', 'refrigerator', 'lamp', 'kitchen cabinet', 'towel',
    'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion',
    'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard',
    'bag', 'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard',
    'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave',
    'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench',
    'board', 'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
    'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
    'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard',
    'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
    'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand',
    'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar',
    'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder',
    'bathroom stall', 'shower wall', 'cup', 'jacket', 'storage bin',
    'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat',
    'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board',
    'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
    'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball',
    'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray',
    'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse',
    'toilet seat cover dispenser', 'furniture', 'cart', 'storage container',
    'scale', 'tissue box', 'light switch', 'crate', 'power outlet',
    'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner',
    'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack', 'broom',
    'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle',
    'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher',
    'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand',
    'projector screen', 'divider', 'laundry detergent', 'bathroom counter',
    'object', 'bathroom vanity', 'closet wall', 'laundry hamper',
    'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell',
    'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
    'coffee kettle', 'structure', 'shower head', 'keyboard piano',
    'case of water bottles', 'coat rack', 'storage organizer', 'folded chair',
    'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant',
    'luggage', 'mattress',
)
assert len(CLASS_LABELS_200) == 200, f"expected 200 labels, got {len(CLASS_LABELS_200)}"

NOT_FOUND = "__not_found__"          # requested oid absent from instance_gt
UNLABELED = "__unlabeled__"          # oid exists but all its points have sem<0
OUT_OF_RANGE = "__class_out_of_range__"  # majority semantic_gt200 >= 200 or < 0

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "playground" / "data" / "scannet" / "train"
VAL_DIR = REPO_ROOT / "playground" / "data" / "scannet" / "val"
TRAIN_OUT = REPO_ROOT / "playground" / "data" / "train_info" / "object_name_maps"
EVAL_OUT = REPO_ROOT / "playground" / "data" / "eval_info" / "object_name_maps"

# name -> {"path": source json, "out": output dir, "native": field name of a
# native object_name carried by the file (used only for the cross-check), or None}
DATASETS = {
    "nr3d":           {"path": REPO_ROOT / "playground" / "data" / "train_info" / "nr3d_train_3d_llava.json", "out": TRAIN_OUT, "native": None},
    "multi3dref":     {"path": REPO_ROOT / "playground" / "data" / "train_info" / "multi3drefer_train_3d_llava.json", "out": TRAIN_OUT, "native": None},
    "scanrefer":      {"path": REPO_ROOT / "playground" / "data" / "train_info" / "scanrefer_train_3d_llava.json", "out": TRAIN_OUT, "native": None},
    "multi3dref_val": {"path": REPO_ROOT / "playground" / "data" / "eval_info" / "multi3drefer" / "multi3drefer_val.json", "out": EVAL_OUT, "native": "object_name"},
    "scanrefer_val":  {"path": REPO_ROOT / "playground" / "data" / "eval_info" / "referseg_scanrefer" / "ScanRefer_filtered_val.json", "out": EVAL_OUT, "native": "object_name"},
}


def collect_pairs(entries):
    """Collect (scene_id, object_id) pairs across all entries.

    Reads `object_ids` (plural, list) if present, else `object_id` (int or str
    or list). Returns (pairs:set, n_empty:int). Empty-target entries contribute
    no pairs but are counted.
    """
    pairs = set()
    n_empty = 0
    for e in entries:
        v = e.get("object_ids", e.get("object_id"))
        oids = v if isinstance(v, list) else [v]
        if len(oids) == 0:
            n_empty += 1
            continue
        for o in oids:
            pairs.add((e["scene_id"], int(o)))
    return pairs, n_empty


def resolve_scene(pth_or_oid_pairs):
    """Load one scene and return {object_id: name} for every requested oid.

    Computes the majority semantic_gt200 class for ALL instances in the scene in
    a single vectorised pass, then looks up the requested object_ids. Points with
    a negative class label (unannotated) are excluded from the majority vote; an
    instance whose every point is unlabeled is reported as UNLABELED.
    """
    scene_id, oids = pth_or_oid_pairs

    # Locate the .pth under train/ then val/.
    pth = TRAIN_DIR / f"{scene_id}.pth"
    split = "train"
    if not pth.is_file():
        pth = VAL_DIR / f"{scene_id}.pth"
        split = "val"
    if not pth.is_file():
        return split, {o: NOT_FOUND for o in oids}, True  # scene missing

    raw = torch.load(str(pth), map_location="cpu", weights_only=False)
    inst = np.asarray(raw["instance_gt"]).ravel()
    sem = np.asarray(raw["semantic_gt200"]).ravel()

    # One-pass majority-class over every instance:
    #   bincount over (instance * K + class), reshape, argmax down axis-1.
    n_cls = max((int(sem.max()) + 1) if sem.size else 1, len(CLASS_LABELS_200))
    valid = (inst >= 0) & (sem >= 0)
    iv, sv = inst[valid], sem[valid]
    if iv.size:
        comb = iv.astype(np.int64) * n_cls + sv.astype(np.int64)
        counts = np.bincount(comb, minlength=(int(iv.max()) + 1) * n_cls).reshape(-1, n_cls)
        present_labeled = set(iv.tolist())            # instances with >=1 labeled point
    else:
        counts = np.zeros((0, n_cls), dtype=np.int64)
        present_labeled = set()
    present_any = set(inst[inst >= 0].tolist()) if inst.size else set()  # exists at all

    out = {}
    for o in oids:
        if o not in present_any:
            out[o] = NOT_FOUND
        elif o not in present_labeled:
            out[o] = UNLABELED
        else:
            cls = int(counts[o].argmax())
            out[o] = CLASS_LABELS_200[cls] if 0 <= cls < len(CLASS_LABELS_200) else OUT_OF_RANGE
    return split, out, False  # (split, {oid: name}, missing_scene)


def _worker(args):
    return resolve_scene(args)


# --- native object_name cross-check (ScanRefer/Multi3DRefer val files carry one) ---
# The val files label each target with a coarser NYU40-style name; we compare it
# to the derived ScanNet200 name. Synonyms bridge the two vocabularies.
_NATIVE_SYN = {
    "sofa": "couch", "couches": "couch", "sofas": "couch",
    "garbagebin": "trash can", "garbage can": "trash can", "garbage bin": "trash can",
    "trashcan": "trash can",
    "television": "tv", "televisions": "tv", "t v": "tv",
    "bookcase": "bookshelf", "bookcases": "bookshelf",
    "fridge": "refrigerator",
    "armchair": "arm chair",
}


def _norm_word(w):
    w = w.lower().replace("_", " ").strip()
    w = re.sub(r"[^a-z ]", "", w).strip()
    if w.endswith("ies") and len(w) > 3:
        w = w[:-3] + "y"
    elif w.endswith(("ses", "xes", "zes", "ches", "shes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return _NATIVE_SYN.get(w, w)


def native_agrees(native, pth):
    """Approximate match between the coarser native name and the ScanNet200 name.

    Returns True/False, or None when the comparison is meaningless (no native
    label, or pth is a sentinel)."""
    if not native or not isinstance(native, str):
        return None
    if not pth or pth.startswith("__"):
        return None
    nn, pn = _norm_word(native), _norm_word(pth)
    if nn == pn:
        return True
    # ScanNet200 is finer grained: the native name is usually the trailing noun.
    if pn.endswith(" " + nn) or nn.endswith(" " + pn):
        return True
    return False


def native_lookup(entries, native_field):
    """Map (scene_id, object_id) -> native object_name string."""
    out = {}
    for e in entries:
        native = e.get(native_field)
        v = e.get("object_ids", e.get("object_id"))
        oids = v if isinstance(v, list) else [v]
        for o in oids:
            if o is None:
                continue
            out[(e["scene_id"], int(o))] = native
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS),
                    choices=list(DATASETS), help="which datasets to build maps for")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel scene-loading workers (1 = sequential)")
    ap.add_argument("--out-dir", default=None,
                    help="force all outputs into this dir (default: each dataset's own dir)")
    args = ap.parse_args()

    force_out = pathlib.Path(args.out_dir) if args.out_dir else None

    per_dataset = {}
    all_pairs = set()
    for ds in args.datasets:
        entries = json.load(open(DATASETS[ds]["path"], encoding="utf-8"))
        pairs, n_empty = collect_pairs(entries)
        native = None
        if DATASETS[ds]["native"]:
            native = native_lookup(entries, DATASETS[ds]["native"])
        per_dataset[ds] = {"entries": len(entries), "pairs": pairs, "n_empty": n_empty, "native": native}
        all_pairs |= pairs
        print(f"[{ds}] {len(entries):,} entries, {len(pairs):,} unique (scene,oid) pairs"
              f", {n_empty:,} empty-target entries"
              + (f"  [native: {DATASETS[ds]['native']}]" if DATASETS[ds]["native"] else ""))

    # Group requested object_ids by scene across ALL datasets (load each scene once).
    by_scene = {}
    for scene_id, oid in all_pairs:
        by_scene.setdefault(scene_id, set()).add(oid)
    print(f"\nResolving {len(all_pairs):,} unique pairs across {len(by_scene)} scenes "
          f"({args.workers} worker(s))...")

    work = [(s, sorted(o for o in by_scene[s])) for s in sorted(by_scene)]
    scene_result = {}  # scene_id -> {oid: name}
    n_missing_scenes = 0
    split_counts = Counter()

    if args.workers <= 1:
        for scene_id, oids in work:
            split, names, missing = resolve_scene((scene_id, oids))
            scene_result[scene_id] = names
            n_missing_scenes += missing
            split_counts[split] += 1
            if missing:
                print(f"  WARNING: scene {scene_id} not found in train/ or val/")
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for (scene_id, oids), (split, names, missing) in zip(work, pool.imap(_worker, work)):
                scene_result[scene_id] = names
                n_missing_scenes += missing
                split_counts[split] += 1
                if missing:
                    print(f"  WARNING: scene {scene_id} not found in train/ or val/")

    def build_map(pairs):
        scenes = {}
        n_found = n_not = 0
        for scene_id, oid in pairs:
            name = scene_result[scene_id][oid]
            scenes.setdefault(scene_id, {})[str(oid)] = name
            if name == NOT_FOUND:
                n_not += 1
            else:
                n_found += 1
        return scenes, n_found, n_not

    # Per-dataset maps (+ native cross-check where available) + a combined union map.
    combined_pairs = set()
    for ds in args.datasets:
        d_out = (force_out or DATASETS[ds]["out"])
        d_out.mkdir(parents=True, exist_ok=True)
        scenes, n_found, n_not = build_map(per_dataset[ds]["pairs"])
        combined_pairs |= per_dataset[ds]["pairs"]
        doc = {
            "_meta": {
                "dataset": ds,
                "source_file": str(DATASETS[ds]["path"].relative_to(REPO_ROOT)),
                "name_source": "majority semantic_gt200 class per instance -> ScanNet200 CLASS_LABELS_200",
                "n_entries": per_dataset[ds]["entries"],
                "n_empty_target_entries": per_dataset[ds]["n_empty"],
                "n_unique_scene_oid_pairs": len(per_dataset[ds]["pairs"]),
                "n_found": n_found,
                "n_not_found": n_not,
                "n_scenes": len(scenes),
            },
            "scenes": scenes,
        }
        p = d_out / f"{ds}_object_id_to_name.json"
        json.dump(doc, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  wrote {p.relative_to(REPO_ROOT)}  (found {n_found}, not_found {n_not})")

        # native-vs-pth cross-check companion
        if per_dataset[ds]["native"]:
            pairs_block = {}
            agree = disagree = skipped = 0
            disagree_patterns = Counter()
            for scene_id, oid in per_dataset[ds]["pairs"]:
                pth = scene_result[scene_id][oid]
                native = per_dataset[ds]["native"].get((scene_id, oid))
                g = native_agrees(native, pth)
                if g is None:
                    skipped += 1
                elif g:
                    agree += 1
                else:
                    disagree += 1
                    disagree_patterns[(_norm_word(native), pth)] += 1
                pairs_block.setdefault(scene_id, {})[str(oid)] = {"native": native, "pth": pth, "agree": g}
            denom = agree + disagree
            doc2 = {
                "_meta": {
                    "dataset": ds,
                    "native_field": DATASETS[ds]["native"],
                    "n_pairs": len(per_dataset[ds]["pairs"]),
                    "n_agree": agree,
                    "n_disagree": disagree,
                    "n_skipped": skipped,
                    "agreement_rate": round(agree / denom, 4) if denom else None,
                    "top_disagreement_patterns": disagree_patterns.most_common(15),
                },
                "scenes": pairs_block,
            }
            p2 = d_out / f"{ds}_native_vs_pth.json"
            json.dump(doc2, open(p2, "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)
            print(f"    native-vs-pth agreement: {agree}/{denom} "
                  f"({100*agree/max(denom,1):.1f}%), {disagree} disagree -> {p2.relative_to(REPO_ROOT)}")

    union_out = force_out or DATASETS[args.datasets[0]]["out"]
    union_out.mkdir(parents=True, exist_ok=True)
    scenes, n_found, n_not = build_map(combined_pairs)
    doc = {
        "_meta": {
            "dataset": "all (" + " + ".join(args.datasets) + ")",
            "name_source": "majority semantic_gt200 class per instance -> ScanNet200 CLASS_LABELS_200",
            "n_unique_scene_oid_pairs": len(combined_pairs),
            "n_found": n_found,
            "n_not_found": n_not,
            "n_scenes": len(scenes),
            "n_missing_scenes": n_missing_scenes,
            "scene_split_counts": dict(split_counts),
        },
        "scenes": scenes,
    }
    p = union_out / "all_object_id_to_name.json"
    json.dump(doc, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)
    print(f"  wrote {p.relative_to(REPO_ROOT)}  (found {n_found}, not_found {n_not}, "
          f"{n_missing_scenes} missing scenes)")

    print("\nDone.")


if __name__ == "__main__":
    main()
