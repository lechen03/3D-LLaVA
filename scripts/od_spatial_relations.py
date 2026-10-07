"""Viewer-centered pairwise spatial-relation inference for EXP-OD4 (pure geometry).

Given a scene's OD text (bbox_N=Bbox(class, cx, cy, cz, angle_z, sx, sy, sz),
SpatialLM layout/entity.py:250 field order), reformat each object as

    object: {name}, position_xyz: (x, y, z), angle_z: a, scale_xyz: (sx, sy, sz), spatial relationship: left of table 1, in front of chair 2, ...

where the relations are inferred as if a person stood at the center of the
scene (center of the xy bounding box of all object centers) and looked at
the REFERENCED object B -- the one named after "of" (egocentric frame
anchored at B's sightline; reciprocal pairs are consistent, e.g.
"A left of B" <=> "B right of A"):

  left/right    lateral offset of A along the viewer's right vector
                r = d x z_hat  (d = unit(B - viewer), horizontal);
                offset > +0.15 m -> "right of B", < -0.15 -> "left of B"
  front/behind  depth along the sightline to B: s_A = (A - V).d vs
                s_B = |B - V|; s_A < s_B - 0.25 -> "in front of B",
                s_A > s_B + 0.25 -> "behind of B"
  above/below   A.z - B.z; > +0.15 -> "above of B", < -0.15 -> "below of B"

Coverage: each object relates to its K=6 nearest neighbors by center
distance (deterministic K-fuse 6 -> 4 -> 3 when the block would exceed the
token budget).  Duplicate classes are numbered per bbox order ("table 1",
"table 2"; single instances unnumbered).  No LLM involved anywhere.

Usage (self-test):
    python scripts/od_spatial_relations.py
"""

import math
import re

BBOX_RE = re.compile(r"^bbox_(\d+)=Bbox\((\w+),([^)]+)\)")

K_NEAREST = 6
K_FALLBACK = (6, 4, 3, 2, 1, 0)  # 0 = keep object lines only (mega scenes)
# calibrated with the real Llama tokenizer: tokens ~= chars * 0.397 median,
# 0.4624 worst case -> 6200 chars <= ~2.9k tokens (room for system prompt + long questions + answers) even at the worst ratio
EST_CHAR_BUDGET = 6200
LAT_DEAD = 0.15  # m, lateral dead-zone for left/right
DEPTH_DEAD = 0.25  # m, depth dead-zone for front/behind
Z_DEAD = 0.15  # m, vertical dead-zone for above/below


def parse_boxes(od_text):
    """-> [(bbox_id, cls, cx, cy, cz, angle_z, sx, sy, sz)] in file order."""
    out = []
    for line in od_text.strip().splitlines():
        m = BBOX_RE.match(line)
        if not m:
            continue
        floats = [float(x) for x in m.group(3).split(",")]
        assert len(floats) == 7, f"unexpected Bbox field count: {line}"
        out.append((int(m.group(1)), m.group(2), *floats))
    return out


def assign_names(boxes):
    """Number duplicate classes per bbox order -> ['table 1', 'table 2', 'chair', ...]."""
    counts = {}
    for _, cls, *_ in boxes:
        counts[cls] = counts.get(cls, 0) + 1
    seen, names = {}, []
    for _, cls, *_ in boxes:
        seen[cls] = seen.get(cls, 0) + 1
        names.append(f"{cls} {seen[cls]}" if counts[cls] > 1 else cls)
    return names


def _relations_for(a, b, viewer):
    """Phrases describing A as seen when the viewer looks at B."""
    (bid_a, _, ax, ay, az, *_), (bid_b, _, bx, by, bz, *_) = a, b
    dx, dy = bx - viewer[0], by - viewer[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:  # target at the viewer's feet: only vertical axis defined
        d = (0.0, 0.0)
    else:
        d = (dx / norm, dy / norm)
    # viewer's right vector when facing B: r = d_hat x z_hat = (dy, -dx)
    rx, ry = d[1], -d[0]
    rels = []
    lateral = (ax - viewer[0]) * rx + (ay - viewer[1]) * ry
    if lateral > LAT_DEAD:
        rels.append("right of")
    elif lateral < -LAT_DEAD:
        rels.append("left of")
    s_a = (ax - viewer[0]) * d[0] + (ay - viewer[1]) * d[1]
    s_b = (bx - viewer[0]) * d[0] + (by - viewer[1]) * d[1]
    if s_a < s_b - DEPTH_DEAD:
        rels.append("in front of")
    elif s_a > s_b + DEPTH_DEAD:
        rels.append("behind of")
    if az - bz > Z_DEAD:
        rels.append("above of")
    elif az - bz < -Z_DEAD:
        rels.append("below of")
    return rels


def _build_block(boxes, names, viewer, k):
    lines = []
    for i, (bid, cls, cx, cy, cz, ang, sx, sy, sz) in enumerate(boxes):
        others = [(j, b) for j, b in enumerate(boxes) if j != i]
        others.sort(key=lambda jb: (boxes[i][2] - jb[1][2]) ** 2 + (boxes[i][3] - jb[1][3]) ** 2 + (boxes[i][4] - jb[1][4]) ** 2)
        rels = []
        for j, b in others[:k]:
            for phrase in _relations_for(boxes[i], b, viewer):
                rels.append(f"{phrase} {names[j]}")
        lines.append(
            f"object: {names[i]}, position_xyz: ({cx:.2f}, {cy:.2f}, {cz:.2f}), "
            f"angle_z: {ang:.2f}, scale_xyz: ({sx:.2f}, {sy:.2f}, {sz:.2f}), "
            f"spatial relationship: {', '.join(rels) if rels else 'none'}"
        )
    return "\n".join(lines)


def relational_od_text(od_text):
    """Full relational OD block content (without <od> wrapper). Deterministic."""
    boxes = parse_boxes(od_text)
    if not boxes:
        return ""
    names = assign_names(boxes)
    xs = [b[2] for b in boxes]
    ys = [b[3] for b in boxes]
    viewer = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    for k in K_FALLBACK:
        block = _build_block(boxes, names, viewer, k)
        if len(block) <= EST_CHAR_BUDGET:
            return block
    return block  # K=3 result even if still large (audit will flag)


if __name__ == "__main__":
    import json
    import os

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --- synthetic scene: viewer at the origin, target "chair" due north ---
    # viewer = xy-center of all box centers = (0, 0) by symmetric placement
    synth = "\n".join([
        "bbox_0=Bbox(chair,0.0,2.0,0.0,0.0,0.5,0.5,0.9)",      # target (north)
        "bbox_1=Bbox(table,2.0,0.0,0.0,0.0,1.0,2.0,0.1)",      # east -> right of chair
        "bbox_2=Bbox(bed,-2.0,0.0,0.0,0.0,2.0,2.0,0.5)",       # west -> left of chair
        "bbox_3=Bbox(door,0.0,1.0,0.0,0.0,1.0,0.1,2.0)",       # near north -> in front of chair
        "bbox_4=Bbox(lamp,0.0,0.0,1.5,0.0,0.3,0.3,0.3)",       # overhead -> above of chair
        "bbox_5=Bbox(sofa,0.0,-2.0,0.0,0.0,2.0,1.0,0.8)",      # south -> behind of chair
    ])
    block = relational_od_text(synth)
    print("=== synthetic scene (viewer at origin, chair due north) ===")
    print(block, "\n")
    chair_line = next(l for l in block.splitlines() if l.startswith("object: chair"))
    rels = chair_line.split("spatial relationship: ")[1]
    # frame is anchored at the REFERENCED object (convention documented above):
    # looking at table (east), chair (north) is on the LEFT -> "left of table";
    # looking at bed (west), chair is on the RIGHT -> "right of bed"
    # door is NEARER than chair -> chair is "behind of door" (and door "in front of chair")
    # subject is the object itself: chair (z=0) is "below of lamp" (z=1.5)
    for expected in ["left of table", "right of bed", "behind of door", "below of lamp"]:
        assert expected in rels, f"synthetic assertion failed: '{expected}' not in: {rels}"
    assert "right of table" not in rels and "left of bed" not in rels
    # reciprocal consistency: table->"right of chair", door->"in front of chair"
    table_line = next(l for l in block.splitlines() if l.startswith("object: table"))
    assert "right of chair" in table_line.split("spatial relationship: ")[1]
    door_line = next(l for l in block.splitlines() if l.startswith("object: door"))
    assert "in front of chair" in door_line.split("spatial relationship: ")[1]
    lamp_line = next(l for l in block.splitlines() if l.startswith("object: lamp"))
    assert "above of chair" in lamp_line.split("spatial relationship: ")[1]
    print("synthetic assertions OK: left/right anchored at referenced object, reciprocals consistent\n")

    # --- two real scenes for eyeball inspection ---
    od = json.load(open(os.path.join(REPO, "playground", "data", "od_exp4.json")))
    for scene in list(od)[5:7]:
        print(f"=== {scene} ({len(parse_boxes(od[scene]))} boxes) ===")
        print(relational_od_text(od[scene]), "\n")
