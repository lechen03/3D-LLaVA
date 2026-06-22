"""Standalone hover-to-inspect point-cloud viewer for preprocessed ScanNet.

This is meant to be launched as its own process (see launch() in the
VisualizePointcloud notebook). Running it as a plain `python` process — not
inside an IPython/Jupyter kernel — avoids Open3D's automatic "WebRTC Jupyter
handshake mode", which otherwise blocks forever waiting for a browser client
that never connects. Here Open3D opens a normal native window instead.

Usage:
    python hover_pointcloud_viewer.py <scene_path_or_id> [color_mode] [point_size] [--split train|val] [--mask masks/<scene>__c0_q0000.npz]

    color_mode: rgb | semantic | instance | superpoint | tpfpfn | pred | gt   (default: rgb; tpfpfn when --mask given)
    point_size: render point size in px       (default: 4.0)
    --split  : restrict a scene_id lookup to 'train' or 'val' (default: search both)
    --mask   : path to a per-question mask .npz (pred_mask, gt_mask) saved by the
               refer-seg eval scripts. Adds the tpfpfn/pred/gt color modes and a
               per-point seg (TP/FP/FN/TN) line in the HUD.

Move the mouse over the cloud: the front-most point under the cursor has its
coord / color / normal / semantic_gt20 / semantic_gt200 / instance_gt /
superpoint shown in the top-right corner, and its whole superpoint is
highlighted (drawn as a larger, brighter cluster on top of the cloud).

With --mask, color modes show refer-seg results:
    tpfpfn -> green=TP (pred∩gt), red=FP (false pos), blue=FN (false neg), gray=rest
    pred   -> orange where the model predicted the object
    gt     -> teal-green where the ground truth object is
"""

import colorsys
import pathlib
import sys

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch


# Standard ScanNet20 benchmark class names (index -> name; -1 = unlabeled).
SCANNET20_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
]


def _scannet_root():
    """Locate playground/data/scannet (containing train/ and val/) by walking up."""
    for base in [pathlib.Path.cwd(), pathlib.Path.cwd().parent, *pathlib.Path.cwd().parents]:
        d = base / "playground" / "data" / "scannet"
        if d.is_dir():
            return d
    return pathlib.Path("playground/data/scannet")


def _resolve_scene(pth_path, split=None):
    """Resolve a scene to an absolute .pth path and report which split it is in.

    - If pth_path ends in '.pth' it is used directly (split inferred from path).
    - Otherwise it is treated as a scene_id and searched under train/ then val/
      (or only the requested split when `split` is given).
    Returns (absolute_path, split_or_None). Raises FileNotFoundError if missing.
    """
    p = pathlib.Path(pth_path)
    if p.suffix == ".pth":
        ap = p.resolve()
        used = ap.parts[-2] if len(ap.parts) >= 2 and ap.parts[-2] in ("train", "val") else None
        return ap, used

    root = _scannet_root()
    splits = [split] if split else ["train", "val"]
    for s in splits:
        cand = root / s / f"{p}.pth"
        if cand.is_file():
            return cand.resolve(), s
    searched = ", ".join(str(root / s / f"{p}.pth") for s in splits)
    raise FileNotFoundError(f"scene '{p}' not found; searched: {searched}")


def _distinct_palette(n, seed=0):
    """Return n visually separable RGB colors in [0, 1] (HSV hue sweep)."""
    rng = np.random.RandomState(seed)
    cols = []
    for i in range(n):
        h = (i / max(n, 1)) % 1.0
        s = 0.7 + 0.3 * rng.rand()
        v = 0.85 + 0.15 * rng.rand()
        cols.append(colorsys.hsv_to_rgb(h, s, v))
    return np.asarray(cols, dtype=np.float64)


def load_scannet_scene(pth_path, weights_only=False):
    """Load a preprocessed ScanNet .pth scene -> plain numpy dict.

    Keys: coord(N,3) color(N,3 in [0,255]) normal(N,3)
          semantic_gt20(N,) semantic_gt200(N,) instance_gt(N,) scene_id(str)
    """
    data = torch.load(str(pth_path), map_location="cpu", weights_only=weights_only)
    out = {k: (np.asarray(v) if not isinstance(v, str) else v) for k, v in data.items()}
    out["coord"] = out["coord"].astype(np.float64)
    out["color"] = out["color"].astype(np.float64)
    return out


def _make_colors(scene, color_mode):
    if color_mode == "rgb":
        return np.clip(scene["color"] / 255.0, 0.0, 1.0)
    if color_mode == "semantic":
        labels = scene["semantic_gt20"]
        palette = _distinct_palette(20)
        colors = palette[np.clip(labels, 0, 19)]
        colors[labels < 0] = 0.0
        return colors
    if color_mode == "instance":
        labels = scene["instance_gt"]
        ids = np.unique(labels[labels >= 0])
        palette = _distinct_palette(len(ids))
        lut = dict(zip(ids.tolist(), palette))
        colors = np.zeros((len(labels), 3))
        for i, c in lut.items():
            colors[labels == i] = c
        return colors
    if color_mode == "superpoint":
        if "superpoint" not in scene:
            raise ValueError("color_mode='superpoint' needs a super_points .bin; none was loaded")
        sp = scene["superpoint"]
        ids = np.unique(sp)
        palette = _distinct_palette(len(ids))
        # shuffle so neighboring superpoints (often consecutive ids) get
        # different hues; otherwise adjacent patches share a near-identical color
        palette = palette[np.random.RandomState(12345).permutation(len(ids))]
        lut = dict(zip(ids.tolist(), palette))
        colors = np.zeros((len(sp), 3))
        for i, c in lut.items():
            colors[sp == i] = c
        return colors
    if color_mode in ("tpfpfn", "pred", "gt"):
        return _seg_colors(scene, color_mode)
    raise ValueError(f"unknown color_mode: {color_mode!r}")


def _seg_colors(scene, color_mode):
    """Color points by refer-seg correctness against pred_mask / gt_mask.

    tpfpfn: green=TP (pred∩gt), red=FP (false positive), blue=FN (false negative),
            dark gray=the rest (true negatives).
    pred:   orange where the model predicted the object; rest dark gray.
    gt:     teal-green where the ground-truth object is; rest dark gray.
    """
    pred = scene.get("pred_mask")
    gt = scene.get("gt_mask")
    if pred is None or gt is None:
        raise ValueError(f"color_mode={color_mode!r} needs pred_mask/gt_mask; none loaded")
    n = len(scene["coord"])
    colors = np.full((n, 3), 0.12, dtype=np.float64)
    if color_mode == "pred":
        colors[pred] = [1.0, 0.65, 0.1]
        return colors
    if color_mode == "gt":
        colors[gt] = [0.1, 0.9, 0.4]
        return colors
    tp = pred & gt
    fp = pred & (~gt)
    fn = (~pred) & gt
    colors[tp] = [0.0, 0.9, 0.0]
    colors[fp] = [0.95, 0.1, 0.1]
    colors[fn] = [0.1, 0.4, 1.0]
    return colors


class HoverPointCloudViewer:
    """Open3D GUI viewer that shows the hovered point's attributes top-right.

    Picking projects all points to the screen with the current camera matrices
    (cached per camera pose so it stays cheap), then takes the front-most point
    within a few pixels of the cursor.
    """

    HUD_WIDTH = 330
    PICK_RADIUS = 12.0  # screen pixels

    def __init__(self, title, pcd, scene_data, point_size, color_mode="rgb"):
        self.points = np.asarray(pcd.points, dtype=np.float64)  # as rendered (centered)
        self.data = scene_data                                  # original arrays, index-aligned
        self.sp = scene_data.get("superpoint")                 # int64 (N,) or None
        self.color_mode = color_mode                           # active color mode
        self.highlight_enabled = False                         # hover superpoint highlight on/off
        self._cam_key = None
        self._screen_wh = None
        self._sx = self._sy = self._camz = None
        self.highlighted_sp = None                             # currently highlighted superpoint id

        self.app = gui.Application.instance
        self.app.initialize()
        self.win = self.app.create_window(title, 1600, 900)
        self._build(pcd, point_size, color_mode)

    def _build(self, pcd, point_size, color_mode):
        self.pcd = pcd
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.win.renderer)
        self.scene.scene.set_background(  # black background
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        self.base_colors = np.asarray(pcd.colors, dtype=np.float64).copy()
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"  # use per-point vertex colors
        mat.point_size = point_size
        self.mat = mat
        self.scene.scene.add_geometry("points", pcd, mat)
        # overlay material for the highlighted superpoint (larger points)
        self.mat_overlay = rendering.MaterialRecord()
        self.mat_overlay.shader = "defaultUnlit"
        self.mat_overlay.point_size = point_size + 4.0
        self.scene.set_on_mouse(self._on_mouse)
        bounds = pcd.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60.0, bounds, bounds.get_center())

        # ---- top-right HUD overlay (light text on the black background) ----
        self.panel = gui.Vert(2, gui.Margins(8, 8, 8, 8))
        hud_color = gui.Color(1.0, 1.0, 1.0, 1.0)  # white
        self.hud_title = gui.Label("Hover a point")
        self.lbl_coord = gui.Label("")
        self.lbl_color = gui.Label("")
        self.lbl_normal = gui.Label("")
        self.lbl_sem20 = gui.Label("")
        self.lbl_sem200 = gui.Label("")
        self.lbl_instance = gui.Label("")
        self.lbl_superpoint = gui.Label("")
        self.lbl_seg = gui.Label("")
        for lbl in (self.hud_title, self.lbl_coord, self.lbl_color, self.lbl_normal,
                    self.lbl_sem20, self.lbl_sem200, self.lbl_instance, self.lbl_superpoint,
                    self.lbl_seg):
            lbl.text_color = hud_color
            self.panel.add_child(lbl)

        # ---- top-left color-mode selector ----
        modes = ["rgb", "semantic", "instance"]
        if self.sp is not None:
            modes.append("superpoint")
        if self.data.get("pred_mask") is not None:
            modes.extend(["tpfpfn", "pred", "gt"])
        self.modes = modes
        self.combo = gui.Combobox()
        for m in modes:
            self.combo.add_item(m)
        self.combo.selected_index = modes.index(color_mode) if color_mode in modes else 0
        self.combo.set_on_selection_changed(self._on_color_mode)

        # ---- highlight toggle (below the color-mode selector) ----
        self.toggle = gui.Horiz(4, gui.Margins(0, 0, 0, 0))
        self.toggle_label = gui.Label("highlight superpoint")
        self.toggle_label.text_color = gui.Color(1.0, 1.0, 1.0, 1.0)
        self.toggle_check = gui.Checkbox("")
        self.toggle_check.checked = False
        self.toggle_check.set_on_checked(self._on_highlight_toggle)
        self.toggle.add_child(self.toggle_label)
        self.toggle.add_child(self.toggle_check)

        self.win.add_child(self.scene)
        self.win.add_child(self.panel)
        self.win.add_child(self.combo)
        self.win.add_child(self.toggle)
        self.win.set_on_layout(self._on_layout)

    def _on_layout(self, ctx):
        r = self.win.content_rect
        self.scene.frame = r
        pref = self.panel.calc_preferred_size(ctx, gui.Widget.Constraints())
        w = min(self.HUD_WIDTH, r.width // 2)
        h = min(int(pref.height), r.height)
        # anchor the HUD to the top-right corner with an 8 px margin
        self.panel.frame = gui.Rect(r.x + r.width - w - 8, r.y + 8, w, h)
        # anchor the color-mode combobox to the top-left corner
        cpref = self.combo.calc_preferred_size(ctx, gui.Widget.Constraints())
        self.combo.frame = gui.Rect(r.x + 8, r.y + 8,
                                    max(150, int(cpref.width)), int(cpref.height))
        # anchor the highlight toggle just below the combobox
        tpref = self.toggle.calc_preferred_size(ctx, gui.Widget.Constraints())
        ty = r.y + 8 + int(cpref.height) + 6
        self.toggle.frame = gui.Rect(r.x + 8, ty,
                                     max(150, int(tpref.width)), int(tpref.height))

    def _on_color_mode(self, text, idx):
        """Recompute cloud colors when the user picks a new color mode."""
        if text == self.color_mode or text not in self.modes:
            return
        self.color_mode = text
        colors = _make_colors(self.data, text)
        self.base_colors = colors
        if self.scene.scene.has_geometry("points"):
            self.scene.scene.remove_geometry("points")
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.scene.scene.add_geometry("points", self.pcd, self.mat)
        # clear the highlight overlay; it rebuilds on next hover with new colors
        self.highlighted_sp = None
        if self.scene.scene.has_geometry("highlight"):
            self.scene.scene.remove_geometry("highlight")
        self.win.post_redraw()

    def _on_highlight_toggle(self, is_checked):
        """Enable/disable the hover superpoint highlight overlay."""
        self.highlight_enabled = bool(is_checked)
        if not self.highlight_enabled:
            self.highlighted_sp = None
            if self.scene.scene.has_geometry("highlight"):
                self.scene.scene.remove_geometry("highlight")
        self.win.post_redraw()

    # ---- mouse -> point picking ----
    def _on_mouse(self, event):
        if event.type in (gui.MouseEvent.Type.MOVE,
                          gui.MouseEvent.Type.DRAG,
                          gui.MouseEvent.Type.BUTTON_DOWN):
            self._update_from_cursor(event.x, event.y)
        return gui.Widget.EventCallbackResult.IGNORED  # let orbit/pan still work

    def _reproject(self, view, proj, w, h):
        pts = self.points
        hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        cam = (view @ hom.T).T
        self._camz = cam[:, 2]
        clip = (proj @ cam.T).T
        cw = clip[:, 3]
        cw = np.where(np.abs(cw) < 1e-9, 1e-9, cw)
        ndc = clip[:, :3] / cw[:, None]
        self._sx = (ndc[:, 0] * 0.5 + 0.5) * w
        self._sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h

    def _update_from_cursor(self, mx, my):
        cam = self.scene.scene.camera
        view = np.asarray(cam.get_view_matrix(), dtype=np.float64)
        proj = np.asarray(cam.get_projection_matrix(), dtype=np.float64)
        w = max(1, self.scene.frame.width)
        h = max(1, self.scene.frame.height)

        key = (view.tobytes(), proj.tobytes())
        if key != self._cam_key or self._screen_wh != (w, h):
            self._reproject(view, proj, w, h)
            self._cam_key = key
            self._screen_wh = (w, h)

        d2 = (self._sx - mx) ** 2 + (self._sy - my) ** 2
        mask = d2 < self.PICK_RADIUS * self.PICK_RADIUS
        if mask.any():
            idxs = np.where(mask)[0]
            pick = int(idxs[int(np.argmax(self._camz[idxs]))])  # front-most (closest to camera)
            self._show_point(pick)
            if self.sp is not None and self.highlight_enabled:
                self._set_highlight(int(self.sp[pick]))
            else:
                self._set_highlight(None)
        else:
            self._clear_hud()
            self._set_highlight(None)
        self.win.post_redraw()

    def _set_highlight(self, sp_id):
        """Rebuild the highlighted-superpoint overlay, only when the SP id changes.

        Open3D's Open3DScene has no in-place color update, so we use a separate
        small overlay geometry (just the SP's points) that we remove+re-add.
        Gated on sp_id change so sweeping within one SP costs nothing.
        """
        if sp_id == self.highlighted_sp:
            return
        self.highlighted_sp = sp_id
        if self.scene.scene.has_geometry("highlight"):
            self.scene.scene.remove_geometry("highlight")
        if sp_id is None or self.sp is None:
            return
        m = self.sp == sp_id
        if not m.any():
            return
        ov = o3d.geometry.PointCloud()
        ov.points = o3d.utility.Vector3dVector(self.points[m])
        ov.colors = o3d.utility.Vector3dVector(self.base_colors[m])
        self.scene.scene.add_geometry("highlight", ov, self.mat_overlay)

    def _show_point(self, i):
        d = self.data
        c = d["coord"][i]
        col = d["color"][i]
        n = d["normal"][i]
        s20 = int(d["semantic_gt20"][i])
        s200 = int(d["semantic_gt200"][i])
        inst = int(d["instance_gt"][i])
        name = SCANNET20_NAMES[s20] if 0 <= s20 < len(SCANNET20_NAMES) else "unlabeled"
        self.hud_title.text = f"Point #{i}"
        self.lbl_coord.text = f"coord : ({c[0]:8.3f}, {c[1]:8.3f}, {c[2]:8.3f})"
        self.lbl_color.text = f"color : ({col[0]:7.1f}, {col[1]:7.1f}, {col[2]:7.1f})"
        self.lbl_normal.text = f"normal: ({n[0]:7.3f}, {n[1]:7.3f}, {n[2]:7.3f})"
        self.lbl_sem20.text = f"sem20 : {s20:>3} ({name})"
        self.lbl_sem200.text = f"sem200: {s200}"
        self.lbl_instance.text = f"inst  : {inst}"
        if self.sp is not None:
            self.lbl_superpoint.text = f"super : {int(self.sp[i])}"
        else:
            self.lbl_superpoint.text = ""
        if "pred_mask" in d and "gt_mask" in d:
            p = bool(d["pred_mask"][i])
            g = bool(d["gt_mask"][i])
            cat = "TP" if (p and g) else ("FP" if p else ("FN" if g else "TN"))
            self.lbl_seg.text = f"seg   : {cat}"
        else:
            self.lbl_seg.text = ""

    def _clear_hud(self):
        self.hud_title.text = "(no point under cursor)"
        for lbl in (self.lbl_coord, self.lbl_color, self.lbl_normal,
                    self.lbl_sem20, self.lbl_sem200, self.lbl_instance,
                    self.lbl_superpoint, self.lbl_seg):
            lbl.text = ""

    def run(self):
        self.app.run()


def run_viewer(pth_path, color_mode="rgb", point_size=4.0, center=True, split=None, mask_path=None):
    """Load a scene, build the colored cloud, and run the viewer (blocking)."""
    pth_path, split_used = _resolve_scene(pth_path, split)
    scene = load_scannet_scene(pth_path)

    # load superpoint ids from the sibling super_points/<scene>.bin
    scene_id = str(scene.get("scene_id", pth_path.stem))
    sp_path = pth_path.parent.parent / "super_points" / f"{scene_id}.bin"
    if sp_path.is_file():
        sp = np.fromfile(sp_path, dtype=np.int64)
        if len(sp) == len(scene["coord"]):
            scene["superpoint"] = sp
        else:
            print(f"WARNING: superpoint length {len(sp)} != points "
                  f"{len(scene['coord'])}; superpoint disabled")
    elif color_mode == "superpoint":
        raise FileNotFoundError(f"superpoint file not found: {sp_path}")
    else:
        print(f"note: no superpoint file at {sp_path} (hover-highlight disabled)")

    if mask_path is not None:
        md = np.load(mask_path)
        pm = np.asarray(md["pred_mask"]).astype(bool)
        gm = np.asarray(md["gt_mask"]).astype(bool)
        if len(pm) == len(scene["coord"]) and len(gm) == len(scene["coord"]):
            scene["pred_mask"] = pm
            scene["gt_mask"] = gm
            if color_mode == "rgb":  # default to the error-map when a mask is supplied
                color_mode = "tpfpfn"
            print(f"loaded mask {pathlib.Path(mask_path).name}: iou={float(md['iou']):.3f}")
        else:
            print(f"WARNING: mask length {len(pm)}/{len(gm)} != points "
                  f"{len(scene['coord'])}; seg overlay disabled")

    coord = scene["coord"]
    render_coord = coord - coord.mean(axis=0) if center else coord

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(render_coord)
    pcd.colors = o3d.utility.Vector3dVector(_make_colors(scene, color_mode))

    title = str(scene.get("scene_id", pth_path.stem))
    split_str = f" | split={split_used}" if split_used else ""
    print(f"{title}: {len(render_coord)} points | color_mode={color_mode}{split_str}")
    HoverPointCloudViewer(title, pcd, scene, point_size, color_mode=color_mode).run()


def main(argv):
    args = list(argv[1:])
    split = None
    mask = None
    for flag in ("--split", "--mask"):  # optional: --split train|val | --mask path.npz
        if flag in args:
            i = args.index(flag)
            args.pop(i)
            val = args.pop(i) if i < len(args) else None
            if flag == "--split":
                split = val
            else:
                mask = val
    if len(args) < 1:
        print(__doc__)
        sys.exit(1)
    scene = args[0]
    color_mode = args[1] if len(args) > 1 else "rgb"
    point_size = float(args[2]) if len(args) > 2 else 4.0
    center = (args[3].lower() not in ("0", "false", "no")) if len(args) > 3 else True
    run_viewer(scene, color_mode=color_mode, point_size=point_size, center=center,
               split=split, mask_path=mask)


if __name__ == "__main__":
    main(sys.argv)
