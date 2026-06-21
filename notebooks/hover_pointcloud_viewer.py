"""Standalone hover-to-inspect point-cloud viewer for preprocessed ScanNet.

This is meant to be launched as its own process (see launch() in the
VisualizePointcloud notebook). Running it as a plain `python` process — not
inside an IPython/Jupyter kernel — avoids Open3D's automatic "WebRTC Jupyter
handshake mode", which otherwise blocks forever waiting for a browser client
that never connects. Here Open3D opens a normal native window instead.

Usage:
    python hover_pointcloud_viewer.py <scene_path_or_id> [color_mode] [point_size] [--split train|val]

    color_mode: rgb | semantic | instance   (default: rgb)
    point_size: render point size in px       (default: 4.0)
    --split  : restrict a scene_id lookup to 'train' or 'val' (default: search both)

Move the mouse over the cloud: the front-most point under the cursor has its
coord / color / normal / semantic_gt20 / semantic_gt200 / instance_gt shown
in the top-right corner of the window.
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
    raise ValueError(f"unknown color_mode: {color_mode!r}")


class HoverPointCloudViewer:
    """Open3D GUI viewer that shows the hovered point's attributes top-right.

    Picking projects all points to the screen with the current camera matrices
    (cached per camera pose so it stays cheap), then takes the front-most point
    within a few pixels of the cursor.
    """

    HUD_WIDTH = 330
    PICK_RADIUS = 12.0  # screen pixels

    def __init__(self, title, pcd, scene_data, point_size):
        self.points = np.asarray(pcd.points, dtype=np.float64)  # as rendered (centered)
        self.data = scene_data                                  # original arrays, index-aligned
        self._cam_key = None
        self._screen_wh = None
        self._sx = self._sy = self._camz = None

        self.app = gui.Application.instance
        self.app.initialize()
        self.win = self.app.create_window(title, 1600, 900)
        self._build(pcd, point_size)

    def _build(self, pcd, point_size):
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.win.renderer)
        self.scene.scene.set_background(  # black background
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"  # use per-point vertex colors
        mat.point_size = point_size
        self.scene.scene.add_geometry("points", pcd, mat)
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
        for lbl in (self.hud_title, self.lbl_coord, self.lbl_color, self.lbl_normal,
                    self.lbl_sem20, self.lbl_sem200, self.lbl_instance):
            lbl.text_color = hud_color
            self.panel.add_child(lbl)

        self.win.add_child(self.scene)
        self.win.add_child(self.panel)
        self.win.set_on_layout(self._on_layout)

    def _on_layout(self, ctx):
        r = self.win.content_rect
        self.scene.frame = r
        pref = self.panel.calc_preferred_size(ctx, gui.Widget.Constraints())
        w = min(self.HUD_WIDTH, r.width // 2)
        h = min(int(pref.height), r.height)
        # anchor to the top-right corner with an 8 px margin
        self.panel.frame = gui.Rect(r.x + r.width - w - 8, r.y + 8, w, h)

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
            pick = idxs[int(np.argmax(self._camz[idxs]))]  # front-most (closest to camera)
            self._show_point(int(pick))
        else:
            self._clear_hud()
        self.win.post_redraw()

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

    def _clear_hud(self):
        self.hud_title.text = "(no point under cursor)"
        for lbl in (self.lbl_coord, self.lbl_color, self.lbl_normal,
                    self.lbl_sem20, self.lbl_sem200, self.lbl_instance):
            lbl.text = ""

    def run(self):
        self.app.run()


def run_viewer(pth_path, color_mode="rgb", point_size=4.0, center=True, split=None):
    """Load a scene, build the colored cloud, and run the viewer (blocking)."""
    pth_path, split_used = _resolve_scene(pth_path, split)
    scene = load_scannet_scene(pth_path)
    coord = scene["coord"]
    render_coord = coord - coord.mean(axis=0) if center else coord

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(render_coord)
    pcd.colors = o3d.utility.Vector3dVector(_make_colors(scene, color_mode))

    title = str(scene.get("scene_id", pth_path.stem))
    split_str = f" | split={split_used}" if split_used else ""
    print(f"{title}: {len(render_coord)} points | color_mode={color_mode}{split_str}")
    HoverPointCloudViewer(title, pcd, scene, point_size).run()


def main(argv):
    args = list(argv[1:])
    split = None
    if "--split" in args:  # optional: --split train|val
        i = args.index("--split")
        args.pop(i)
        split = args.pop(i) if i < len(args) else None
    if len(args) < 1:
        print(__doc__)
        sys.exit(1)
    scene = args[0]
    color_mode = args[1] if len(args) > 1 else "rgb"
    point_size = float(args[2]) if len(args) > 2 else 4.0
    center = (args[3].lower() not in ("0", "false", "no")) if len(args) > 3 else True
    run_viewer(scene, color_mode=color_mode, point_size=point_size, center=center, split=split)


if __name__ == "__main__":
    main(sys.argv)
