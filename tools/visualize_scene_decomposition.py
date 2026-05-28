"""
Generate a schematic two-panel figure showing scene decomposition:
  Left  panel: each dynamic object's Gaussians in its own LOCAL/canonical frame.
  Right panel: full scene in WORLD space (background Gaussians + objects placed
               via their per-frame SE(3) poses), with pose arrows from origin to
               each object center.

Usage:
  python tools/visualize_scene_decomposition.py \
      --checkpoint <path-to-checkpoint_30000.pth> \
      [--config <path-to-config.yaml>] \
      [--frame N] [--output decomposition.png] [--separate-panels] [--save-npz]
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.driving_dataset import DrivingDataset  # noqa: E402
from utils.misc import import_str  # noqa: E402
from pytorch3d.transforms import quaternion_to_matrix  # noqa: E402


# ---------- I/O helpers ----------

def load_trainer(checkpoint_path: str, config_path: str = None, device: str = "cuda",
                 strict: bool = False):
    log_dir = os.path.dirname(checkpoint_path)
    if config_path is None:
        config_path = os.path.join(log_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config at {config_path}")

    cfg = OmegaConf.load(config_path)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    dataset = DrivingDataset(data_cfg=cfg.data)

    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device_t,
    )

    # Load with strict=False so checkpoints that pre-date newer parameter
    # registrations (e.g. PartRigidNodes.seg_*_residuals) still work for
    # visualization-only use.
    print(f"      loading state_dict (strict={strict})...")
    state_dict = torch.load(checkpoint_path, map_location=device_t)
    trainer.load_state_dict(state_dict, load_only_model=True, strict=strict)
    trainer.set_eval()
    return trainer, dataset, cfg


# ---------- Data extraction ----------

def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _subsample(xyz: np.ndarray, rgb: np.ndarray, n: int, seed: int = 0):
    if xyz.shape[0] <= n:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=n, replace=False)
    return xyz[idx], rgb[idx]


def extract_background(trainer, alpha_thresh: float, max_points: int):
    if "Background" not in trainer.models:
        return None
    bg = trainer.models["Background"]
    with torch.no_grad():
        out = bg.export_gaussians_to_ply(alpha_thresh=alpha_thresh)
    xyz = _to_np(out["positions"])
    rgb = np.clip(_to_np(out["colors"]), 0.0, 1.0)
    return _subsample(xyz, rgb, max_points, seed=0)


def get_camera_position(dataset, frame: int, cam_id: int = 0) -> np.ndarray:
    """Return the world-space (x,y,z) of camera `cam_id` at `frame`."""
    cam = dataset.pixel_source.camera_data[cam_id]
    c2w = cam.cam_to_worlds[frame]
    if hasattr(c2w, "detach"):
        c2w = c2w.detach().cpu().numpy()
    return np.asarray(c2w[:3, 3], dtype=np.float64)


def extract_objects(trainer, frame: int, alpha_thresh: float, max_points: int):
    """
    For each instance in each dynamic node class, return
       local_xyz, local_rgb, world_xyz, R(3x3), t(3,), class_name, instance_id
    Local means are read directly from `_means` (canonical frame). World means
    are obtained by R @ local + t with per-frame quaternion/translation.
    """
    results: List[Dict] = []

    dynamic_classes = [c for c in ["RigidNodes", "SMPLNodes", "DeformableNodes", "PartRigidNodes"]
                       if c in trainer.models]

    for cname in dynamic_classes:
        node = trainer.models[cname]
        if not hasattr(node, "point_ids") or node.point_ids is None:
            continue

        point_ids = _to_np(node.point_ids[..., 0])
        unique_ids = np.unique(point_ids).tolist()

        # opacity mask (per gaussian)
        with torch.no_grad():
            op = node.get_opacity.squeeze().detach().cpu().numpy()
        means_all = _to_np(node._means)
        colors_all = np.clip(_to_np(node.colors), 0.0, 1.0)

        # instances_quats: rigid is (F, I, 4); smpl is (F, I, 1, 4); handle both
        with torch.no_grad():
            q = node.instances_quats
            if q.dim() == 4:
                q = q[:, :, 0, :]
            t_all = node.instances_trans  # (F, I, 3)

        # check frame validity
        num_frames = q.shape[0]
        use_frame = int(np.clip(frame, 0, num_frames - 1))

        # instances_fv: (F, I) bool  -> if False find nearest valid
        fv = getattr(node, "instances_fv", None)
        if fv is not None:
            fv_np = _to_np(fv).astype(bool)
        else:
            fv_np = np.ones((num_frames, q.shape[1]), dtype=bool)

        for ins_id in unique_ids:
            ins_id = int(ins_id)
            mask = (point_ids == ins_id) & (op > alpha_thresh)
            if mask.sum() == 0:
                continue
            local_xyz = means_all[mask]
            local_rgb = colors_all[mask]
            local_xyz, local_rgb = _subsample(local_xyz, local_rgb, max_points, seed=ins_id + 1)

            # pick a valid frame for this instance, fall back to nearest
            f = use_frame
            if not fv_np[f, ins_id]:
                # nearest valid
                offsets = np.argsort(np.abs(np.arange(num_frames) - use_frame))
                for cand in offsets:
                    if fv_np[cand, ins_id]:
                        f = int(cand)
                        break

            qf = q[f, ins_id].detach()
            tf = t_all[f, ins_id].detach()
            R = _to_np(quaternion_to_matrix(qf))  # (3,3)
            t = _to_np(tf)                         # (3,)
            world_xyz = (R @ local_xyz.T).T + t[None, :]

            results.append({
                "class": cname,
                "instance_id": ins_id,
                "local_xyz": local_xyz,
                "local_rgb": local_rgb,
                "world_xyz": world_xyz,
                "R": R,
                "t": t,
                "frame": f,
            })
    return results


# ---------- Plotting ----------

PALETTE = plt.get_cmap("tab10").colors  # 10 distinct colors


def _palette_color(i: int):
    return np.asarray(PALETTE[i % len(PALETTE)], dtype=np.float32)


def _blended_colors(rgb: np.ndarray, palette_color: np.ndarray, alpha_palette: float = 0.2):
    """Per-Gaussian color = alpha * palette + (1-alpha) * learned RGB.
    Default leans heavily on the learned RGB so objects keep their real appearance,
    with only a subtle palette tint to keep instances distinguishable."""
    pc = palette_color[None, :]
    blended = alpha_palette * pc + (1.0 - alpha_palette) * rgb
    return np.clip(blended, 0.0, 1.0)


def _set_3d_axes(ax, title: str = "", lims=None):
    """Strip all axis chrome: no ticks, no panes, no box outline, no grid.
    Keeps equal-aspect framing based on the data bounds in `lims`."""
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    if lims is not None:
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        # equal aspect from real data extents
        spans = [abs(lims[i][1] - lims[i][0]) for i in range(3)]
        try:
            ax.set_box_aspect(spans)
        except Exception:
            pass


def _compute_lims(points: np.ndarray, pad_frac: float = 0.03):
    """Tight bounds around the data so nothing floats in dead space."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    pad = pad_frac * (maxs - mins + 1e-6)
    return [(mins[i] - pad[i], maxs[i] + pad[i]) for i in range(3)]


def plot_local_panel(fig, gs_left, objects, elev, azim):
    n = len(objects)
    if n == 0:
        ax = fig.add_subplot(gs_left)
        ax.text(0.5, 0.5, "No dynamic objects", ha="center", va="center")
        ax.set_axis_off()
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    inner = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs_left, hspace=0.3, wspace=0.05)
    for i, obj in enumerate(objects):
        ax = fig.add_subplot(inner[i // cols, i % cols], projection="3d")
        xyz = obj["local_xyz"]
        rgb = obj["local_rgb"]
        col = _blended_colors(rgb, _palette_color(i))
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1.0, c=col,
                   depthshade=False, linewidths=0)
        lims = _compute_lims(xyz, pad_frac=0.08)
        _set_3d_axes(ax, title=f"{obj['class']} #{obj['instance_id']}", lims=lims)
        ax.view_init(elev=elev, azim=azim)


def plot_world_panel(ax, bg, objects, elev, azim,
                     draw_arrows=True, draw_bg=True, draw_objs=True,
                     arrow_origin="centroid", show_labels=True,
                     roi_center=None, roi_radius=None, roi_z_scale=0.3):
    """If roi_center+roi_radius are given, frame the panel on that ROI rather
    than fitting all data extents (useful when the scene trajectory is huge
    but only a small region is interesting)."""
    all_pts = []
    if draw_bg and bg is not None:
        all_pts.append(bg[0])
    if draw_objs:
        for o in objects:
            all_pts.append(o["world_xyz"])

    if len(all_pts) == 0:
        ax.text2D(0.5, 0.5, "Empty scene", transform=ax.transAxes, ha="center")
        return

    cat = np.concatenate(all_pts, axis=0)
    if roi_center is not None and roi_radius is not None:
        c = np.asarray(roi_center, dtype=np.float64)
        r = float(roi_radius)
        rz = float(roi_radius) * float(roi_z_scale)
        lims = [(c[0] - r, c[0] + r),
                (c[1] - r, c[1] + r),
                (c[2] - rz, c[2] + rz)]
        scene_centroid = c
    else:
        lims = _compute_lims(cat, pad_frac=0.03)
        scene_centroid = cat.mean(axis=0)

    if draw_bg and bg is not None:
        bg_xyz, bg_rgb = bg
        ax.scatter(bg_xyz[:, 0], bg_xyz[:, 1], bg_xyz[:, 2],
                   s=0.4, c=np.clip(bg_rgb, 0, 1), alpha=0.35,
                   depthshade=False, linewidths=0)

    if draw_objs:
        for i, obj in enumerate(objects):
            xyz = obj["world_xyz"]
            col = _blended_colors(obj["local_rgb"], _palette_color(i))
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1.5, c=col,
                       depthshade=False, linewidths=0)

        if draw_arrows:
            # arrow origin: either the world origin (can be far away) or the
            # scene centroid (visually cleaner for thesis figures).
            if arrow_origin == "origin":
                origin = np.zeros(3)
            else:
                origin = scene_centroid
            # Stagger label z-offset slightly so they don't all collide.
            z_span = lims[2][1] - lims[2][0]
            for i, obj in enumerate(objects):
                center = obj["world_xyz"].mean(axis=0)
                pc = _palette_color(i)
                ax.plot([origin[0], center[0]],
                        [origin[1], center[1]],
                        [origin[2], center[2]],
                        color=pc, linewidth=1.0, alpha=0.7)
                ax.scatter([center[0]], [center[1]], [center[2]],
                           s=18, c=[pc], edgecolors="black", linewidths=0.4)
                if show_labels:
                    z_off = 0.02 * z_span * (1 + (i % 3))
                    ax.text(center[0], center[1], center[2] + z_off,
                            f"#{obj['instance_id']}", fontsize=7, color="black")

    _set_3d_axes(ax, title="World composition", lims=lims)
    ax.view_init(elev=elev, azim=azim)


def make_combined_figure(bg, objects, out_path: str, elev: float, azim: float, dpi: int,
                         roi_center=None, roi_radius=None, roi_z_scale=0.3,
                         show_labels=True, draw_arrows=True):
    """Two-panel figure: object-local frames on the left, world composition on the right."""
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.2], wspace=0.05)
    plot_local_panel(fig, gs[0, 0], objects, elev=elev, azim=azim)
    ax_world = fig.add_subplot(gs[0, 1], projection="3d")
    plot_world_panel(ax_world, bg, objects, elev=elev, azim=azim,
                     roi_center=roi_center, roi_radius=roi_radius, roi_z_scale=roi_z_scale,
                     show_labels=show_labels, draw_arrows=draw_arrows)
    fig.suptitle("Scene decomposition: object-local frames (left) → world space (right)", fontsize=12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_world_only_figure(bg, objects, out_path: str, elev: float, azim: float, dpi: int,
                           roi_center=None, roi_radius=None, roi_z_scale=0.3,
                           show_labels=True, draw_arrows=True, transparent=False):
    """Single-panel: full composed world (background + objects + pose arrows)."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_world_panel(ax, bg, objects, elev=elev, azim=azim,
                     roi_center=roi_center, roi_radius=roi_radius, roi_z_scale=roi_z_scale,
                     show_labels=show_labels, draw_arrows=draw_arrows)
    ax.set_title("")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def make_separate_panels(bg, objects, out_dir: str, elev: float, azim: float, dpi: int,
                         roi_center=None, roi_radius=None, roi_z_scale=0.3,
                         show_labels=True, draw_arrows=True):
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_world_panel(ax, bg, objects, elev=elev, azim=azim,
                     draw_objs=False, draw_arrows=False,
                     roi_center=roi_center, roi_radius=roi_radius, roi_z_scale=roi_z_scale)
    fig.savefig(os.path.join(out_dir, "background_only.png"), dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_world_panel(ax, bg, objects, elev=elev, azim=azim, draw_bg=False,
                     roi_center=roi_center, roi_radius=roi_radius, roi_z_scale=roi_z_scale,
                     show_labels=show_labels, draw_arrows=draw_arrows)
    fig.savefig(os.path.join(out_dir, "objects_world.png"), dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 1)
    plot_local_panel(fig, gs[0, 0], objects, elev=elev, azim=azim)
    fig.savefig(os.path.join(out_dir, "objects_local.png"), dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close(fig)


def save_npz(bg, objects, out_path: str):
    data = {}
    if bg is not None:
        data["bg_xyz"] = bg[0]
        data["bg_rgb"] = bg[1]
    for o in objects:
        key = f"{o['class']}_{o['instance_id']}"
        data[f"{key}_local_xyz"] = o["local_xyz"]
        data[f"{key}_local_rgb"] = o["local_rgb"]
        data[f"{key}_world_xyz"] = o["world_xyz"]
        data[f"{key}_R"] = o["R"]
        data[f"{key}_t"] = o["t"]
        data[f"{key}_frame"] = np.array(o["frame"])
    np.savez_compressed(out_path, **data)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--frame", default=None, type=int,
                        help="Timestamp index to visualize (default: middle frame)")
    parser.add_argument("--output", default="decomposition.png", type=str)
    parser.add_argument("--max-points-bg", default=50000, type=int)
    parser.add_argument("--max-points-obj", default=5000, type=int)
    parser.add_argument("--alpha-thresh", default=0.1, type=float)
    parser.add_argument("--elev", default=25.0, type=float)
    parser.add_argument("--azim", default=-60.0, type=float)
    parser.add_argument("--separate-panels", action="store_true")
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--strict-load", action="store_true",
                        help="Use strict=True when loading the checkpoint (default: non-strict)")
    # Region-of-interest cropping
    parser.add_argument("--roi-radius", type=float, default=15.0,
                        help="Half-extent in meters around the ROI center used to crop the "
                             "world panel and pick which objects to show. Set <=0 to disable.")
    parser.add_argument("--roi-z-scale", type=float, default=0.3,
                        help="Vertical extent multiplier relative to roi-radius (smaller = less empty sky).")
    parser.add_argument("--roi-center", type=float, nargs=3, default=None,
                        help="World-space (x y z) of ROI center. Defaults to the camera "
                             "position at the chosen frame.")
    parser.add_argument("--roi-cam-id", type=int, default=0,
                        help="Which camera's position to use as ROI center when --roi-center is not given.")
    parser.add_argument("--max-objects", type=int, default=8,
                        help="Cap on number of objects to draw in the world panel (closest to ROI center first).")
    parser.add_argument("--no-labels", action="store_true", help="Hide #id labels on objects.")
    parser.add_argument("--no-arrows", action="store_true", help="Hide pose arrows in the world panel.")
    args = parser.parse_args()

    print(f"[1/4] Loading checkpoint: {args.checkpoint}")
    trainer, dataset, cfg = load_trainer(args.checkpoint, args.config,
                                         device=args.device, strict=args.strict_load)

    num_frames = dataset.num_img_timesteps
    frame = args.frame if args.frame is not None else num_frames // 2
    print(f"      num_frames={num_frames}, visualizing frame={frame}")

    print("[2/4] Extracting background Gaussians...")
    bg = extract_background(trainer, alpha_thresh=args.alpha_thresh, max_points=args.max_points_bg)
    if bg is not None:
        print(f"      background points: {bg[0].shape[0]:,}")

    print("[3/4] Extracting object Gaussians + poses...")
    objects = extract_objects(trainer, frame=frame, alpha_thresh=args.alpha_thresh,
                              max_points=args.max_points_obj)
    print(f"      objects (raw): {len(objects)}")

    # Determine ROI center
    roi_center = None
    roi_radius = args.roi_radius if args.roi_radius > 0 else None
    if roi_radius is not None:
        if args.roi_center is not None:
            roi_center = np.asarray(args.roi_center, dtype=np.float64)
        else:
            try:
                roi_center = get_camera_position(dataset, frame, args.roi_cam_id)
            except Exception as e:
                print(f"      WARN: could not get camera pose ({e}); ROI disabled")
                roi_radius = None
    if roi_center is not None:
        print(f"      ROI center = {roi_center}, radius = {roi_radius}")

    # Filter background and objects to ROI, cap object count by proximity
    if roi_center is not None and roi_radius is not None:
        if bg is not None:
            bg_xyz, bg_rgb = bg
            d = np.linalg.norm(bg_xyz - roi_center[None, :], axis=1)
            keep = d < roi_radius
            bg = (bg_xyz[keep], bg_rgb[keep])
            print(f"      background after ROI crop: {bg[0].shape[0]:,} pts")
        # rank objects by distance to ROI center, keep nearest --max-objects
        objects.sort(key=lambda o: np.linalg.norm(o["world_xyz"].mean(axis=0) - roi_center))
        objects = [o for o in objects
                   if np.linalg.norm(o["world_xyz"].mean(axis=0) - roi_center) < roi_radius]
        if args.max_objects > 0:
            objects = objects[: args.max_objects]
    elif args.max_objects > 0:
        objects = objects[: args.max_objects]

    print(f"      objects (drawn): {len(objects)}")
    for o in objects:
        print(f"        - {o['class']} #{o['instance_id']}: "
              f"{o['local_xyz'].shape[0]:,} pts, frame={o['frame']}")

    print("[4/4] Rendering figure...")
    make_combined_figure(bg, objects, args.output,
                         elev=args.elev, azim=args.azim, dpi=args.dpi,
                         roi_center=roi_center, roi_radius=roi_radius,
                         roi_z_scale=args.roi_z_scale,
                         show_labels=not args.no_labels, draw_arrows=not args.no_arrows)
    print(f"      wrote {args.output}")

    # Single combined world-only render (background + objects + arrows in one panel).
    world_path = os.path.splitext(args.output)[0] + "_world.png"
    make_world_only_figure(bg, objects, world_path,
                           elev=args.elev, azim=args.azim, dpi=args.dpi,
                           roi_center=roi_center, roi_radius=roi_radius,
                           roi_z_scale=args.roi_z_scale,
                           show_labels=not args.no_labels, draw_arrows=not args.no_arrows)
    print(f"      wrote {world_path}")

    if args.separate_panels:
        sep_dir = os.path.splitext(args.output)[0] + "_panels"
        make_separate_panels(bg, objects, sep_dir,
                             elev=args.elev, azim=args.azim, dpi=args.dpi,
                             roi_center=roi_center, roi_radius=roi_radius,
                             roi_z_scale=args.roi_z_scale,
                             show_labels=not args.no_labels, draw_arrows=not args.no_arrows)
        print(f"      wrote separate panels to {sep_dir}/")

    if args.save_npz:
        npz_path = os.path.splitext(args.output)[0] + "_data.npz"
        save_npz(bg, objects, npz_path)
        print(f"      wrote {npz_path}")

    print("Done.")


if __name__ == "__main__":
    main()
