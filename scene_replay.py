#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from utils.params import ISAAC_TO_MUJOCO
from utils.urdf_to_mujoco import (
    merge_free_box_into_scene_xml,
    merge_object_into_scene,
    merge_terrain_box_into_scene_xml,
    merge_terrain_boxes_into_scene_xml,
    merge_terrain_into_scene_from_string,
    merge_terrain_wedges_into_scene_xml,
)


def _resolve_path(path_like: str, config_dir: Path, workspace_root: Path) -> str:
    p = Path(path_like)
    if p.is_absolute():
        resolved = p.resolve()
        if resolved.exists():
            return str(resolved)
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    candidates = [
        (config_dir / p).resolve(),
        (workspace_root / p).resolve(),
        (Path.cwd() / p).resolve(),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)

    raise FileNotFoundError(
        "Path does not exist. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + f"\nOriginal value: {path_like}"
    )


def _clamp_start_index(start: int, motion_length: int) -> int:
    if motion_length <= 0:
        return 0
    return max(0, min(int(start), motion_length - 1))


def _load_config(config_path: str) -> Tuple[Dict[str, Any], Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "rl_policy" not in raw:
        raise ValueError("Config file must contain top-level key `rl_policy`.")
    cfg = raw["rl_policy"]
    if not isinstance(cfg, dict):
        raise ValueError("`rl_policy` must be a mapping.")
    return cfg, path.parent


def _terrain_friction_from_value(value: Any, ctx: str) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        friction_t = tuple(float(x) for x in value)
        if len(friction_t) != 3:
            raise ValueError(f"{ctx}: friction must be scalar or length-3 [slide, torsion, roll]")
    else:
        f = float(value)
        friction_t = (f, f, f)
    if min(friction_t) < 0.0:
        raise ValueError(f"{ctx}: friction values must be >= 0")
    return friction_t


def _yaw_from_cfg(spec: Dict[str, Any], ctx: str) -> float:
    has_yaw = "yaw" in spec
    has_yaw_deg = "yaw_deg" in spec
    if has_yaw and has_yaw_deg:
        raise ValueError(f"{ctx}: use only one of yaw (radians) or yaw_deg")
    if has_yaw_deg:
        return math.radians(float(spec["yaw_deg"]))
    if has_yaw:
        return float(spec["yaw"])
    return 0.0


def _wedge_size_from_cfg(spec: Dict[str, Any], ctx: str) -> Tuple[float, float, float]:
    size_ = spec.get("size")
    if size_ is None:
        raise ValueError(f"{ctx} must include size [run, width, rise] (or [run, width] with angle/angle_deg)")
    size_t = tuple(float(x) for x in size_)
    has_angle = "angle" in spec or "angle_deg" in spec
    if "angle" in spec and "angle_deg" in spec:
        raise ValueError(f"{ctx}: use only one of angle (radians) or angle_deg")
    if len(size_t) == 2:
        if not has_angle:
            raise ValueError(f"{ctx}: size length-2 [run, width] requires angle or angle_deg")
        run, width = size_t
        angle = math.radians(float(spec["angle_deg"])) if "angle_deg" in spec else float(spec["angle"])
        size_t = (run, width, run * math.tan(angle))
    elif len(size_t) == 3:
        if has_angle:
            raise ValueError(f"{ctx}: provide rise via size[2] OR angle/angle_deg, not both")
    else:
        raise ValueError(f"{ctx}: size must be length-3 [run, width, rise] or length-2 [run, width]")
    if min(size_t) <= 0:
        raise ValueError(f"{ctx}: run, width, and rise must be positive")
    return size_t


def _merge_scene_xml(
    base_xml: str,
    cfg: Dict[str, Any],
    config_dir: Path,
    workspace_root: Path,
) -> str:
    scene_xml = base_xml

    # terrain_boxes (new style) or terrain_box_pos/size (legacy)
    terrain_boxes_cfg = cfg.get("terrain_boxes")
    tb_pos = cfg.get("terrain_box_pos")
    tb_size = cfg.get("terrain_box_size")
    has_terrain_box_legacy = tb_pos is not None and tb_size is not None
    has_terrain_box_multi = isinstance(terrain_boxes_cfg, (dict, list)) and len(terrain_boxes_cfg) > 0
    if has_terrain_box_legacy and has_terrain_box_multi:
        raise ValueError("Use either terrain_box_pos/terrain_box_size or terrain_boxes, not both")

    if has_terrain_box_multi:
        default_rgba = (0.55, 0.52, 0.48, 1.0)
        default_friction = _terrain_friction_from_value(
            cfg.get("terrain_boxes_friction", cfg.get("terrain_box_friction")),
            "terrain_boxes_friction",
        )
        boxes = []
        if isinstance(terrain_boxes_cfg, dict):
            iterator = terrain_boxes_cfg.items()
        else:
            iterator = [(str(i), v) for i, v in enumerate(terrain_boxes_cfg)]
        for name, spec in iterator:
            if not isinstance(spec, dict):
                raise TypeError(f"terrain_boxes[{name!r}] must be a dict")
            if "pos" not in spec or "size" not in spec:
                raise ValueError(f"terrain_boxes[{name!r}] must include pos and size")
            pos_t = tuple(float(x) for x in spec["pos"])
            size_t = tuple(float(x) for x in spec["size"])
            if len(pos_t) != 3 or len(size_t) != 3:
                raise ValueError(f"terrain_boxes[{name!r}]: pos/size must be length-3")
            if min(size_t) <= 0:
                raise ValueError(f"terrain_boxes[{name!r}]: size must be positive")
            rgba_t = None
            if "rgba" in spec and spec["rgba"] is not None:
                rgba_t = tuple(float(x) for x in spec["rgba"])
                if len(rgba_t) != 4:
                    raise ValueError(f"terrain_boxes[{name!r}]: rgba must be length-4")
            yaw_t = _yaw_from_cfg(spec, f"terrain_boxes[{name!r}]")
            friction_t = _terrain_friction_from_value(spec.get("friction"), f"terrain_boxes[{name!r}]")
            boxes.append((str(spec.get("name", name)), pos_t, size_t, rgba_t, friction_t, yaw_t))
        scene_xml = merge_terrain_boxes_into_scene_xml(
            scene_xml,
            boxes,
            default_rgba=default_rgba,
            default_friction=default_friction,
        )
    elif has_terrain_box_legacy:
        pos_t = tuple(float(x) for x in tb_pos)
        size_t = tuple(float(x) for x in tb_size)
        if len(pos_t) != 3 or len(size_t) != 3:
            raise ValueError("terrain_box_pos and terrain_box_size must each be length-3")
        if min(size_t) <= 0:
            raise ValueError("terrain_box_size entries must be positive")
        if "terrain_box_yaw" in cfg and "terrain_box_yaw_deg" in cfg:
            raise ValueError("Use only one of terrain_box_yaw (rad) or terrain_box_yaw_deg")
        if "terrain_box_yaw_deg" in cfg:
            yaw_t = math.radians(float(cfg["terrain_box_yaw_deg"]))
        else:
            yaw_t = float(cfg.get("terrain_box_yaw", 0.0))

        rgba_t = tuple(float(x) for x in cfg.get("terrain_box_rgba", (0.55, 0.52, 0.48, 1.0)))
        if len(rgba_t) != 4:
            raise ValueError("terrain_box_rgba must be length-4")
        friction_t = _terrain_friction_from_value(cfg.get("terrain_box_friction"), "terrain_box_friction")
        scene_xml = merge_terrain_box_into_scene_xml(
            scene_xml,
            pos_t,
            size_t,
            rgba=rgba_t,
            friction=friction_t,
            yaw_rad=yaw_t,
        )

    # terrain_wedges (procedural ramps/slopes)
    terrain_wedges_cfg = cfg.get("terrain_wedges")
    has_terrain_wedge = isinstance(terrain_wedges_cfg, (dict, list)) and len(terrain_wedges_cfg) > 0
    if has_terrain_wedge:
        default_rgba = (0.55, 0.52, 0.48, 1.0)
        default_friction = _terrain_friction_from_value(
            cfg.get("terrain_wedges_friction", cfg.get("terrain_wedge_friction")),
            "terrain_wedges_friction",
        )
        if isinstance(terrain_wedges_cfg, dict):
            wedge_iter = terrain_wedges_cfg.items()
        else:
            wedge_iter = [(f"wedge_{i}", v) for i, v in enumerate(terrain_wedges_cfg)]
        wedges = []
        for name, spec in wedge_iter:
            ctx = f"terrain_wedges[{name!r}]"
            if not isinstance(spec, dict):
                raise TypeError(f"{ctx} must be a dict")
            if "pos" not in spec:
                raise ValueError(f"{ctx} must include pos [x, y, z]")
            pos_t = tuple(float(x) for x in spec["pos"])
            if len(pos_t) != 3:
                raise ValueError(f"{ctx}: pos must be length-3")
            size_t = _wedge_size_from_cfg(spec, ctx)
            rgba_t = None
            if "rgba" in spec and spec["rgba"] is not None:
                rgba_t = tuple(float(x) for x in spec["rgba"])
                if len(rgba_t) != 4:
                    raise ValueError(f"{ctx}: rgba must be length-4")
            yaw_t = _yaw_from_cfg(spec, ctx)
            friction_t = _terrain_friction_from_value(spec.get("friction"), ctx)
            wedges.append((str(spec.get("name", name)), pos_t, size_t, rgba_t, friction_t, yaw_t))
        scene_xml = merge_terrain_wedges_into_scene_xml(
            scene_xml,
            wedges,
            default_rgba=default_rgba,
            default_friction=default_friction,
        )

    # free dynamic box
    fb_pos = cfg.get("free_box_pos")
    fb_size = cfg.get("free_box_size")
    if fb_pos is not None and fb_size is not None:
        pos_t = tuple(float(x) for x in fb_pos)
        size_t = tuple(float(x) for x in fb_size)
        if len(pos_t) != 3 or len(size_t) != 3:
            raise ValueError("free_box_pos and free_box_size must each be length-3")
        if min(size_t) <= 0:
            raise ValueError("free_box_size entries must be positive")
        mass = float(cfg.get("free_box_mass", 1.0))
        if mass <= 0:
            raise ValueError("free_box_mass must be > 0")
        rgba_t = tuple(float(x) for x in cfg.get("free_box_rgba", (0.65, 0.45, 0.35, 1.0)))
        if len(rgba_t) != 4:
            raise ValueError("free_box_rgba must be length-4")
        scene_xml = merge_free_box_into_scene_xml(scene_xml, pos_t, size_t, mass, rgba_t)

    # terrain URDF
    terrain_urdf = str(cfg.get("terrain_urdf", "") or "").strip()
    if terrain_urdf:
        terrain_path = _resolve_path(terrain_urdf, config_dir, workspace_root)
        use_columns_for_collision = not bool(cfg.get("terrain_mesh_collision", False))
        terrain_urdf_offset = tuple(float(x) for x in cfg.get("terrain_urdf_offset", (0.0, 0.0, 0.0)))
        if len(terrain_urdf_offset) != 3:
            raise ValueError("terrain_urdf_offset must be length-3 [x, y, z]")
        scene_xml = merge_terrain_into_scene_from_string(
            scene_xml,
            terrain_path,
            use_columns_for_collision=use_columns_for_collision,
            terrain_column_res=float(cfg.get("terrain_column_res", 0.2)),
            terrain_floor_threshold=float(cfg.get("terrain_floor_threshold", 0.02)),
            terrain_urdf_offset=terrain_urdf_offset,
        )

    # object URDF (must be merged last to preserve robot qpos indexing)
    object_urdf = str(cfg.get("object_urdf", "") or "").strip()
    if object_urdf:
        object_path = _resolve_path(object_urdf, config_dir, workspace_root)
        # Replay uses visual-only object geometry (no collision geoms from URDF).
        scene_xml = merge_object_into_scene(
            scene_xml,
            object_path,
            use_collision_geometry=False,
            enable_collision=False,
        )

    return scene_xml


def _parse_slow_motion_params(
    slow_motion_end_frame: Any,
    slow_down_times: Any,
    motion_length: int,
) -> Tuple[Optional[int], Optional[int]]:
    """Match rl_policy._parse_slow_motion_params: (end_frame, repeat_times) or (None, None)."""
    if motion_length <= 0:
        return None, None
    if slow_motion_end_frame is None or slow_down_times is None:
        return None, None
    try:
        end_frame = int(slow_motion_end_frame)
        repeat_times = int(slow_down_times)
    except (TypeError, ValueError):
        return None, None
    if end_frame <= 0 or repeat_times <= 1:
        return None, None
    end_frame = min(end_frame, motion_length)
    if end_frame <= 0:
        return None, None
    return end_frame, repeat_times


def _stretch_prefix_along_time(arr: np.ndarray, end_frame: int, repeat_times: int) -> np.ndarray:
    """Repeat frames [0:end_frame) along axis-0 by repeat_times (same as rl_policy)."""
    prefix = np.repeat(arr[:end_frame], repeat_times, axis=0)
    return np.concatenate((prefix, arr[end_frame:]), axis=0)


def _stretch_motion_dict_prefix_if_enabled(
    motion: Dict[str, np.ndarray],
    motion_length: int,
    slow_motion_end_frame: Any,
    slow_down_times: Any,
    label: str,
) -> Dict[str, np.ndarray]:
    end_frame, repeat_times = _parse_slow_motion_params(
        slow_motion_end_frame, slow_down_times, motion_length
    )
    if end_frame is None:
        return motion
    stretched: Dict[str, np.ndarray] = {}
    for key, value in motion.items():
        value = np.asarray(value)
        if value.ndim > 0 and value.shape[0] == motion_length:
            stretched[key] = _stretch_prefix_along_time(value, end_frame, repeat_times)
        else:
            stretched[key] = value.copy()
    new_len = end_frame * repeat_times + (motion_length - end_frame)
    print(
        f"[scene_replay] {label} slow prefix: first {end_frame} frames x{repeat_times} "
        f"(length {motion_length} -> {new_len})",
        flush=True,
    )
    return stretched


def _load_motion(
    path: str,
    slow_motion_end_frame: Any = None,
    slow_down_times: Any = None,
) -> Dict[str, np.ndarray]:
    data = np.load(path)
    required = ("joint_pos", "body_pos_w", "body_quat_w")
    for key in required:
        if key not in data:
            raise KeyError(f"Motion file missing required key `{key}`: {path}")
    motion = {k: np.asarray(data[k]) for k in data.files}
    motion_length = int(motion["joint_pos"].shape[0])
    return _stretch_motion_dict_prefix_if_enabled(
        motion,
        motion_length,
        slow_motion_end_frame,
        slow_down_times,
        label="ref_motion",
    )


def _load_object_motion(
    path: str,
    slow_motion_end_frame: Any = None,
    slow_down_times: Any = None,
) -> Dict[str, np.ndarray]:
    data = np.load(path)
    required = ("object_trans", "object_quat_wxyz")
    for key in required:
        if key not in data:
            raise KeyError(f"Object motion file missing required key `{key}`: {path}")
    motion = {k: np.asarray(data[k]) for k in data.files}
    motion_length = int(motion["object_trans"].shape[0])
    return _stretch_motion_dict_prefix_if_enabled(
        motion,
        motion_length,
        slow_motion_end_frame,
        slow_down_times,
        label="object_motion",
    )


_FOOT_SOLE_OFFSET_Z = 0.03
_CONTACT_VIZ_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
_CONTACT_VIZ_WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
_CONTACT_VIZ_PELVIS_BODY = "pelvis"


class _ContactVizBodies(NamedTuple):
    foot_l: int
    foot_r: int
    wrist_l: int
    wrist_r: int
    pelvis: int


def _load_contact_mask(
    cfg: Dict[str, Any],
    config_dir: Path,
    workspace_root: Path,
    motion_length: int,
    slow_motion_end_frame: Any,
    slow_down_times: Any,
) -> Optional[np.ndarray]:
    path_raw = str(cfg.get("contact_labels_path", "") or "").strip()
    if not path_raw or path_raw.lower() == "default":
        return None
    path = _resolve_path(path_raw, config_dir, workspace_root)
    labels = np.load(path, allow_pickle=True).item()
    if not isinstance(labels, dict) or "contact_mask" not in labels:
        raise ValueError(f'Contact labels file must be a dict with "contact_mask": {path}')
    mask = np.asarray(labels["contact_mask"], dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"contact_mask must be 2D (T, C), got shape {mask.shape}")
    raw_len = int(mask.shape[0])
    end_frame, repeat_times = _parse_slow_motion_params(
        slow_motion_end_frame, slow_down_times, raw_len
    )
    if end_frame is not None:
        mask = _stretch_prefix_along_time(mask, end_frame, repeat_times).astype(np.float32)
        new_len = end_frame * repeat_times + (raw_len - end_frame)
        print(
            f"[scene_replay] contact_mask slow prefix: first {end_frame} frames x{repeat_times} "
            f"(length {raw_len} -> {new_len})",
            flush=True,
        )
    if mask.shape[0] < motion_length:
        pad = np.zeros((motion_length - mask.shape[0], mask.shape[1]), dtype=np.float32)
        mask = np.vstack([mask, pad])
    elif mask.shape[0] > motion_length:
        mask = mask[:motion_length]
    return mask


def _contact_active(value: float) -> bool:
    return float(value) > 0.5


def _contact_viz_flags(row: np.ndarray) -> Tuple[bool, bool, bool, bool, bool]:
    """Return (L foot, R foot, L wrist, R wrist, pelvis) on/off for visualization."""
    m = np.asarray(row, dtype=np.float32).ravel()
    nc = int(m.size)
    pelvis = False
    if nc >= 10:
        lf = _contact_active(m[0]) or _contact_active(m[1])
        rf = _contact_active(m[2]) or _contact_active(m[3])
        lh = _contact_active(m[4]) or _contact_active(m[5])
        rh = _contact_active(m[6]) or _contact_active(m[7])
        pelvis = _contact_active(m[8]) or _contact_active(m[9])
    elif nc == 8:
        lf = _contact_active(m[0]) or _contact_active(m[1])
        rf = _contact_active(m[2]) or _contact_active(m[3])
        lh = _contact_active(m[4]) or _contact_active(m[5])
        rh = _contact_active(m[6]) or _contact_active(m[7])
    elif nc >= 5:
        lf, rf, lh, rh = (_contact_active(m[i]) for i in range(4))
        pelvis = _contact_active(m[4])
    elif nc >= 4:
        lf, rf, lh, rh = (_contact_active(m[i]) for i in range(4))
    elif nc >= 2:
        lf, rf = _contact_active(m[0]), _contact_active(m[1])
        lh, rh = False, False
    else:
        lf = rf = lh = rh = False
    return lf, rf, lh, rh, pelvis


def _resolve_contact_viz_bodies(model: mujoco.MjModel) -> _ContactVizBodies:
    def body_id(name: str) -> int:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body not found for contact visualization: {name}")
        return int(bid)

    return _ContactVizBodies(
        body_id(_CONTACT_VIZ_FOOT_BODIES[0]),
        body_id(_CONTACT_VIZ_FOOT_BODIES[1]),
        body_id(_CONTACT_VIZ_WRIST_BODIES[0]),
        body_id(_CONTACT_VIZ_WRIST_BODIES[1]),
        body_id(_CONTACT_VIZ_PELVIS_BODY),
    )


def _foot_sole_world_pos(data: mujoco.MjData, body_id: int) -> np.ndarray:
    p = data.xpos[body_id]
    r = data.xmat[body_id].reshape(3, 3)
    sole = r @ np.array([0.0, 0.0, -_FOOT_SOLE_OFFSET_Z], dtype=np.float64)
    return (p + sole).astype(np.float64)


def _contact_marker_rgba(on: bool, active_rgb: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    if on:
        return (active_rgb[0], active_rgb[1], active_rgb[2], 0.92)
    return (0.42, 0.42, 0.42, 0.28)


def _draw_contact_markers(
    viewer: mujoco.viewer.Handle,
    data: mujoco.MjData,
    bodies: _ContactVizBodies,
    contact_mask: np.ndarray,
    frame_idx: int,
) -> None:
    idx = min(int(frame_idx), int(contact_mask.shape[0]) - 1)
    lf_on, rf_on, lh_on, rh_on, pelvis_on = _contact_viz_flags(contact_mask[idx])

    geoms = viewer.user_scn.geoms
    mat_id = np.eye(3, dtype=np.float64).flatten()
    n = 0
    r_foot, r_wrist, r_pelvis = 0.042, 0.034, 0.038

    def add_sphere(pos: np.ndarray, radius: float, rgba: Tuple[float, float, float, float]) -> None:
        nonlocal n
        mujoco.mjv_initGeom(
            geoms[n],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0.0, 0.0],
            pos=pos,
            mat=mat_id,
            rgba=np.array(rgba, dtype=np.float64),
        )
        n += 1

    add_sphere(
        _foot_sole_world_pos(data, bodies.foot_l),
        r_foot,
        _contact_marker_rgba(lf_on, (0.12, 0.95, 0.22)),
    )
    add_sphere(
        _foot_sole_world_pos(data, bodies.foot_r),
        r_foot,
        _contact_marker_rgba(rf_on, (0.2, 0.45, 1.0)),
    )
    add_sphere(
        data.xpos[bodies.wrist_l].copy(),
        r_wrist,
        _contact_marker_rgba(lh_on, (0.98, 0.86, 0.12)),
    )
    add_sphere(
        data.xpos[bodies.wrist_r].copy(),
        r_wrist,
        _contact_marker_rgba(rh_on, (0.98, 0.2, 0.75)),
    )
    add_sphere(
        data.xpos[bodies.pelvis].copy(),
        r_pelvis,
        _contact_marker_rgba(pelvis_on, (1.0, 0.55, 0.12)),
    )
    viewer.user_scn.ngeom = n


def _is_floor_geom(model: mujoco.MjModel, geom_id: int, geom_name: str) -> bool:
    return model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE or any(
        tok in geom_name.lower() for tok in ("floor", "ground", "plane")
    )


def _apply_replay_colors(
    model: mujoco.MjModel,
    object_body_id: Optional[int],
    floor_mode: str = "default",
) -> None:
    """Replay-only visualization tweak: floor appearance and blue object."""
    floor_rgba_by_mode = {
        "default": np.array([0.55, 0.55, 0.55, 1.0], dtype=np.float32),
        "transparent": np.array([0.55, 0.55, 0.55, 0.18], dtype=np.float32),
        "hidden": np.array([0.55, 0.55, 0.55, 0.0], dtype=np.float32),
    }
    floor_rgba = floor_rgba_by_mode[floor_mode]
    object_rgba = np.array([0.20, 0.45, 0.95, 1.0], dtype=np.float32)

    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if _is_floor_geom(model, geom_id, geom_name):
            # Remove floor material to avoid checker/grid textures in replay.
            model.geom_matid[geom_id] = -1
            model.geom_rgba[geom_id, :4] = floor_rgba

    if object_body_id is not None:
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) == int(object_body_id):
                model.geom_rgba[geom_id, :4] = object_rgba


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay robot reference motion and optional object motion from experiment config."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    parser.add_argument("--loop", action="store_true", help="Loop playback forever.")
    parser.add_argument("--fps", type=float, default=None, help="Override playback FPS.")
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Play continuously. Default is step-by-step (press Enter each frame).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Override ref_motion_start_index from config.",
    )
    parser.add_argument(
        "--floor",
        choices=("default", "hidden", "transparent"),
        default="default",
        help="Floor plane in viewer: default (opaque grey), hidden (invisible), or transparent.",
    )
    args = parser.parse_args()

    cfg, config_dir = _load_config(args.config)
    workspace_root = Path(__file__).resolve().parent
    xml_path = _resolve_path(cfg["mujoco_xml_path"], config_dir, workspace_root)
    ref_motion_path = _resolve_path(cfg["ref_motion_path"], config_dir, workspace_root)

    with open(xml_path, "r", encoding="utf-8") as f:
        base_scene_xml = f.read()
    merged_scene_xml = _merge_scene_xml(base_scene_xml, cfg, config_dir, workspace_root)

    # Keep merged xml on disk for debugging and parity with existing workflow.
    temp_xml = tempfile.NamedTemporaryFile(prefix="scene_replay_", suffix=".xml", delete=False)
    temp_xml_path = Path(temp_xml.name)
    temp_xml.write(merged_scene_xml.encode("utf-8"))
    temp_xml.flush()
    temp_xml.close()

    model = mujoco.MjModel.from_xml_string(merged_scene_xml)
    data = mujoco.MjData(model)

    slow_motion_end_frame = cfg.get("slow_motion_end_frame")
    slow_down_times = cfg.get("slow_down_times")
    motion = _load_motion(
        ref_motion_path,
        slow_motion_end_frame=slow_motion_end_frame,
        slow_down_times=slow_down_times,
    )
    motion_length = int(motion["joint_pos"].shape[0])
    start_idx = _clamp_start_index(
        cfg.get("ref_motion_start_index", 0) if args.start_index is None else args.start_index,
        motion_length,
    )

    control_dt = float(cfg.get("control_dt", 0.02))
    fps = args.fps if args.fps is not None else (1.0 / control_dt if control_dt > 0 else 50.0)
    if fps <= 0:
        raise ValueError("fps must be positive")
    frame_dt = 1.0 / fps

    object_motion = None
    object_qposadr = None
    object_body_id = None
    object_motion_path_raw = str(cfg.get("object_motion", "") or "").strip()
    if object_motion_path_raw:
        object_motion = _load_object_motion(
            _resolve_path(object_motion_path_raw, config_dir, workspace_root),
            slow_motion_end_frame=slow_motion_end_frame,
            slow_down_times=slow_down_times,
        )
        for candidate_name in ("floating_object", "free_box"):
            try:
                object_body_id = model.body(candidate_name).id
                break
            except KeyError:
                continue
        if object_body_id is None:
            raise RuntimeError(
                "object_motion is configured, but neither `floating_object` nor `free_box` "
                "exists in merged scene."
            )
        try:
            object_jnt_id = model.body_jntadr[object_body_id]
            object_qposadr = int(model.jnt_qposadr[object_jnt_id])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "object_motion is configured, but the object body is not free-jointed in merged scene."
            ) from e

    _apply_replay_colors(model, object_body_id, floor_mode=args.floor)

    contact_mask = _load_contact_mask(
        cfg,
        config_dir,
        workspace_root,
        motion_length,
        slow_motion_end_frame,
        slow_down_times,
    )
    contact_bodies: Optional[_ContactVizBodies] = None
    if contact_mask is not None:
        contact_bodies = _resolve_contact_viz_bodies(model)

    print(f"[scene_replay] Loaded config: {Path(args.config).resolve()}", flush=True)
    print(f"[scene_replay] Temporary merged scene: {temp_xml_path}", flush=True)
    print(f"[scene_replay] Motion frames: {motion_length}, start={start_idx}, fps={fps:.2f}", flush=True)
    if args.floor != "default":
        print(f"[scene_replay] Floor mode: {args.floor}", flush=True)
    if object_motion is not None:
        print(f"[scene_replay] Object motion frames: {len(object_motion['object_trans'])}", flush=True)
    if contact_mask is not None:
        print(
            f"[scene_replay] Contact viz: (T, C)={contact_mask.shape} — "
            "L foot=green, R foot=blue, L wrist=yellow, R wrist=magenta, pelvis=orange (dim=off)",
            flush=True,
        )
    if args.autoplay:
        print("[scene_replay] Mode: autoplay (Enter to start, q+Enter to quit)", flush=True)
    else:
        print("[scene_replay] Mode: step-by-step (Enter=next, q+Enter=quit)", flush=True)

    def _apply_frame(idx: int) -> None:
        data.qpos[:3] = motion["body_pos_w"][idx, 0]
        # ref body_quat_w stores [w, x, y, z], matching MuJoCo qpos quaternion layout.
        data.qpos[3:7] = motion["body_quat_w"][idx, 0]
        data.qpos[7:36] = motion["joint_pos"][idx, ISAAC_TO_MUJOCO]
        if object_motion is not None and object_qposadr is not None:
            obj_idx = min(idx, len(object_motion["object_trans"]) - 1)
            data.qpos[object_qposadr : object_qposadr + 3] = object_motion["object_trans"][obj_idx]
            data.qpos[object_qposadr + 3 : object_qposadr + 7] = object_motion["object_quat_wxyz"][obj_idx]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    viewer = mujoco.viewer.launch_passive(model, data)

    def _sync_frame(idx: int) -> None:
        _apply_frame(idx)
        if contact_mask is not None and contact_bodies is not None:
            _draw_contact_markers(viewer, data, contact_bodies, contact_mask, idx)
        viewer.sync()

    frame_idx = start_idx
    if args.autoplay:
        _sync_frame(start_idx)
        cmd = input("Press Enter to start autoplay (q to quit): ").strip().lower()
        if cmd in {"q", "quit", "exit"}:
            viewer.close()
            return
        frame_idx = start_idx + 1
        next_tick = time.perf_counter()

    while viewer.is_running():
        _sync_frame(frame_idx)

        frame_idx += 1
        if frame_idx >= motion_length:
            if not args.loop:
                break
            frame_idx = start_idx

        if args.autoplay:
            next_tick += frame_dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()
        else:
            cmd = input("Press Enter for next frame (q to quit): ").strip().lower()
            if cmd in {"q", "quit", "exit"}:
                break

    viewer.close()


if __name__ == "__main__":
    main()
