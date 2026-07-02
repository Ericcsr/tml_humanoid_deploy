#!/usr/bin/env python3
"""
Extract per-frame foot/hand contact labels from a deploy reference motion (.npz).

Input format matches control / ref_motion_visualizer: keys
  joint_pos (T, 29) Isaac joint order
  joint_vel (T, 29)
  body_pos_w  (T, B, 3) — index 0 = root
  body_quat_w (T, B, 4) — w, x, y, z per RL policy conventions
Optional: fps (scalar). If missing, use --fps (match the motion sampling rate used at export).

Detection follows contact_terrain_reconstruction robot-first footing (RobotContactModule +
KinematicsModule + merge_contact_nodes_to_mask from generate_robot_contact_terrain), but runs
entirely in this repo:
  - Joint order: Isaac → MuJoCo via utils.params.ISAAC_TO_MUJOCO (deploy convention).
  - FPS: passes into filtering / gradients (contact_terrain NPZ defaults are often 30 Hz;
          deploy is often 50 Hz — set --fps accordingly).

Output: .npy dict loadable as RLContactPolicy expects:
  np.load(path, allow_pickle=True).item() -> {"contact_mask": (T, 4) float32}
Columns: [left_foot, right_foot, left_wrist, right_wrist] (same as terrain pipeline).

Optional --heightmap: *_heightmap.npy dict with hmap, grid_res, grid_size — enables hand-terrain
pass (crawl) using wrist link positions vs surface, analogous to RobotHandTerrainModule. Deploy
g1_29dof uses left_wrist_yaw_link / right_wrist_yaw_link (not rubber hands).

Optional --detect-hand-motion-contact: same velocity/acceleration heuristic as feet, applied to
wrist link world positions (no heightmap). By default, hand contact before the first wrist
movement (see --hand-movement-min-displacement) is suppressed; pass --hand-contact-from-start
to label from frame 0.

Optional --visualize: passive MuJoCo viewer — plays back the motion and draws spheres at foot soles
and wrists; green/blue/yellow/magenta when that limb is in contact (per contact_mask), dim gray when not.

Optional --terrain-mesh: write a static URDF (same pattern as exported_policies/terrain_test/*.urdf) for the
mesh; default output is '<mesh_stem>_terrain.urdf' next to the mesh (--terrain-urdf-out to override).
If the mesh has multiple edge-disconnected components, they are written as separate OBJ files
(<stem>_terrain_part000.obj, ...) and combined in one URDF (fixed joints from base_link). Use
--terrain-no-split to keep a single mesh reference (original behavior).
With --visualize, the terrain is merged into the scene (utils.urdf_to_mujoco.merge_terrain_into_scene) so
the mesh / column collision is shown. Use --terrain-no-columns if the mesh is STL (heightmap columns use OBJ).

Each motion frame logs left/right foot sole world z (height) after FK; use --no-print-foot-heights to disable.

Usage (run from repo root, or pass absolute --robot-xml):
  python extract_contact_labels_from_motion.py --ref-motion path/to/ref.npz --output contacts.npy \\
      --fps 50 --robot-xml assets/g1/scene_29dof.xml
"""

from __future__ import annotations

import argparse
import os
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
from scipy import signal

import mujoco
import mujoco.viewer

from utils.params import ISAAC_TO_MUJOCO
import qpos_to_motion

HAND_VIZ_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")

# Same sole offset convention as contact_terrain_reconstruction.robot_terrain_module
ROBOT_FOOT_SOLE_OFFSET_Z = 0.03
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
DEFAULT_DETECT_CONFIG = {
    "FILTER_ORDER": 4,
    "FILTER_CUTOFF": 6.0,
    "VEL_THRESH_RANGE": (0.05, 0.5),
    "ACC_THRESH_RANGE": (1.0, 6.0),
    "ENERGY_SCALE": 0.5,
    "ENERGY_GAUSSIAN_SIGMA": 10,
}

# --------------------------------------------------------------------------- #
# Joint / body geometry
# --------------------------------------------------------------------------- #


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_robot(robot_xml: Path):
    """
    Match mujoco_env.run_simulation: read MJCF as a string and compile with from_xml_string
    while cwd is the deploy repo root. That way compiler meshdir paths in assets/g1/*.xml
    resolve like the simulator (from_xml_path would nest paths relative to the xml file dir).
    """
    repo = _repo_root()
    path = robot_xml if robot_xml.is_absolute() else (repo / robot_xml)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Robot XML not found: {path}")
    old_cwd = os.getcwd()
    try:
        os.chdir(repo)
        with open(path, "r") as f:
            xml = f.read()
        model = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(old_cwd)
    data = mujoco.MjData(model)
    return model, data


# Vertex quantization for shared-edge detection (matches typical mesh export precision).
_MESH_VERTEX_ROUND = 6


def _mesh_vertex_key(v: np.ndarray) -> tuple[float, float, float]:
    return (round(float(v[0]), _MESH_VERTEX_ROUND), round(float(v[1]), _MESH_VERTEX_ROUND), round(float(v[2]), _MESH_VERTEX_ROUND))


def _mesh_edge_key(a: np.ndarray, b: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ka, kb = _mesh_vertex_key(a), _mesh_vertex_key(b)
    return (ka, kb) if ka <= kb else (kb, ka)


def load_mesh_triangles(mesh_path: Path) -> np.ndarray:
    """
    Load triangle soup as (N, 3, 3) float64. Supports .obj and .stl (ASCII or binary).
    """
    mesh_path = mesh_path.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Terrain mesh not found: {mesh_path}")
    suf = mesh_path.suffix.lower()
    if suf == ".obj":
        return _load_triangles_from_obj(mesh_path)
    if suf == ".stl":
        return _load_triangles_from_stl(mesh_path)
    raise ValueError(f"Unsupported terrain mesh extension {suf!r} (use .obj or .stl)")


def _load_triangles_from_obj(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idxs = []
                for p in parts[1:]:
                    vi = p.split("/")[0]
                    idxs.append(int(vi))
                # Triangulate fan (blender-style ngons)
                for k in range(1, len(idxs) - 1):
                    a, b, c = idxs[0], idxs[k], idxs[k + 1]
                    triangles.append([a, b, c])
    if not vertices or not triangles:
        raise ValueError(f"No mesh geometry in OBJ: {path}")
    v = np.asarray(vertices, dtype=np.float64)
    tris = np.zeros((len(triangles), 3, 3), dtype=np.float64)
    nvert = len(vertices)
    for ti, tri in enumerate(triangles):
        for j, ix in enumerate(tri):
            ii = ix - 1 if ix > 0 else nvert + ix  # OBJ 1-based; negative = relative
            if ii < 0 or ii >= nvert:
                raise ValueError(f"Invalid vertex index {ix} in {path}")
            tris[ti, j] = v[ii]
    return tris


def _load_triangles_from_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        n_tri = struct.unpack("<I", data[80:84])[0]
        if 84 + n_tri * 50 == len(data):
            return _load_triangles_from_stl_binary(data)
    text = data.decode("utf-8", errors="replace")
    if "facet" in text.lower() and "vertex" in text.lower():
        return _load_triangles_from_stl_ascii(text)
    raise ValueError(f"Unrecognized STL format: {path}")


def _load_triangles_from_stl_binary(data: bytes) -> np.ndarray:
    n = struct.unpack("<I", data[80:84])[0]
    tris = np.empty((n, 3, 3), dtype=np.float64)
    off = 84
    for i in range(n):
        chunk = data[off : off + 50]
        off += 50
        v = struct.unpack("<9f", chunk[12:48])
        tris[i, 0] = v[0:3]
        tris[i, 1] = v[3:6]
        tris[i, 2] = v[6:9]
    return tris


def _load_triangles_from_stl_ascii(text: str) -> np.ndarray:
    tris: list[list[list[float]]] = []
    lines = [ln.strip() for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        if lines[i].lower().startswith("outer loop"):
            i += 1
            vs: list[list[float]] = []
            for _ in range(3):
                if i >= len(lines):
                    break
                parts = lines[i].split()
                if len(parts) < 4 or parts[0].lower() != "vertex":
                    break
                vs.append([float(parts[1]), float(parts[2]), float(parts[3])])
                i += 1
            if len(vs) == 3:
                tris.append(vs)
            continue
        i += 1
    if not tris:
        raise ValueError("No triangles parsed from ASCII STL")
    return np.asarray(tris, dtype=np.float64)


def split_triangles_by_edge_connectivity(tris: np.ndarray) -> list[np.ndarray]:
    """
    Split (N,3,3) triangle array into edge-connected components (triangles sharing an edge).
    Isolated triangles (no shared edges) are each their own component.
    """
    n = tris.shape[0]
    if n == 0:
        return []
    edge_to_tris: dict[tuple, list[int]] = defaultdict(list)
    for ti in range(n):
        t = tris[ti]
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_to_tris[_mesh_edge_key(a, b)].append(ti)

    adj: list[set[int]] = [set() for _ in range(n)]
    for tlist in edge_to_tris.values():
        if len(tlist) < 2:
            continue
        for j in range(len(tlist)):
            for k in range(j + 1, len(tlist)):
                u, v = tlist[j], tlist[k]
                adj[u].add(v)
                adj[v].add(u)

    seen = [False] * n
    out: list[np.ndarray] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        idxs: list[int] = []
        while stack:
            u = stack.pop()
            idxs.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        out.append(tris[idxs])
    return out


def write_obj_triangles(path: Path, tris: np.ndarray) -> None:
    """Write (N,3,3) triangle soup as a minimal OBJ (deduplicated vertices)."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    vert_map: dict[tuple[float, float, float], int] = {}
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for ti in range(tris.shape[0]):
        tri_idx: list[int] = []
        for k in range(3):
            key = _mesh_vertex_key(tris[ti, k])
            if key not in vert_map:
                vert_map[key] = len(verts) + 1
                verts.append(key)
            tri_idx.append(vert_map[key])
        faces.append((tri_idx[0], tri_idx[1], tri_idx[2]))
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    for a, b, c in faces:
        lines.append(f"f {a} {b} {c}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _urdf_link_block(
    link_name: str,
    rel_mesh: str,
    visual_name: str,
    collision_name: str = "terrain_collision",
) -> str:
    return f"""    <link name="{link_name}">
        <visual name="{visual_name}">
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{rel_mesh}" scale="1 1 1"/>
            </geometry>
            <material name="terrain_material">
                <color rgba="0.5 0.5 0.5 1.0"/>
            </material>
        </visual>
        <collision name="{collision_name}">
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{rel_mesh}" scale="1 1 1"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="0"/>
            <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>
    </link>
"""


def write_terrain_urdf_from_mesh(
    mesh_path: Path,
    urdf_out: Path,
    *,
    split_components: bool = True,
) -> None:
    """
    Write a minimal static URDF referencing the mesh (relative path from URDF directory).
    Matches the style of exported_policies/terrain_test/*_terrain.urdf for merge_terrain_into_scene.

    When split_components is True, the mesh is decomposed into edge-connected triangle components.
    Multiple components are saved as separate OBJ files next to urdf_out and combined via fixed
    joints from an empty base_link (utils.urdf_to_mujoco already iterates all links).
    """
    mesh_path = mesh_path.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Terrain mesh not found: {mesh_path}")
    urdf_out = urdf_out.expanduser().resolve()
    urdf_out.parent.mkdir(parents=True, exist_ok=True)

    if not split_components:
        rel = os.path.relpath(mesh_path, urdf_out.parent).replace("\\", "/")
        xml = f"""<?xml version="1.0"?>
<robot name="terrain">
{_urdf_link_block("base_link", rel, "terrain_visual")}
</robot>
"""
        urdf_out.write_text(xml, encoding="utf-8")
        print(f"[terrain] Wrote URDF {urdf_out} (--terrain-no-split)", flush=True)
        return

    tris = load_mesh_triangles(mesh_path)
    parts = split_triangles_by_edge_connectivity(tris)
    if len(parts) == 1:
        rel = os.path.relpath(mesh_path, urdf_out.parent).replace("\\", "/")
        xml = f"""<?xml version="1.0"?>
<robot name="terrain">
{_urdf_link_block("base_link", rel, "terrain_visual")}
</robot>
"""
        urdf_out.write_text(xml, encoding="utf-8")
        print(f"[terrain] Wrote URDF {urdf_out} (single connected mesh)", flush=True)
        return

    stem = urdf_out.stem
    blocks: list[str] = [
        '<?xml version="1.0"?>',
        '<robot name="terrain">',
        "    <link name=\"base_link\">",
        "        <inertial>",
        '            <mass value="0"/>',
        '            <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>',
        "        </inertial>",
        "    </link>",
    ]
    for i, part_tris in enumerate(parts):
        file_stem = f"{stem}_part{i:03d}"
        link_name = f"terrain_part{i:03d}"
        mesh_part_path = urdf_out.parent / f"{file_stem}.obj"
        write_obj_triangles(mesh_part_path, part_tris)
        rel = os.path.relpath(mesh_part_path, urdf_out.parent).replace("\\", "/")
        joint_name = f"{file_stem}_joint"
        blocks.append(
            f'    <joint name="{joint_name}" type="fixed">\n'
            f'        <parent link="base_link"/>\n'
            f'        <child link="{link_name}"/>\n'
            f'        <origin xyz="0 0 0" rpy="0 0 0"/>\n'
            f"    </joint>"
        )
        blocks.append(
            _urdf_link_block(
                link_name,
                rel,
                f"terrain_visual_{i}",
                collision_name=f"terrain_collision_{i}",
            )
        )
    blocks.append("</robot>")
    urdf_out.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(
        f"[terrain] Wrote URDF {urdf_out} ({len(parts)} disconnected components -> {stem}_part*.obj)",
        flush=True,
    )


def _load_scene_with_terrain(
    robot_xml: Path,
    terrain_urdf: Path,
    *,
    use_columns_for_collision: bool,
    terrain_column_res: float,
    terrain_floor_threshold: float,
    terrain_urdf_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load robot scene XML from disk and merge terrain URDF (same as MujocoRobot / urdf_to_mujoco)."""
    from utils.urdf_to_mujoco import merge_terrain_into_scene

    repo = _repo_root()
    scene_path = robot_xml if robot_xml.is_absolute() else (repo / robot_xml)
    scene_path = scene_path.resolve()
    terrain_path = terrain_urdf.expanduser().resolve()
    if not terrain_path.is_file():
        raise FileNotFoundError(f"Terrain URDF not found: {terrain_path}")

    merged = merge_terrain_into_scene(
        str(scene_path),
        str(terrain_path),
        use_columns_for_collision=use_columns_for_collision,
        terrain_column_res=terrain_column_res,
        terrain_floor_threshold=terrain_floor_threshold,
        terrain_urdf_offset=terrain_urdf_offset,
    )
    old_cwd = os.getcwd()
    try:
        os.chdir(repo)
        model = mujoco.MjModel.from_xml_string(merged)
    finally:
        os.chdir(old_cwd)
    return model, mujoco.MjData(model)


def ref_motion_to_mujoco_qpos(
    joint_pos_isaac: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w_wxyz: np.ndarray,
) -> np.ndarray:
    """(T,29) Isaac joints + root -> (T, nq) MuJoCo qpos (free joint + actuated)."""
    root_p = body_pos_w[:, 0, :].astype(np.float64)
    # Stored w,x,y,z; MuJoCo also uses wxyz for free joint quaternion.
    root_q = body_quat_w_wxyz[:, 0, :].astype(np.float64)
    q_j = joint_pos_isaac[:, ISAAC_TO_MUJOCO].astype(np.float64)
    return np.concatenate([root_p, root_q, q_j], axis=1)


def compute_foot_sole_world_positions(model, data, qpos: np.ndarray) -> np.ndarray:
    """(T, 2, 3) left / right foot sole world positions (sole below ankle body frame)."""
    n = qpos.shape[0]
    out = np.zeros((n, 2, 3), dtype=np.float64)
    body_ids = []
    for name in FOOT_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body not found in {model.names}: {name}")
        body_ids.append(bid)
    for t in range(n):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for i, bid in enumerate(body_ids):
            p = data.xpos[bid].copy()
            r = data.xmat[bid].reshape(3, 3)
            sole = r @ np.array([0.0, 0.0, -ROBOT_FOOT_SOLE_OFFSET_Z])
            out[t, i] = p + sole
    return out


def compute_body_positions(model, data, qpos: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    n = qpos.shape[0]
    out = np.zeros((n, len(names), 3), dtype=np.float64)
    bids = []
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body not found: {name}")
        bids.append(bid)
    for t in range(n):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for i, bid in enumerate(bids):
            out[t, i] = data.xpos[bid].copy()
    return out


# --------------------------------------------------------------------------- #
# Kinematics + contact (robot-first foot heuristic)
# --------------------------------------------------------------------------- #


def kinematics_process(
    pos: np.ndarray,
    fps: float,
    filter_order: int,
    filter_cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pos.shape[0] < 20:
        vel = np.gradient(pos, axis=0) * fps
        acc = np.gradient(vel, axis=0) * fps
        return pos, vel, acc
    b, a = signal.butter(filter_order, filter_cutoff_hz / (0.5 * fps), btype="low")
    pos_f = signal.filtfilt(b, a, pos, axis=0)
    vel = np.gradient(pos_f, axis=0) * fps
    acc = np.gradient(vel, axis=0) * fps
    return pos_f, vel, acc


def detect_motion_contact_nodes(
    pos_filt: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    limb_joint_names: tuple[str, str],
    config: dict,
    *,
    contact_label: str = "floor",
) -> list[dict]:
    """Velocity/acceleration contact heuristic (RobotContactModule.detect_feet equivalent)."""
    joint_speeds = np.linalg.norm(vel, axis=2)
    body_energy = np.mean(joint_speeds, axis=1)
    energy_smooth = ndimage.gaussian_filter1d(
        body_energy, sigma=config["ENERGY_GAUSSIAN_SIGMA"]
    )
    alpha = np.clip(energy_smooth * config["ENERGY_SCALE"], 0.0, 1.0)
    v_min, v_max = config["VEL_THRESH_RANGE"]
    a_min, a_max = config["ACC_THRESH_RANGE"]
    adaptive_v = v_min + (v_max - v_min) * alpha
    adaptive_a = a_min + (a_max - a_min) * alpha

    nodes: list[dict] = []
    for t in range(pos_filt.shape[0]):
        cv, ca = adaptive_v[t], adaptive_a[t]
        for i in range(2):
            v = np.linalg.norm(vel[t, i])
            a = np.linalg.norm(acc[t, i])
            if v < cv and a < ca:
                nodes.append(
                    {
                        "position": pos_filt[t, i].copy(),
                        "timestamp": int(t),
                        "label": contact_label,
                        "entity_info": {
                            "joint_name": limb_joint_names[i],
                            "joint_index": int(i),
                        },
                    }
                )
    return nodes


def detect_foot_contact_nodes(
    pos_filt: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    foot_joint_names: tuple[str, str],
    config: dict,
) -> list[dict]:
    """Same criterion as RobotContactModule.detect_feet; returns graph-style node dicts."""
    return detect_motion_contact_nodes(
        pos_filt, vel, acc, foot_joint_names, config, contact_label="floor"
    )


def detect_hand_terrain_nodes(
    hand_pos: np.ndarray,
    hand_names: tuple[str, str],
    hmap: np.ndarray,
    grid_res: float,
    grid_size: float,
    z_tolerance: float,
    *,
    min_timestamp: int = 0,
) -> list[dict]:
    """RobotHandTerrainModule.detect equivalent (no SceneInteractionGraph)."""
    half = grid_size / 2.0
    dim = hmap.shape[0]
    nodes: list[dict] = []
    for t in range(hand_pos.shape[0]):
        if t < min_timestamp:
            continue
        for i in range(2):
            x, y, z = hand_pos[t, i]
            u = int((x + half) / grid_res)
            v = int((y + half) / grid_res)
            if u < 0 or u >= dim or v < 0 or v >= dim:
                continue
            tz = float(hmap[v, u])
            if abs(z - tz) <= z_tolerance:
                nodes.append(
                    {
                        "position": hand_pos[t, i].copy(),
                        "timestamp": int(t),
                        "label": "hand_terrain",
                        "entity_info": {
                            "joint_name": hand_names[i],
                            "joint_index": int(i),
                        },
                    }
                )
    return nodes


def first_hand_movement_frame(
    hand_pos: np.ndarray,
    *,
    min_displacement: float = 0.03,
) -> int:
    """First frame where either wrist moves min_displacement (m) from its frame-0 pose.

    Returns ``hand_pos.shape[0]`` if neither wrist ever moves that far (suppress all hand contact).
    """
    if hand_pos.shape[0] <= 1 or min_displacement <= 0:
        return 0
    origin = hand_pos[0]  # (2, 3)
    disp = np.linalg.norm(hand_pos - origin[np.newaxis, :, :], axis=2)  # (T, 2)
    moved = np.any(disp >= min_displacement, axis=1)
    idxs = np.flatnonzero(moved)
    return int(idxs[0]) if idxs.size else int(hand_pos.shape[0])


def filter_hand_contact_nodes_before_frame(
    nodes: list[dict],
    start_frame: int,
) -> tuple[list[dict], int]:
    """Drop hand_terrain nodes with timestamp < start_frame. Returns (filtered, dropped_count)."""
    if start_frame <= 0:
        return nodes, 0
    kept: list[dict] = []
    dropped = 0
    for node in nodes:
        if node.get("label") == "hand_terrain" and int(node.get("timestamp", 0)) < start_frame:
            dropped += 1
            continue
        kept.append(node)
    return kept, dropped


# --------------------------------------------------------------------------- #
# merge_contact_nodes_to_mask (contact_label_merge.py), + G1 wrist naming
# --------------------------------------------------------------------------- #

LEFT_FOOT_NAMES = (
    "L_Foot",
    "L_Toe",
    "LeftFoot",
    "LeftToe",
    "LeftToe_EndSite",
    "LeftToeBase",
    "left_ankle_roll_link",
    "left_ankle_pitch_link",
)
RIGHT_FOOT_NAMES = (
    "R_Foot",
    "R_Toe",
    "RightFoot",
    "RightToe",
    "RightToe_EndSite",
    "RightToeBase",
    "right_ankle_roll_link",
    "right_ankle_pitch_link",
)
LEFT_HAND_NAMES = (
    "L_Hand",
    "L_Wrist",
    "LeftHand",
    "LeftForeArm",
    "left_rubber_hand",
    "left_wrist_yaw_link",
)
RIGHT_HAND_NAMES = (
    "R_Hand",
    "R_Wrist",
    "RightHand",
    "RightForeArm",
    "right_rubber_hand",
    "right_wrist_yaw_link",
)


def _entity_to_column(label: str, joint_name: str) -> int | None:
    if label == "floor":
        if any(n in joint_name for n in LEFT_FOOT_NAMES) or (
            "Left" in joint_name and ("Foot" in joint_name or "Toe" in joint_name)
        ):
            return 0
        if any(n in joint_name for n in RIGHT_FOOT_NAMES) or (
            "Right" in joint_name and ("Foot" in joint_name or "Toe" in joint_name)
        ):
            return 1
    elif label == "hand_terrain":
        if any(n in joint_name for n in LEFT_HAND_NAMES) or (
            joint_name.startswith("left_") and "wrist" in joint_name
        ):
            return 2
        if any(n in joint_name for n in RIGHT_HAND_NAMES) or (
            joint_name.startswith("right_") and "wrist" in joint_name
        ):
            return 3
    return None


def merge_contact_nodes_to_mask(
    nodes: list[dict],
    num_frames: int,
    max_time_gap: int = 10,
    max_spatial_dist: float = 0.15,
    gap_fill_frames: int = 5,
    min_contact_duration_frames: int = 0,
    contact_lead_frames: int = 0,
) -> np.ndarray:
    mask = np.zeros((num_frames, 4), dtype=bool)
    groups: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
    for node in nodes:
        ts = node.get("timestamp", 0)
        label = node.get("label", "floor")
        entity = node.get("entity_info", {})
        joint_name = entity.get("joint_name", "")
        pos = np.asarray(node.get("position", [0, 0, 0]), dtype=float)
        col = _entity_to_column(label, joint_name)
        if col is None:
            continue
        if ts < 0 or ts >= num_frames:
            continue
        key = (label, joint_name)
        groups.setdefault(key, []).append((ts, pos))

    for key, items in groups.items():
        col = _entity_to_column(key[0], key[1])
        if col is None:
            continue
        items.sort(key=lambda x: x[0])
        intervals: list[tuple[int, int]] = []
        start, end = items[0][0], items[0][0]
        last_pos = items[0][1]
        for i in range(1, len(items)):
            ts, pos = items[i]
            dt = ts - end
            dist = float(np.linalg.norm(pos - last_pos))
            if dt <= max_time_gap and dist <= max_spatial_dist:
                end = ts
                last_pos = pos
            else:
                intervals.append((start, end))
                start, end = ts, ts
                last_pos = pos
        intervals.append((start, end))

        if gap_fill_frames > 0 and len(intervals) > 1:
            merged: list[tuple[int, int]] = [intervals[0]]
            for s, e in intervals[1:]:
                ls, le = merged[-1]
                gap = s - le - 1
                if gap <= gap_fill_frames and gap >= 0:
                    merged[-1] = (ls, e)
                else:
                    merged.append((s, e))
            intervals = merged

        if min_contact_duration_frames > 0:
            extended: list[tuple[int, int]] = []
            for s, e in intervals:
                dur = e - s + 1
                if dur < min_contact_duration_frames:
                    deficit = min_contact_duration_frames - dur
                    b = deficit // 2
                    a = deficit - b
                    extended.append((max(0, s - b), min(num_frames - 1, e + a)))
                else:
                    extended.append((s, e))
            intervals = extended
            if len(intervals) > 1:
                intervals.sort(key=lambda x: x[0])
                merged2: list[tuple[int, int]] = [intervals[0]]
                for s, e in intervals[1:]:
                    ls, le = merged2[-1]
                    if s <= le + 1:
                        merged2[-1] = (ls, max(le, e))
                    else:
                        merged2.append((s, e))
                intervals = merged2

        for s, e in intervals:
            st = max(0, s - contact_lead_frames)
            mask[st : e + 1, col] = True

    return mask


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #


def _try_compute_hand_world(model, data, qpos: np.ndarray) -> np.ndarray | None:
    try:
        return compute_body_positions(model, data, qpos, HAND_VIZ_BODIES)
    except ValueError:
        return None


def visualize_motion_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    foot_world: np.ndarray,
    hand_world: np.ndarray | None,
    contact_mask: np.ndarray,
    fps: float,
    playback_speed: float = 1.0,
) -> None:
    """
    Passive viewer: replay qpos trajectory and draw contact markers at feet (sole) and wrists.
    Colors when in contact: L foot green, R foot blue, L hand yellow, R hand magenta.
    """
    t_horizon = qpos.shape[0]
    if t_horizon == 0:
        return

    mat_id = np.eye(3, dtype=np.float64).flatten()
    r_foot, r_hand = 0.042, 0.034

    def rgba_contact(on: bool, active: tuple[float, float, float], inactive_alpha: float = 0.28):
        if on:
            return (active[0], active[1], active[2], 0.92)
        return (0.42, 0.42, 0.42, inactive_alpha)

    print(
        "[viz] Markers: L foot=green, R foot=blue, L hand=yellow, R hand=magenta (dim=off). "
        "Terrain shown if scene was loaded with merge_terrain_into_scene. "
        "Close window to continue.",
        flush=True,
    )

    step_dt = 1.0 / max(fps * playback_speed, 1e-6)
    frame = 0
    last_t = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            now = time.time()
            if now - last_t < step_dt:
                time.sleep(0.001)
                continue
            last_t = now

            i = frame % t_horizon
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)

            m = contact_mask[i]
            lf_on = bool(m[0])
            rf_on = bool(m[1])
            lh_on = bool(m[2]) if contact_mask.shape[1] > 2 else False
            rh_on = bool(m[3]) if contact_mask.shape[1] > 3 else False

            geoms = viewer.user_scn.geoms
            n = 0

            def add_sphere(pos, size_r, rgba):
                nonlocal n
                mujoco.mjv_initGeom(
                    geoms[n],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[size_r, 0, 0],
                    pos=pos.astype(np.float64),
                    mat=mat_id,
                    rgba=np.array(rgba, dtype=np.float64),
                )
                n += 1

            add_sphere(foot_world[i, 0], r_foot, rgba_contact(lf_on, (0.12, 0.95, 0.22)))
            add_sphere(foot_world[i, 1], r_foot, rgba_contact(rf_on, (0.2, 0.45, 1.0)))
            if hand_world is not None:
                add_sphere(hand_world[i, 0], r_hand, rgba_contact(lh_on, (0.98, 0.86, 0.12)))
                add_sphere(hand_world[i, 1], r_hand, rgba_contact(rh_on, (0.98, 0.2, 0.75)))

            viewer.user_scn.ngeom = n
            viewer.sync()
            frame += 1


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip())
    parser.add_argument("--ref-motion", type=Path, required=True, help="Deploy ref .npz")
    parser.add_argument("--output", type=Path, required=True, help="Output *_contact_nodes.npy")
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=Path("assets/g1/scene_29dof.xml"),
        help="MJCF path relative to repo root or absolute (same style as config mujoco_xml_path)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Motion sampling rate (Hz). If ref has no 'fps' key, this is required.",
    )
    parser.add_argument(
        "--root-layout",
        choices=["pos_first", "quat_first"],
        default=None,
        help="For qpos-format input only: root layout of the first 7 qpos entries. "
        "'pos_first' = [x, y, z, qw, qx, qy, qz] (default), "
        "'quat_first' = [qw, qx, qy, qz, x, y, z]. Overrides --rot-first if both given.",
    )
    parser.add_argument(
        "--rot-first",
        action="store_true",
        help="For qpos-format input only: alias for --root-layout quat_first.",
    )
    parser.add_argument(
        "--save-converted-motion",
        type=Path,
        default=None,
        help="For qpos-format input only: also save the converted joint+body motion to this "
        ".npz (same schema as omni_to_npz: fps, joint_pos, joint_vel, body_pos_w, body_quat_w, "
        "body_lin_vel_w, body_ang_vel_w, body_names).",
    )
    parser.add_argument("--merge-time-gap", type=int, default=10)
    parser.add_argument("--merge-spatial-dist", type=float, default=0.15)
    parser.add_argument("--merge-gap-fill", type=int, default=5)
    parser.add_argument(
        "--min-contact-duration",
        type=float,
        default=0.2,
        help="Seconds — same default as generate_robot_contact_terrain",
    )
    parser.add_argument(
        "--contact-lead-time",
        type=float,
        default=0.2,
        help="Seconds — extend contact backward each interval",
    )
    parser.add_argument(
        "--heightmap",
        type=Path,
        default=None,
        help="Optional terrain_heightmap npy dict (keys hmap, grid_res, grid_size) for hand pass",
    )
    parser.add_argument(
        "--detect-hand-motion-contact",
        action="store_true",
        help="Detect hand contact with the same velocity/acceleration heuristic as feet "
        "(wrist link FK positions; no heightmap). Can be combined with --heightmap.",
    )
    parser.add_argument(
        "--hand-contact-from-start",
        action="store_true",
        help="Allow hand contact labels from frame 0. Default: ignore hand contact until "
        "either wrist moves --hand-movement-min-displacement from its initial pose "
        "(static start pose is often not in contact).",
    )
    parser.add_argument(
        "--hand-movement-min-displacement",
        type=float,
        default=0.03,
        help="Meters — either wrist must move this far from frame-0 pose before hand contact "
        "is considered (default 0.03). Only used when --hand-contact-from-start is not set.",
    )
    parser.add_argument(
        "--hand-z-tolerance",
        type=float,
        default=0.08,
        help="Hand vs terrain Z tolerance (m)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open MuJoCo viewer: replay motion + foot/hand contact markers",
    )
    parser.add_argument(
        "--vis-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier vs motion fps (default 1.0 = realtime)",
    )
    parser.add_argument(
        "--terrain-mesh",
        type=Path,
        default=None,
        help="Terrain mesh (.obj/.stl); writes URDF for sim (see --terrain-urdf-out)",
    )
    parser.add_argument(
        "--terrain-urdf-out",
        type=Path,
        default=None,
        help="Output path for generated URDF (default: <mesh_dir>/<mesh_stem>_terrain.urdf)",
    )
    parser.add_argument(
        "--terrain-urdf",
        type=Path,
        default=None,
        help="Existing terrain URDF: use for --visualize merge only (ignored if --terrain-mesh is set)",
    )
    parser.add_argument(
        "--terrain-no-columns",
        action="store_true",
        help="Terrain collision = mesh (convex hull); default uses heightmap columns (best with .obj meshes)",
    )
    parser.add_argument(
        "--terrain-column-res",
        type=float,
        default=0.2,
        help="Column grid resolution (m) when using heightmap collision",
    )
    parser.add_argument(
        "--terrain-floor-threshold",
        type=float,
        default=0.02,
        help="Min height (m) for heightmap column collision",
    )
    parser.add_argument(
        "--terrain-urdf-offset",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="World translation (m) for merged terrain URDF (--visualize only); default 0 0 0",
    )
    parser.add_argument(
        "--terrain-no-split",
        action="store_true",
        help="Do not split terrain mesh into connected components; reference the original file only",
    )
    parser.add_argument(
        "--no-print-foot-heights",
        action="store_true",
        help="Disable per-frame sole height lines (world z, left then right)",
    )
    args = parser.parse_args()

    terrain_urdf_for_viz: Path | None = None
    if args.terrain_mesh is not None:
        mesh = args.terrain_mesh.expanduser().resolve()
        urdf_out = (
            args.terrain_urdf_out.expanduser().resolve()
            if args.terrain_urdf_out is not None
            else (mesh.parent / f"{mesh.stem}_terrain.urdf")
        )
        write_terrain_urdf_from_mesh(mesh, urdf_out, split_components=not args.terrain_no_split)
        terrain_urdf_for_viz = urdf_out.resolve()
    elif args.terrain_urdf is not None:
        terrain_urdf_for_viz = args.terrain_urdf.expanduser().resolve()
        if not terrain_urdf_for_viz.is_file():
            parser.error(f"--terrain-urdf not found: {terrain_urdf_for_viz}")

    data = np.load(args.ref_motion, allow_pickle=True)
    model, mj_data = _load_robot(args.robot_xml)

    # Auto-detect raw qpos-format input (e.g. retargeted motion, input to omni_to_npz) and
    # convert it to joint + body-state format with MuJoCo FK before processing.
    if qpos_to_motion.is_qpos_motion(data):
        if args.root_layout is not None:
            rot_first = args.root_layout == "quat_first"
        else:
            rot_first = args.rot_first
        print(
            "[extract] Detected qpos-format input; converting to joint+body data via MuJoCo FK "
            f"(root_layout={'quat_first' if rot_first else 'pos_first'}).",
            flush=True,
        )
        qpos_arr, fps_in = qpos_to_motion.load_qpos_motion(args.ref_motion)
        motion = qpos_to_motion.convert_qpos_to_motion(
            qpos_arr, fps_in, model, rot_first=rot_first, output_fps=None
        )
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
        body_pos_w = np.asarray(motion["body_pos_w"], dtype=np.float64)
        body_quat_w = np.asarray(motion["body_quat_w"], dtype=np.float64)
        if args.fps is None:
            args.fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        if args.save_converted_motion is not None:
            qpos_to_motion.save_motion_npz(args.save_converted_motion, motion)
            print(
                f"[extract] Saved converted motion to {args.save_converted_motion} "
                f"(joint_pos={motion['joint_pos'].shape}, body_pos_w={motion['body_pos_w'].shape})",
                flush=True,
            )
    else:
        if args.save_converted_motion is not None:
            print(
                "[extract] --save-converted-motion ignored: input is already in joint+body "
                "format (not qpos).",
                flush=True,
            )
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)

    t = joint_pos.shape[0]
    if body_pos_w.shape[0] != t or body_quat_w.shape[0] != t:
        raise ValueError("joint_pos / body_pos_w / body_quat_w length mismatch")

    fps = args.fps
    if fps is None:
        if "fps" in data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        elif "control_dt" in data:
            fps = 1.0 / float(np.asarray(data["control_dt"]).reshape(-1)[0])
        else:
            parser.error(
                "No fps in ref motion; pass --fps (e.g. 50.0 for typical deploy / sim export)."
            )

    qpos = ref_motion_to_mujoco_qpos(joint_pos, body_pos_w, body_quat_w)
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"Built qpos width {qpos.shape[1]} but model.nq={model.nq} — robot-xml must match motion"
        )

    foot_world = compute_foot_sole_world_positions(model, mj_data, qpos)
    if not args.no_print_foot_heights:
        print(
            f"foot_height fps={fps:g}  format: frame z_left_sole z_right_sole (world m)",
            flush=True,
        )
        for ti in range(foot_world.shape[0]):
            print(
                f"foot_height {ti} {foot_world[ti, 0, 2]:.6f} {foot_world[ti, 1, 2]:.6f}",
                flush=True,
            )
    cfg = dict(DEFAULT_DETECT_CONFIG)
    pos_f, vel, acc = kinematics_process(
        foot_world, fps, cfg["FILTER_ORDER"], cfg["FILTER_CUTOFF"]
    )
    names_feet = FOOT_BODIES
    nodes = detect_foot_contact_nodes(pos_f, vel, acc, names_feet, cfg)

    hand_links = HAND_VIZ_BODIES
    hand_motion_start = 0
    hand_world_for_gate: np.ndarray | None = None
    if (args.detect_hand_motion_contact or args.heightmap is not None) and not args.hand_contact_from_start:
        hand_world_for_gate = compute_body_positions(model, mj_data, qpos, hand_links)
        hand_motion_start = first_hand_movement_frame(
            hand_world_for_gate,
            min_displacement=args.hand_movement_min_displacement,
        )
        if hand_motion_start >= t:
            print(
                f"[extract] Hand contact gated: no wrist moved >= {args.hand_movement_min_displacement:g} m "
                f"from frame-0 pose; suppressing all hand contact.",
                flush=True,
            )
        elif hand_motion_start > 0:
            print(
                f"[extract] Hand contact gated: ignoring hand contact before frame {hand_motion_start} "
                f"(first wrist movement >= {args.hand_movement_min_displacement:g} m from start).",
                flush=True,
            )

    if args.detect_hand_motion_contact:
        hand_world = (
            hand_world_for_gate
            if hand_world_for_gate is not None
            else compute_body_positions(model, mj_data, qpos, hand_links)
        )
        hand_pos_f, hand_vel, hand_acc = kinematics_process(
            hand_world, fps, cfg["FILTER_ORDER"], cfg["FILTER_CUTOFF"]
        )
        hand_nodes = detect_motion_contact_nodes(
            hand_pos_f, hand_vel, hand_acc, hand_links, cfg, contact_label="hand_terrain"
        )
        hand_nodes, dropped = filter_hand_contact_nodes_before_frame(hand_nodes, hand_motion_start)
        nodes.extend(hand_nodes)
        print(
            f"[extract] Hand motion-contact pass: {len(hand_nodes)} raw nodes "
            f"(dropped {dropped} before movement start; wrists: {hand_links[0]}, {hand_links[1]})",
            flush=True,
        )

    if args.heightmap is not None:
        loaded = np.load(args.heightmap, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            hm = loaded.item()
        else:
            hm = loaded
        if not isinstance(hm, dict):
            raise ValueError("heightmap file must contain dict keys hmap, grid_res, grid_size")
        hmap = np.asarray(hm["hmap"], dtype=np.float64)
        grid_res = float(hm["grid_res"])
        grid_size = float(hm["grid_size"])
        hand_pos = (
            hand_world_for_gate
            if hand_world_for_gate is not None
            else compute_body_positions(model, mj_data, qpos, hand_links)
        )
        hm_nodes = detect_hand_terrain_nodes(
            hand_pos,
            hand_links,
            hmap,
            grid_res,
            grid_size,
            args.hand_z_tolerance,
            min_timestamp=hand_motion_start,
        )
        nodes.extend(hm_nodes)

    min_dur_fr = max(0, int(args.min_contact_duration * fps))
    lead_fr = max(0, int(args.contact_lead_time * fps))
    mask = merge_contact_nodes_to_mask(
        nodes,
        num_frames=t,
        max_time_gap=args.merge_time_gap,
        max_spatial_dist=args.merge_spatial_dist,
        gap_fill_frames=args.merge_gap_fill,
        min_contact_duration_frames=min_dur_fr,
        contact_lead_frames=lead_fr,
    )

    out_dict = {"contact_mask": mask.astype(np.float32)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out_dict, allow_pickle=True)
    print(
        f"Saved {args.output}  contact_mask shape {mask.shape}  "
        f"fps={fps:g}  foot_frames={(mask[:, :2].any(axis=1)).sum()}  "
        f"hand_frames={(mask[:, 2:].any(axis=1)).sum()}"
    )

    if args.visualize:
        hand_world = _try_compute_hand_world(model, mj_data, qpos)
        if terrain_urdf_for_viz is not None:
            t_off = args.terrain_urdf_offset
            terrain_urdf_offset = (0.0, 0.0, 0.0) if t_off is None else (float(t_off[0]), float(t_off[1]), float(t_off[2]))
            v_model, v_data = _load_scene_with_terrain(
                args.robot_xml,
                terrain_urdf_for_viz,
                use_columns_for_collision=not args.terrain_no_columns,
                terrain_column_res=args.terrain_column_res,
                terrain_floor_threshold=args.terrain_floor_threshold,
                terrain_urdf_offset=terrain_urdf_offset,
            )
            if v_model.nq != model.nq:
                raise ValueError(
                    f"Terrain merge changed nq ({model.nq} -> {v_model.nq}); use a static terrain URDF only"
                )
            vm = v_model
            vd = v_data
            print(f"[viz] Terrain merged from {terrain_urdf_for_viz}", flush=True)
        else:
            vm, vd = model, mj_data
        visualize_motion_contacts(
            vm,
            vd,
            qpos,
            foot_world,
            hand_world,
            mask,
            fps,
            playback_speed=max(args.vis_speed, 0.05),
        )
    elif terrain_urdf_for_viz is not None:
        print(
            "[terrain] Pass --visualize to open the viewer with robot + terrain + contact markers.",
            flush=True,
        )


if __name__ == "__main__":
    main()
