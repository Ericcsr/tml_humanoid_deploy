"""
URDF to MuJoCo XML converter for static terrain/objects.
Handles links with mesh geometry (visual and collision).
MuJoCo uses convex hull for mesh collision, which is poor for height-map terrain.
Use independent column boxes for collision instead.
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
from typing import Tuple, List


def _load_obj_vertices(obj_path: str, scale: Tuple[float, float, float] = (1, 1, 1)) -> np.ndarray:
    """Load vertices from OBJ file. Returns (N, 3) array."""
    vertices = []
    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v ") and not line.startswith("vt ") and not line.startswith("vn "):
                parts = line.split()
                if len(parts) >= 4:
                    v = [float(parts[1]) * scale[0], float(parts[2]) * scale[1], float(parts[3]) * scale[2]]
                    vertices.append(v)
    return np.array(vertices) if vertices else np.empty((0, 3))


def _mesh_to_heightmap(vertices: np.ndarray, grid_res: float, floor_threshold: float = 0.02) -> Tuple[np.ndarray, float, float]:
    """
    Sample mesh vertices into a height map grid. Returns (hmap, x_min, y_min).
    Each cell stores max z of vertices in that cell. Cells with no vertices get 0.
    """
    if len(vertices) == 0:
        return np.zeros((1, 1)), 0.0, 0.0
    x_min, y_min = vertices[:, 0].min(), vertices[:, 1].min()
    x_max, y_max = vertices[:, 0].max(), vertices[:, 1].max()
    n_cols = max(1, int(np.ceil((x_max - x_min) / grid_res)))
    n_rows = max(1, int(np.ceil((y_max - y_min) / grid_res)))
    hmap = np.zeros((n_rows, n_cols))
    counts = np.zeros((n_rows, n_cols))
    for v in vertices:
        c = int((v[0] - x_min) / grid_res)
        r = int((v[1] - y_min) / grid_res)
        c = min(c, n_cols - 1)
        r = min(r, n_rows - 1)
        if v[2] > hmap[r, c]:
            hmap[r, c] = v[2]
        counts[r, c] += 1
    # Prune floor: cells below threshold get 0
    hmap[hmap < floor_threshold] = 0.0
    return hmap, x_min, y_min


def _heightmap_to_columns(
    hmap: np.ndarray, x_min: float, y_min: float, grid_res: float,
    min_height: float = 0.02,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Convert height map to column boxes. Each non-zero cell becomes a box.
    Returns list of ((pos_x, pos_y, pos_z), (half_x, half_y, half_z)).
    """
    columns = []
    half_res = grid_res / 2.0
    n_rows, n_cols = hmap.shape
    for r in range(n_rows):
        for c in range(n_cols):
            h = hmap[r, c]
            if h < min_height:
                continue
            cx = x_min + (c + 0.5) * grid_res
            cy = y_min + (r + 0.5) * grid_res
            cz = h / 2.0
            pos = (cx, cy, cz)
            size = (half_res, half_res, h / 2.0)
            columns.append((pos, size))
    return columns


def _parse_xyz(s: str) -> Tuple[float, float, float]:
    """Parse 'x y z' string to tuple."""
    parts = s.split()
    return (float(parts[0]), float(parts[1]), float(parts[2])) if len(parts) >= 3 else (0, 0, 0)


def _rpy_to_quat(rpy: Tuple[float, float, float]) -> str:
    """Convert roll-pitch-yaw (rad) to MuJoCo quat string 'w x y z'."""
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler("xyz", rpy)
    q = r.as_quat()  # [x, y, z, w]
    return f"{q[3]} {q[0]} {q[1]} {q[2]}"


def urdf_to_mujoco_xml(
    urdf_path: str,
    use_columns_for_collision: bool = True,
    terrain_column_res: float = 0.2,
    terrain_floor_threshold: float = 0.02,
) -> Tuple[str, str]:
    """
    Convert a URDF file (e.g. static terrain) to MuJoCo MJCF XML fragment.

    Returns (asset_xml, body_xml) strings to merge into a MuJoCo scene.
    When use_columns_for_collision=True (default), samples the mesh to a height map
    and generates independent box columns for collision (avoids convex-hull issues).
    Keeps mesh for visual. Mesh file paths are absolute.
    """
    urdf_path = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # URDF namespace
    ns = {"urdf": "http://www.ros.org/wiki/urdf"} if root.tag.startswith("{") else {}
    if ns:
        # Strip namespace for simpler queries
        def strip_ns(elem):
            if elem.tag.startswith("{"):
                elem.tag = elem.tag.split("}", 1)[1]
            for child in elem:
                strip_ns(child)
        strip_ns(root)

    mesh_assets = []
    body_elements = []

    for link in root.findall("link"):
        link_name = link.get("name", "link")
        # Default origin
        origin_xyz = (0, 0, 0)
        origin_rpy = (0, 0, 0)
        origin_elem = link.find("visual/origin") or link.find("collision/origin")
        if origin_elem is not None:
            xyz = origin_elem.get("xyz")
            rpy = origin_elem.get("rpy")
            if xyz:
                origin_xyz = _parse_xyz(xyz)
            if rpy:
                origin_rpy = _parse_xyz(rpy)

        # Prefer collision mesh for physics; fallback to visual
        geom_elem = link.find("collision/geometry/mesh") or link.find("visual/geometry/mesh")
        if geom_elem is None:
            continue

        filename = geom_elem.get("filename")
        if not filename:
            continue

        scale_str = geom_elem.get("scale", "1 1 1")
        scale_parts = scale_str.split()
        scale_tuple = (
            (float(scale_parts[0]), float(scale_parts[1]), float(scale_parts[2]))
            if len(scale_parts) >= 3 else (1.0, 1.0, 1.0)
        )
        scale = " ".join(scale_parts[:3]) if len(scale_parts) >= 3 else "1 1 1"

        # Resolve mesh path (use absolute for reliable loading when XML is from string)
        mesh_abs = os.path.normpath(os.path.join(urdf_dir, filename))
        mesh_path = mesh_abs.replace("\\", "/")

        # Material/color from visual
        rgba = "0.5 0.5 0.5 1.0"
        mat_elem = link.find("visual/material/color")
        if mat_elem is not None and mat_elem.get("rgba"):
            rgba = mat_elem.get("rgba")

        mesh_name = f"terrain_mesh_{link_name}".replace(" ", "_")
        mesh_assets.append(
            f'    <mesh name="{mesh_name}" file="{mesh_path}" scale="{scale}"/>'
        )

        pos_str = f"{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}"
        quat_str = _rpy_to_quat(origin_rpy)

        # Build body: visual mesh (no collision) + collision columns (boxes)
        geom_parts = []
        # Visual: mesh geom with contype=0 conaffinity=0 (no collision)
        geom_parts.append(
            f'      <geom name="terrain_visual_{link_name}" type="mesh" mesh="{mesh_name}" '
            f'contype="0" conaffinity="0" group="2" rgba="{rgba}"/>'
        )

        if use_columns_for_collision:
            # Collision: independent columns from height map
            vertices = _load_obj_vertices(mesh_path, scale_tuple)
            hmap, x_min, y_min = _mesh_to_heightmap(
                vertices, terrain_column_res, terrain_floor_threshold
            )
            columns = _heightmap_to_columns(
                hmap, x_min, y_min, terrain_column_res, terrain_floor_threshold
            )
            for i, (pos, size) in enumerate(columns):
                px, py, pz = pos
                sx, sy, sz = size
                geom_parts.append(
                    f'      <geom name="terrain_col_{link_name}_{i}" type="box" '
                    f'pos="{px} {py} {pz}" size="{sx} {sy} {sz}" '
                    f'contype="1" conaffinity="1" rgba="0.4 0.5 0.4 0.5"/>'
                )
        else:
            # Original: mesh for collision (convex hull - poor for height maps)
            geom_parts.append(
                f'      <geom name="terrain_geom_{link_name}" type="mesh" mesh="{mesh_name}" '
                f'contype="1" conaffinity="1" rgba="{rgba}"/>'
            )

        body_elements.append(
            f'    <body name="terrain_{link_name}" pos="{pos_str}" quat="{quat_str}">\n'
            + "\n".join(geom_parts) + "\n"
            f'    </body>'
        )

    if not mesh_assets or not body_elements:
        return "", ""

    mesh_xml = "\n".join(mesh_assets)
    body_xml = "\n".join(body_elements)
    return mesh_xml, body_xml


def merge_terrain_into_scene(
    scene_xml_path: str,
    terrain_urdf_path: str,
    output_path: str = None,
    use_columns_for_collision: bool = True,
    terrain_column_res: float = 0.2,
    terrain_floor_threshold: float = 0.02,
) -> str:
    """
    Merge terrain from URDF into a MuJoCo scene XML.
    Inserts terrain mesh assets and bodies into the scene.
    Returns the merged XML string. If output_path is given, also writes to file.
    """
    with open(scene_xml_path, "r") as f:
        scene_xml = f.read()

    mesh_xml, body_xml = urdf_to_mujoco_xml(
        terrain_urdf_path,
        use_columns_for_collision=use_columns_for_collision,
        terrain_column_res=terrain_column_res,
        terrain_floor_threshold=terrain_floor_threshold,
    )
    if not mesh_xml or not body_xml:
        return scene_xml

    # Insert mesh assets before </asset>
    if "<asset>" in scene_xml and "</asset>" in scene_xml:
        scene_xml = scene_xml.replace("</asset>", "\n" + mesh_xml + "\n  </asset>")
    else:
        insert = f"  <asset>\n{mesh_xml}\n  </asset>\n  "
        scene_xml = scene_xml.replace("<worldbody>", insert + "<worldbody>")

    # Insert terrain body after <worldbody>
    worldbody_start = scene_xml.find("<worldbody>")
    if worldbody_start >= 0:
        insert_pos = scene_xml.find(">", worldbody_start) + 1
        while insert_pos < len(scene_xml) and scene_xml[insert_pos] in " \t\n":
            insert_pos += 1
        scene_xml = scene_xml[:insert_pos] + "\n" + body_xml + "\n    " + scene_xml[insert_pos:]

    if output_path:
        with open(output_path, "w") as f:
            f.write(scene_xml)

    return scene_xml
