"""
URDF to MuJoCo XML converter for static terrain/objects.
Handles links with mesh geometry (visual and collision).

Terrain: with use_columns_for_collision=True (default in merge_terrain_into_scene), the mesh is
sampled to a heightmap and approximated with box columns. With use_columns_for_collision=False, a
single mesh collision geom is used (controller config: terrain_mesh_collision: true in mujoco_env).

Note: MuJoCo mesh collision still uses a convex hull per mesh geom (not the column decomposition).

Procedural box terrain (no URDF): mujoco_env reads config keys terrain_box_pos, terrain_box_size
(full dimensions); see merge_terrain_box_into_scene_xml.
Dynamic free box (no URDF): mujoco_env reads config keys free_box_pos, free_box_size, free_box_mass;
see merge_free_box_into_scene_xml.
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


def urdf_to_mujoco_object(urdf_path: str) -> Tuple[str, str]:
    """
    Convert an object URDF (multi-link with fixed joints) to a single MuJoCo free body.
    Flattens all links into one body with freejoint. Handles box, cylinder, sphere, mesh.
    Returns body_xml string to insert into worldbody.
    """
    urdf_path = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root_elem = tree.getroot()

    if root_elem.tag.startswith("{"):
        def strip_ns(elem):
            if elem.tag.startswith("{"):
                elem.tag = elem.tag.split("}", 1)[1]
            for child in elem:
                strip_ns(child)
        strip_ns(root_elem)

    # Build parent->child map from joints
    parent_map = {}
    joint_origins = {}
    for joint in root_elem.findall("joint"):
        jtype = joint.get("type", "fixed")
        if jtype != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        parent_map[child_name] = parent_name
        origin = joint.find("origin")
        xyz, rpy = (0, 0, 0), (0, 0, 0)
        if origin is not None:
            if origin.get("xyz"):
                xyz = _parse_xyz(origin.get("xyz"))
            if origin.get("rpy"):
                rpy = _parse_xyz(origin.get("rpy"))
        joint_origins[(parent_name, child_name)] = (xyz, rpy)

    # Find root: link that is never a child
    all_links = {l.get("name") for l in root_elem.findall("link")}
    children = set(parent_map.keys())
    root_candidates = all_links - children
    root_link = next(iter(root_candidates), None) if root_candidates else next(iter(all_links))

    def link_to_root_transform(link_name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute (pos, rpy) of link in root frame by traversing up."""
        if link_name == root_link:
            return (0, 0, 0), (0, 0, 0)
        parent = parent_map.get(link_name)
        if parent is None:
            return (0, 0, 0), (0, 0, 0)
        xyz, rpy = joint_origins.get((parent, link_name), ((0, 0, 0), (0, 0, 0)))
        p_pos, p_rpy = link_to_root_transform(parent)
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler("xyz", rpy)
        pos_in_parent = np.array(xyz)
        pos_in_root = np.array(p_pos) + Rotation.from_euler("xyz", p_rpy).apply(pos_in_parent)
        R_child = R
        R_parent = Rotation.from_euler("xyz", p_rpy)
        R_total = R_parent * R_child
        rpy_total = R_total.as_euler("xyz")
        return tuple(pos_in_root.tolist()), tuple(rpy_total.tolist())

    geom_parts = []
    mesh_assets = []
    for link in root_elem.findall("link"):
        link_name = link.get("name", "link")
        pos, rpy = link_to_root_transform(link_name)
        quat_str = _rpy_to_quat(rpy)
        pos_str = f"{pos[0]} {pos[1]} {pos[2]}"

        geom_elems = link.findall("collision/geometry")
        if not geom_elems:
            geom_elems = link.findall("visual/geometry")
        for geom in geom_elems:
            geo = geom.find("box")
            if geo is None:
                geo = geom.find("cylinder")
            if geo is None:
                geo = geom.find("sphere")
            if geo is None:
                geo = geom.find("mesh")
            if geo is None:
                continue
            rgba = "0.5 0.5 0.5 1.0"
            mat = link.find("visual/material/color") or link.find("collision/material/color")
            if mat is not None and mat.get("rgba"):
                rgba = mat.get("rgba")

            if geo.tag == "box":
                size_str = geo.get("size", "0.1 0.1 0.1")
                parts = size_str.split()
                half = [float(p) / 2 for p in parts[:3]] if len(parts) >= 3 else [0.05, 0.05, 0.05]
                geom_parts.append(
                    f'      <geom name="object_{link_name}_{geo.tag}" type="box" pos="{pos_str}" quat="{quat_str}" '
                    f'size="{" ".join(map(str, half))}" contype="1" conaffinity="1" rgba="{rgba}"/>'
                )
            elif geo.tag == "cylinder":
                r = float(geo.get("radius", 0.05))
                l = float(geo.get("length", 0.1))
                geom_parts.append(
                    f'      <geom name="object_{link_name}_{geo.tag}" type="cylinder" pos="{pos_str}" quat="{quat_str}" '
                    f'size="{r} {l/2}" contype="1" conaffinity="1" rgba="{rgba}"/>'
                )
            elif geo.tag == "sphere":
                r = float(geo.get("radius", 0.05))
                geom_parts.append(
                    f'      <geom name="object_{link_name}_{geo.tag}" type="sphere" pos="{pos_str}" quat="{quat_str}" '
                    f'size="{r}" contype="1" conaffinity="1" rgba="{rgba}"/>'
                )
            elif geo.tag == "mesh":
                filename = geo.get("filename")
                if not filename:
                    continue
                scale_str = geo.get("scale", "1 1 1")
                scale_parts = scale_str.split()
                scale = " ".join(scale_parts[:3]) if len(scale_parts) >= 3 else "1 1 1"
                mesh_path = os.path.normpath(os.path.join(urdf_dir, filename)).replace("\\", "/")
                mesh_name = f"object_mesh_{link_name}".replace(" ", "_")
                mesh_assets.append(f'    <mesh name="{mesh_name}" file="{mesh_path}" scale="{scale}"/>')
                geom_parts.append(
                    f'      <geom name="object_{link_name}_mesh" type="mesh" pos="{pos_str}" quat="{quat_str}" '
                    f'mesh="{mesh_name}" contype="1" conaffinity="1" rgba="{rgba}"/>'
                )

    if not geom_parts:
        return "", ""

    body_xml = (
        '    <body name="floating_object" pos="0 0 0" quat="1 0 0 0">\n'
        '      <freejoint name="object_floating_joint"/>\n'
        '      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>\n'
        + "\n".join(geom_parts) + "\n"
        "    </body>"
    )
    asset_xml = "\n".join(mesh_assets) if mesh_assets else ""
    return body_xml, asset_xml


def merge_object_into_scene(scene_xml: str, object_urdf_path: str) -> str:
    """
    Merge a floating object from URDF into the scene (at end of worldbody).
    Object is added LAST so robot qpos/qvel indices (0-35, 0-34) stay unchanged.
    """
    body_xml, asset_xml = urdf_to_mujoco_object(object_urdf_path)
    if not body_xml:
        return scene_xml

    if asset_xml and "<asset>" in scene_xml and "</asset>" in scene_xml:
        scene_xml = scene_xml.replace("</asset>", "\n" + asset_xml + "\n  </asset>")

    # Insert before </worldbody> so object is last
    worldbody_end = scene_xml.rfind("</worldbody>")
    if worldbody_end >= 0:
        scene_xml = scene_xml[:worldbody_end] + "\n" + body_xml + "\n    " + scene_xml[worldbody_end:]
    return scene_xml


def _insert_after_worldbody_open(scene_xml: str, body_fragment: str) -> str:
    """Insert a worldbody child right after <worldbody> ... newline (same as terrain insert)."""
    worldbody_start = scene_xml.find("<worldbody>")
    if worldbody_start < 0:
        return scene_xml
    insert_pos = scene_xml.find(">", worldbody_start) + 1
    while insert_pos < len(scene_xml) and scene_xml[insert_pos] in " \t\n":
        insert_pos += 1
    return scene_xml[:insert_pos] + "\n" + body_fragment + "\n    " + scene_xml[insert_pos:]


def merge_terrain_box_into_scene_xml(
    scene_xml: str,
    pos_xyz: Tuple[float, float, float],
    size_xyz: Tuple[float, float, float],
    rgba: Tuple[float, float, float, float] = (0.55, 0.52, 0.48, 1.0),
) -> str:
    """
    Insert a static axis-aligned box (world body) — simple procedural terrain / platform.

    Args:
        pos_xyz: World position of the box center (m).
        size_xyz: Full outer dimensions (lx, ly, lz) in meters; converted to MuJoCo half-sizes.
        rgba: Visual/collision rgba (alpha only affects visualization).

    The body is named ``terrain_box``; geom ``terrain_box_geom``.
    """
    hx = float(size_xyz[0]) / 2.0
    hy = float(size_xyz[1]) / 2.0
    hz = float(size_xyz[2]) / 2.0
    px, py, pz = float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])
    r, g, b, a = float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])
    body_xml = (
        f'    <body name="terrain_box" pos="{px} {py} {pz}" quat="1 0 0 0">\n'
        f'      <geom name="terrain_box_geom" type="box" pos="0 0 0" size="{hx} {hy} {hz}" '
        f'contype="1" conaffinity="1" rgba="{r} {g} {b} {a}"/>\n'
        f"    </body>"
    )
    return _insert_after_worldbody_open(scene_xml, body_xml)


def merge_free_box_into_scene_xml(
    scene_xml: str,
    pos_xyz: Tuple[float, float, float],
    size_xyz: Tuple[float, float, float],
    mass: float,
    rgba: Tuple[float, float, float, float] = (0.65, 0.45, 0.35, 1.0),
) -> str:
    """
    Insert a dynamic axis-aligned box with a free joint.

    Args:
        pos_xyz: World position of the box center (m).
        size_xyz: Full outer dimensions (lx, ly, lz) in meters; converted to MuJoCo half-sizes.
        mass: Box mass in kg.
        rgba: Visual/collision rgba (alpha only affects visualization).

    The body is named ``free_box``; freejoint ``free_box_joint``; geom ``free_box_geom``.
    """
    hx = float(size_xyz[0]) / 2.0
    hy = float(size_xyz[1]) / 2.0
    hz = float(size_xyz[2]) / 2.0
    px, py, pz = float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])
    m = float(mass)
    r, g, b, a = float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])
    body_xml = (
        f'    <body name="free_box" pos="{px} {py} {pz}" quat="1 0 0 0">\n'
        f'      <freejoint name="free_box_joint"/>\n'
        f'      <geom name="free_box_geom" type="box" pos="0 0 0" size="{hx} {hy} {hz}" '
        f'mass="{m}" contype="1" conaffinity="1" rgba="{r} {g} {b} {a}"/>\n'
        f"    </body>"
    )
    return _insert_after_worldbody_open(scene_xml, body_xml)


def merge_terrain_into_scene_from_string(
    scene_xml: str,
    terrain_urdf_path: str,
    use_columns_for_collision: bool = True,
    terrain_column_res: float = 0.2,
    terrain_floor_threshold: float = 0.02,
) -> str:
    """
    Merge terrain URDF into an in-memory MJCF string (same as merge_terrain_into_scene, no file read).
    """
    mesh_xml, body_xml = urdf_to_mujoco_xml(
        terrain_urdf_path,
        use_columns_for_collision=use_columns_for_collision,
        terrain_column_res=terrain_column_res,
        terrain_floor_threshold=terrain_floor_threshold,
    )
    if not mesh_xml or not body_xml:
        return scene_xml

    if "<asset>" in scene_xml and "</asset>" in scene_xml:
        scene_xml = scene_xml.replace("</asset>", "\n" + mesh_xml + "\n  </asset>")
    else:
        insert = f"  <asset>\n{mesh_xml}\n  </asset>\n  "
        scene_xml = scene_xml.replace("<worldbody>", insert + "<worldbody>")

    worldbody_start = scene_xml.find("<worldbody>")
    if worldbody_start >= 0:
        insert_pos = scene_xml.find(">", worldbody_start) + 1
        while insert_pos < len(scene_xml) and scene_xml[insert_pos] in " \t\n":
            insert_pos += 1
        scene_xml = scene_xml[:insert_pos] + "\n" + body_xml + "\n    " + scene_xml[insert_pos:]

    return scene_xml


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
    scene_xml = merge_terrain_into_scene_from_string(
        scene_xml,
        terrain_urdf_path,
        use_columns_for_collision=use_columns_for_collision,
        terrain_column_res=terrain_column_res,
        terrain_floor_threshold=terrain_floor_threshold,
    )
    if output_path:
        with open(output_path, "w") as f:
            f.write(scene_xml)
    return scene_xml
