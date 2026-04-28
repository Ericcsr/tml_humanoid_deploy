import os
import tempfile
import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import scipy
import pickle
import redis
from utils.math_utils import *

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from utils.robot_utils import Rate
from utils.redis_utils import REDIS_IP, REDIS_PORT

G1_LINKS = [
    'pelvis', 
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_track_site',    # idx: 5
    'left_ankle_pitch_link', 'left_ankle_roll_link',
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_track_site',   # idx: 12
    'right_ankle_pitch_link', 'right_ankle_roll_link',
    'waist_yaw_link', 'waist_roll_link',
    'torso_link',
    'head_track_site',
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
    'left_elbow_link', 
    'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
    'left_hand_track_site',
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 
    'right_hand_track_site'
]

ZERO_TORQUE = -np.ones(29, dtype=np.float32) * 200.0
DAMPING = np.ones(29, dtype=np.float32) * 200.0

shm_buffer = []

# class ElasticBand:

#     def __init__(self):
#         self.stiffness = 200
#         self.damping = 100
#         self.point = np.array([0, 0, 3])
#         self.length = 0
#         self.enable = True

#     def Advance(self, x, dx):
#         """
#         Args:
#           δx: desired position - current position
#           dx: current velocity
#         """
#         δx = self.point - x
#         distance = np.linalg.norm(δx)
#         direction = δx / distance
#         v = np.dot(dx, direction)
#         f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
#         return f

#     def MujuocoKeyCallback(self, key):
#         glfw = mujoco.glfw.glfw
#         if key == glfw.KEY_7:
#             self.length += 0.1
#         if key == glfw.KEY_8:
#             self.length -= 0.1
#         if key == glfw.KEY_9:
#             self.enable = not self.enable

class ElasticBand:
    """
    ref: https://github.com/unitreerobotics/unitree_mujoco
    Tethers robot to a point and orientation. When init_pos/init_quat are provided
    (e.g. from first frame with terrain), uses those as target instead of world origin.
    """

    def __init__(self, init_pos=None, init_quat_xyzw=None):
        self.kp_pos = 10000
        self.kd_pos = 1000
        self.kp_ang = 1000
        self.kd_ang = 10
        # Tether point: above init position (or origin when not specified)
        if init_pos is not None:
            init_pos[2] = 0
            self.point = np.array(init_pos, dtype=np.float64) + np.array([0, 0, 1])
        else:
            self.point = np.array([0, 0, 1])
        self.length = 0
        self.enable = True
        # Target orientation for PD (None = identity / forward-facing)
        self.target_rot = (
            scipy.spatial.transform.Rotation.from_quat(init_quat_xyzw)
            if init_quat_xyzw is not None
            else None
        )

    def Advance(self, pose):
        """
        Args:
          pose: 13D array containing:
               - pose[0:3]: position in world frame
               - pose[3:7]: quaternion [w,x,y,z] in world frame
               - pose[7:10]: linear velocity in world frame
               - pose[10:13]: angular velocity in world frame
        Returns:
          np.ndarray: 6D vector [fx, fy, fz, tx, ty, tz]
        """
        pos = pose[0:3]
        quat = pose[3:7]
        lin_vel = pose[7:10]
        ang_vel = pose[10:13]

        δx = self.point - pos
        f = self.kp_pos * (δx + np.array([0, 0, self.length])) + self.kd_pos * (0 - lin_vel)

        # --- Orientation PD: error from current to target ---
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot = scipy.spatial.transform.Rotation.from_quat(quat_xyzw)
        if self.target_rot is not None:
            # Error: rotation from target to current (we want to reduce this)
            err_rot = rot * self.target_rot.inv()
            rotvec = err_rot.as_rotvec()
        else:
            rotvec = rot.as_rotvec()  # target = identity
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel

        return np.concatenate([f, torque])

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable

    def disable_band(self):
        self.enable = False

    def handle_keyboard_button(self, key):
        if key == "9":
            self.enable = not self.enable
            print(f"ElasticBand enable: {self.enable}")

def shared_np(size, name, dtype=np.float32):
    try:
        shm = SharedMemory(create=True, size=np.prod(size) * np.dtype(dtype).itemsize, name=name)
        arr = np.ndarray(size, dtype=dtype, buffer=shm.buf)
        arr[:] = 0.0
        shm_buffer.append(shm) # prevent crash
    except FileExistsError:
        print("Shared memory already exists")
        shm = SharedMemory(create=False, name=name)
        arr = np.ndarray(size, dtype=dtype, buffer=shm.buf)
        arr[:] = 0.0
        shm_buffer.append(shm) # prevent crash
    return arr

def pd_control(target_q, data, kp, kd):
    return (target_q - data.qpos[7:36]) * kp - data.qvel[6:35] * kd

def damping_control(data, kd):
    return -data.qvel[6:35] * kd

def zero_torque_control():
    return np.zeros(29, dtype=np.float32)


# Matches rl_policy ref_vr_3point_offsets for left/right wrist (local frame → world); torso row unused here.
_DEFAULT_VR_3POINT_HAND_OFFSETS = np.array(
    [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0]], dtype=np.float64
)


def _body_anchor_world_pos(data, body_id, offset_local):
    """Body origin plus offset expressed in body frame (same convention as vr_3point in rl_policy)."""
    q_wxyz = data.xquat[body_id]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    return data.xpos[body_id] + Rotation.from_quat(q_xyzw).apply(offset_local)


def _object_hand_center_world_pos(data, left_wrist_body_id, right_wrist_body_id, hand_offsets):
    """Midpoint between left/right hand anchor positions (wrist + 3pt offset in each wrist frame)."""
    left_p = _body_anchor_world_pos(data, left_wrist_body_id, hand_offsets[0])
    right_p = _body_anchor_world_pos(data, right_wrist_body_id, hand_offsets[1])
    return (left_p + right_p) * 0.5


def run_simulation(control_lock, data_lock, xml_path, config, ticker_value=None):
        try:
            with open(xml_path, "r") as f:
                xml = f.read()
            model = mujoco.MjModel.from_xml_string(xml)
        except Exception as e:
            print(f"[run_simulation] Failed to load MuJoCo model: {e}", flush=True)
            raise
        data = mujoco.MjData(model)

        # When terrain + sim: init robot at first frame xy, heading, and joints
        sim_init_pos = config.get("sim_init_root_pos")
        sim_init_orn = config.get("sim_init_root_orn")
        sim_init_joints = config.get("sim_init_joint_pos")
        if sim_init_pos is not None and sim_init_orn is not None:
            data.qpos[:3] = np.array(sim_init_pos, dtype=np.float64)
            # MuJoCo quat: wxyz; ref uses xyzw
            q_xyzw = np.array(sim_init_orn, dtype=np.float64)
            data.qpos[3:7] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
            if sim_init_joints is not None:
                data.qpos[7:36] = np.array(sim_init_joints, dtype=np.float64)
            mujoco.mj_forward(model, data)

        # Object: init at first frame, kinematic control until wrist contact (then physics takes over)
        object_trans = None
        object_quat_wxyz = None
        object_qposadr = None
        object_dofadr = None
        object_body_id = None
        object_kinematic_released = False
        wrist_contacted = set()  # body ids of wrists that have contacted object
        left_wrist_body_id = None
        right_wrist_body_id = None
        object_position_from_hand_center = bool(config.get("object_position_from_hand_center", False))
        object_motion_init_only = bool(config.get("object_motion_init_only", False))
        if object_position_from_hand_center:
            vho = config.get("vr_3point_hand_offsets")
            if vho is not None:
                vr_3point_hand_offsets = np.asarray(vho, dtype=np.float64).reshape(2, 3)
            else:
                vr_3point_hand_offsets = _DEFAULT_VR_3POINT_HAND_OFFSETS.copy()
        else:
            vr_3point_hand_offsets = None
        object_motion = config.get("object_motion")
        if object_motion:
            object_motion = os.path.abspath(object_motion) if os.path.isabs(object_motion) else os.path.normpath(os.path.join(os.getcwd(), object_motion))
            try:
                object_body_id = model.body("floating_object").id
                jnt_id = model.body_jntadr[object_body_id]
                object_qposadr = int(model.jnt_qposadr[jnt_id])
                object_dofadr = int(model.jnt_dofadr[jnt_id])
                left_wrist_body_id = model.body("left_wrist_yaw_link").id
                right_wrist_body_id = model.body("right_wrist_yaw_link").id
                obj_motion_data = np.load(object_motion)
                object_trans = obj_motion_data["object_trans"]
                object_quat_wxyz = obj_motion_data["object_quat_wxyz"]
                # Initialize object at first frame pose (position: trajectory or midpoint of wrists)
                data.qpos[object_qposadr + 3:object_qposadr + 7] = object_quat_wxyz[0]
                if object_position_from_hand_center:
                    data.qpos[object_qposadr:object_qposadr + 3] = _object_hand_center_world_pos(
                        data, left_wrist_body_id, right_wrist_body_id, vr_3point_hand_offsets
                    )
                else:
                    data.qpos[object_qposadr:object_qposadr + 3] = object_trans[0]
                data.qvel[object_dofadr:object_dofadr + 6] = 0.0
                mujoco.mj_forward(model, data)
                pos_mode = (
                    "midpoint of left/right hand anchors (wrist + vr_3point offset) + traj orientation"
                    if object_position_from_hand_center
                    else "trajectory pose"
                )
                if object_motion_init_only:
                    object_kinematic_released = True
                    print(
                        f"[run_simulation] Object loaded: {object_trans.shape[0]} frames, position={pos_mode} "
                        f"(first frame only; physics-only afterward)",
                        flush=True,
                    )
                else:
                    print(
                        f"[run_simulation] Object loaded: {object_trans.shape[0]} frames, position={pos_mode} "
                        f"(kinematic until both wrists contact)",
                        flush=True,
                    )
            except Exception as e:
                print(f"[run_simulation] Object init failed: {e}", flush=True)
                object_trans = None
                object_quat_wxyz = None
                object_qposadr = None
                object_dofadr = None
                object_body_id = None
                left_wrist_body_id = None
                right_wrist_body_id = None

        redis_client = redis.Redis(host=REDIS_IP, port=REDIS_PORT, db=0)

        # Elastic band: use first-frame pos/orn when init at first frame (terrain)
        band_init_pos = sim_init_pos if sim_init_pos is not None else None
        band_init_quat = sim_init_orn if sim_init_orn is not None else None
        elastic_band = ElasticBand(init_pos=band_init_pos, init_quat_xyzw=band_init_quat)
        band_attached_link = model.body("torso_link").id

        viewer = mujoco.viewer.launch_passive(model, data, key_callback=elastic_band.MujuocoKeyCallback)
        
        ### prepare shared data
        control = shared_np(30, "control", np.float32)
        q = shared_np(29, "q", np.float32)
        dq = shared_np(29, "dq", np.float32)
        omega_w = shared_np(3, "omega", np.float32)
        imu_quat = shared_np(4, "imu_quat", np.float32)
        root_pos = shared_np(3, "root_pos", np.float32)
        root_vel = shared_np(3, "root_vel", np.float32)
        torso_pos = shared_np(3, "torso_pos", np.float32)
        torso_orn = shared_np(4, "torso_orn", np.float32)
        ### prepare other data
        kp = np.array(config['joint_stiffness'], dtype=np.float32)
        kd = np.array(config['joint_damping'], dtype=np.float32)
        torque_limit = np.array(config['torque_limit'], dtype=np.float32)

        model.opt.timestep = config.get("simulation_dt", 0.005)
        slow_down = config.get("slow_down", 1.0)
        rate = Rate(1 / (model.opt.timestep * slow_down))
        # Redis SLAM mimic: pose + world-frame velocities (see run_state_estimation_slam_only / robot_model)
        slam_redis_hz = float(config.get("slam_redis_hz", 10.0))
        slam_redis_period = 1.0 / slam_redis_hz if slam_redis_hz > 0 else 0.1
        torso_slam_body_id = model.body("torso_link").id
        ts = time.time()
        ts_acc = time.time()
        while True:
            
            with control_lock:
                if control[0] > 180:
                    print("Damping mode")
                    tau = damping_control(data, kd)
                elif control[0] < -180:
                    print("Zero torque mode")
                    tau = zero_torque_control()
                else:
                    tau = pd_control(control[:29], data, kp, kd)
            data.ctrl[:] = tau.clip(-torque_limit, torque_limit)
            if control[29] > 100:
                elastic_band.disable_band()
            if elastic_band.enable:
                pose = np.concatenate(
                    [
                        data.xpos[band_attached_link],  # link position in world
                        data.xquat[
                            band_attached_link
                        ],  # link quaternion in world [w,x,y,z]
                        np.zeros(6),  # placeholder for velocity
                    ]
                )

                # Get velocity in world frame
                mujoco.mj_objectVelocity(
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    band_attached_link,
                    pose[7:13],
                    0,  # 0 for world frame
                )

                # Reorder velocity from [ang, lin] to [lin, ang]
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()

                data.xfrc_applied[band_attached_link] = elastic_band.Advance(pose)
            else:
                data.xfrc_applied[band_attached_link] = np.zeros(6)

            mujoco.mj_step(model, data)

            # Object: require both wrists to contact before releasing kinematic control (let physics act)
            if object_body_id is not None and left_wrist_body_id is not None and right_wrist_body_id is not None and not object_kinematic_released:
                for i in range(data.ncon):
                    b1 = model.geom_bodyid[data.contact[i].geom1]
                    b2 = model.geom_bodyid[data.contact[i].geom2]
                    if b1 == object_body_id or b2 == object_body_id:
                        other = b2 if b1 == object_body_id else b1
                        if other == left_wrist_body_id:
                            wrist_contacted.add(left_wrist_body_id)
                        elif other == right_wrist_body_id:
                            wrist_contacted.add(right_wrist_body_id)
                if left_wrist_body_id in wrist_contacted and right_wrist_body_id in wrist_contacted:
                    object_kinematic_released = True
                    print("[run_simulation] Both wrists contacted object, releasing kinematic control", flush=True)

            # Object trajectory: kinematic control until wrist contact; then physics takes over
            if (object_trans is not None and object_qposadr is not None and ticker_value is not None
                    and not object_kinematic_released):
                ticker = ticker_value.value
                if ticker >= 0:
                    frame_idx = min(int(ticker), len(object_trans) - 1)
                    if object_position_from_hand_center:
                        data.qpos[object_qposadr:object_qposadr + 3] = _object_hand_center_world_pos(
                            data, left_wrist_body_id, right_wrist_body_id, vr_3point_hand_offsets
                        )
                    else:
                        data.qpos[object_qposadr:object_qposadr + 3] = object_trans[frame_idx]
                    data.qpos[object_qposadr + 3:object_qposadr + 7] = object_quat_wxyz[frame_idx]
                    data.qvel[object_dofadr:object_dofadr + 6] = 0.0
                    mujoco.mj_forward(model, data)
            with data_lock:
                q[:] = data.qpos[7:36].copy()
                dq[:] = data.qvel[6:35].copy()
                imu_quat[:] = data.qpos[3:7][[1,2,3,0]].copy()
                omega_w[:] = data.qvel[3:6].copy()
                root_pos[:] = data.qpos[:3].copy()
                root_vel[:] = data.qvel[:3].copy()
                torso_pos[:] = data.xpos[model.body("torso_link").id].copy()
                torso_orn[:] = data.xquat[model.body("torso_link").id][[1,2,3,0]].copy()
            # viewer.render()
            now = time.time()
            if now - ts >= slam_redis_period:
                head_pos = data.xpos[torso_slam_body_id].copy()
                head_quat_xyzw = data.xquat[torso_slam_body_id][[1, 2, 3, 0]].copy()
                vel6 = np.zeros(6, dtype=np.float64)
                mujoco.mj_objectVelocity(
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    torso_slam_body_id,
                    vel6,
                    0,  # world frame: res[0:3] angular vel, res[3:6] linear vel
                )
                head_ang_vel_w = vel6[0:3].astype(np.float32)
                head_lin_vel_w = vel6[3:6].astype(np.float32)
                redis_client.set("head_pos", pickle.dumps(head_pos))
                redis_client.set("head_quat", pickle.dumps(head_quat_xyzw))
                redis_client.set("head_lin_vel", pickle.dumps(head_lin_vel_w))
                redis_client.set("head_ang_vel", pickle.dumps(head_ang_vel_w))
                ts = now
            if now - ts_acc > 0.02:
                root_rot = data.xmat[1].reshape(3, 3) 
                linear_accel_local = root_rot.T @ (data.qacc[0:3] - np.array([0.0, 0.0, 9.81]))
                angular_accel_local = root_rot.T @ data.qacc[3:6]
                root_a = np.hstack([linear_accel_local, angular_accel_local])
                redis_client.set("ddq", pickle.dumps(data.qacc[6:35]))
                redis_client.set("root_a", pickle.dumps(root_a))
                redis_client.set("tau", pickle.dumps(data.ctrl[:]))
                ts_acc = now
            viewer.sync()                                                                  
            rate.sleep()
            #print("Sim step fps:", 1/(time.time() - ts))

class MujocoRobot:
    def __init__(
            self,
            xml_path,
            config,
            ticker_value=None,
        ):
        self.xml_path = xml_path
        self._terrain_temp_file = None
        xml_path_to_load = xml_path

        terrain_path = None
        terrain_urdf = config.get("terrain_urdf") or ""
        terrain_urdf = str(terrain_urdf).strip() if terrain_urdf else ""

        tb_pos = config.get("terrain_box_pos")
        tb_size = config.get("terrain_box_size")
        terrain_boxes_cfg = config.get("terrain_boxes")
        has_terrain_box_legacy = tb_pos is not None and tb_size is not None
        has_terrain_box_multi = False
        if isinstance(terrain_boxes_cfg, dict):
            has_terrain_box_multi = len(terrain_boxes_cfg) > 0
        elif isinstance(terrain_boxes_cfg, list):
            has_terrain_box_multi = len(terrain_boxes_cfg) > 0
        elif terrain_boxes_cfg is not None:
            raise TypeError("terrain_boxes must be a dict, a list, or omitted (not a single box)")
        if has_terrain_box_legacy and has_terrain_box_multi:
            raise ValueError(
                "Use either terrain_box_pos/terrain_box_size (single box) or terrain_boxes, not both"
            )
        has_terrain_box = has_terrain_box_legacy or has_terrain_box_multi
        fb_pos = config.get("free_box_pos")
        fb_size = config.get("free_box_size")
        has_free_box = fb_pos is not None and fb_size is not None

        scene_dirty = False
        with open(xml_path, "r") as f:
            scene_xml = f.read()

        if has_terrain_box_multi:
            from utils.urdf_to_mujoco import merge_terrain_boxes_into_scene_xml

            default_rgba = (0.55, 0.52, 0.48, 1.0)
            boxes: list = []
            if isinstance(terrain_boxes_cfg, dict):
                for name, spec in terrain_boxes_cfg.items():
                    if not isinstance(spec, dict):
                        raise TypeError(f"terrain_boxes[{name!r}] must be a dict with pos, size, ...")
                    pos_ = spec.get("pos")
                    size_ = spec.get("size")
                    if pos_ is None or size_ is None:
                        raise ValueError(
                            f"terrain_boxes[{name!r}] must include pos and size (length-3 each)"
                        )
                    pos_t = tuple(float(x) for x in pos_)
                    size_t = tuple(float(x) for x in size_)
                    if len(pos_t) != 3 or len(size_t) != 3:
                        raise ValueError(
                            f"terrain_boxes[{name!r}]: pos and size must be length-3 [x,y,z] / [lx,ly,lz]"
                        )
                    if min(size_t) <= 0:
                        raise ValueError(
                            f"terrain_boxes[{name!r}]: size entries must be positive (full dimensions in meters)"
                        )
                    rgba_ = spec.get("rgba")
                    if rgba_ is not None:
                        rgba_t = tuple(float(x) for x in rgba_)
                        if len(rgba_t) != 4:
                            raise ValueError(
                                f"terrain_boxes[{name!r}]: rgba must be length-4 [r,g,b,a]"
                            )
                    else:
                        rgba_t = None
                    boxes.append((str(name), pos_t, size_t, rgba_t))
            else:
                for i, spec in enumerate(terrain_boxes_cfg):
                    if not isinstance(spec, dict):
                        raise TypeError(f"terrain_boxes[{i}] must be a dict with pos, size, ...")
                    pos_ = spec.get("pos")
                    size_ = spec.get("size")
                    if pos_ is None or size_ is None:
                        raise ValueError(
                            f"terrain_boxes[{i}] must include pos and size (length-3 each)"
                        )
                    pos_t = tuple(float(x) for x in pos_)
                    size_t = tuple(float(x) for x in size_)
                    if len(pos_t) != 3 or len(size_t) != 3:
                        raise ValueError(
                            f"terrain_boxes[{i}]: pos and size must be length-3 [x,y,z] / [lx,ly,lz]"
                        )
                    if min(size_t) <= 0:
                        raise ValueError(
                            f"terrain_boxes[{i}]: size entries must be positive (full dimensions in meters)"
                        )
                    rgba_ = spec.get("rgba")
                    if rgba_ is not None:
                        rgba_t = tuple(float(x) for x in rgba_)
                        if len(rgba_t) != 4:
                            raise ValueError(
                                f"terrain_boxes[{i}]: rgba must be length-4 [r,g,b,a]"
                            )
                    else:
                        rgba_t = None
                    name = spec.get("name", f"box_{i}")
                    boxes.append((str(name), pos_t, size_t, rgba_t))

            scene_xml = merge_terrain_boxes_into_scene_xml(scene_xml, boxes, default_rgba=default_rgba)
            scene_dirty = True
            print(
                f"[MujocoRobot] Procedural terrain boxes: {len(boxes)} (terrain_boxes)",
                flush=True,
            )

        elif has_terrain_box_legacy:
            from utils.urdf_to_mujoco import merge_terrain_box_into_scene_xml

            pos_t = tuple(float(x) for x in tb_pos)
            size_t = tuple(float(x) for x in tb_size)
            if len(pos_t) != 3 or len(size_t) != 3:
                raise ValueError(
                    "terrain_box_pos and terrain_box_size must each be length-3 lists [x,y,z] / [lx,ly,lz]"
                )
            if min(size_t) <= 0:
                raise ValueError("terrain_box_size entries must be positive (full dimensions in meters)")
            tb_rgba = config.get("terrain_box_rgba")
            if tb_rgba is not None:
                rgba_t = tuple(float(x) for x in tb_rgba)
                if len(rgba_t) != 4:
                    raise ValueError("terrain_box_rgba must be length-4 [r,g,b,a]")
                scene_xml = merge_terrain_box_into_scene_xml(scene_xml, pos_t, size_t, rgba_t)
            else:
                scene_xml = merge_terrain_box_into_scene_xml(scene_xml, pos_t, size_t)
            scene_dirty = True
            print(
                f"[MujocoRobot] Procedural terrain box center={pos_t} full_size={size_t} (m)",
                flush=True,
            )

        if has_free_box:
            from utils.urdf_to_mujoco import merge_free_box_into_scene_xml

            pos_t = tuple(float(x) for x in fb_pos)
            size_t = tuple(float(x) for x in fb_size)
            if len(pos_t) != 3 or len(size_t) != 3:
                raise ValueError(
                    "free_box_pos and free_box_size must each be length-3 lists [x,y,z] / [lx,ly,lz]"
                )
            if min(size_t) <= 0:
                raise ValueError("free_box_size entries must be positive (full dimensions in meters)")
            free_box_mass = float(config.get("free_box_mass", 1.0))
            if free_box_mass <= 0:
                raise ValueError("free_box_mass must be > 0 (kg)")
            fb_rgba = config.get("free_box_rgba")
            if fb_rgba is not None:
                rgba_t = tuple(float(x) for x in fb_rgba)
                if len(rgba_t) != 4:
                    raise ValueError("free_box_rgba must be length-4 [r,g,b,a]")
                scene_xml = merge_free_box_into_scene_xml(
                    scene_xml, pos_t, size_t, free_box_mass, rgba_t
                )
            else:
                scene_xml = merge_free_box_into_scene_xml(
                    scene_xml, pos_t, size_t, free_box_mass
                )
            scene_dirty = True
            print(
                f"[MujocoRobot] Procedural free box center={pos_t} full_size={size_t} (m), mass={free_box_mass:.3f} kg",
                flush=True,
            )

        if "terrain_urdf" in config and terrain_urdf:
            from utils.urdf_to_mujoco import merge_terrain_into_scene_from_string

            terrain_path = os.path.abspath(terrain_urdf) if os.path.isabs(terrain_urdf) else os.path.normpath(os.path.join(os.getcwd(), terrain_urdf))
            if not os.path.exists(terrain_path):
                raise FileNotFoundError(
                    f"terrain_urdf not found: {terrain_path}\n"
                    f"  (resolved from config terrain_urdf: {terrain_urdf})"
                )
            if config.get("terrain_mesh_collision", False):
                use_columns_for_collision = False
            else:
                use_columns_for_collision = config.get("terrain_use_columns", True)
            tuo = config.get("terrain_urdf_offset")
            if tuo is not None:
                terrain_urdf_offset = tuple(float(x) for x in tuo)
                if len(terrain_urdf_offset) != 3:
                    raise ValueError("terrain_urdf_offset must be length-3 [x, y, z] (m)")
            else:
                terrain_urdf_offset = (0.0, 0.0, 0.0)
            scene_xml = merge_terrain_into_scene_from_string(
                scene_xml,
                terrain_path,
                use_columns_for_collision=use_columns_for_collision,
                terrain_column_res=config.get("terrain_column_res", 0.2),
                terrain_floor_threshold=config.get("terrain_floor_threshold", 0.02),
                terrain_urdf_offset=terrain_urdf_offset,
            )
            scene_dirty = True
            if terrain_urdf_offset != (0.0, 0.0, 0.0):
                print(
                    f"[MujocoRobot] terrain_urdf_offset world (m)={terrain_urdf_offset}",
                    flush=True,
                )

        if scene_dirty:
            fd, self._terrain_temp_file = tempfile.mkstemp(
                suffix=".xml",
                prefix="mujoco_scene_",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(scene_xml)
                xml_path_to_load = self._terrain_temp_file
            except Exception:
                os.close(fd)
                if self._terrain_temp_file and os.path.exists(self._terrain_temp_file):
                    os.unlink(self._terrain_temp_file)
                raise

        # Validate model loads before starting subprocess (errors there are silenced)
        try:
            with open(xml_path_to_load, "r") as f:
                xml_content = f.read()
            mujoco.MjModel.from_xml_string(xml_content)
        except Exception as e:
            msg = (
                f"Failed to load MuJoCo model from {xml_path_to_load}\n"
                f"  Error: {e}\n"
            )
            if terrain_urdf and terrain_path is not None:
                msg += (
                    f"  Terrain was merged from: {terrain_path}\n"
                    f"  Check that the URDF and mesh file exist and are valid."
                )
            if has_terrain_box:
                msg += "  Check terrain_box_pos / terrain_box_size or terrain_boxes and scene XML.\n"
            if has_free_box:
                msg += "  Check free_box_pos / free_box_size / free_box_mass and scene XML.\n"
            raise RuntimeError(msg) from e

        if terrain_urdf and terrain_path is not None:
            if config.get("terrain_mesh_collision", False):
                terr_col = "terrain collision: URDF mesh (no heightmap box columns)"
            elif config.get("terrain_use_columns", True):
                terr_col = (
                    f"terrain collision: heightmap columns "
                    f"(res={config.get('terrain_column_res', 0.2)})"
                )
            else:
                terr_col = "terrain collision: URDF mesh (terrain_use_columns: false)"
            print(f"[MujocoRobot] Terrain loaded from {terrain_path} — {terr_col}", flush=True)

        # Merge object from URDF when configured (object added LAST to preserve robot indices)
        object_urdf = config.get("object_urdf") or ""
        object_urdf = str(object_urdf).strip() if object_urdf else ""
        if "object_urdf" in config and object_urdf:
            from utils.urdf_to_mujoco import merge_object_into_scene
            object_path = os.path.abspath(object_urdf) if os.path.isabs(object_urdf) else os.path.normpath(os.path.join(os.getcwd(), object_urdf))
            if not os.path.exists(object_path):
                raise FileNotFoundError(f"object_urdf not found: {object_path}")
            with open(xml_path_to_load, "r") as f:
                scene_xml = f.read()
            merged_xml = merge_object_into_scene(scene_xml, object_path)
            fd_obj, obj_temp = tempfile.mkstemp(
                suffix=".xml",
                prefix="mujoco_scene_obj_",
            )
            try:
                with os.fdopen(fd_obj, "w") as f:
                    f.write(merged_xml)
                xml_path_to_load = obj_temp
                if self._terrain_temp_file:
                    os.unlink(self._terrain_temp_file)
                self._terrain_temp_file = obj_temp
            except Exception:
                os.close(fd_obj)
                if os.path.exists(obj_temp):
                    os.unlink(obj_temp)
                raise
            print(f"[MujocoRobot] Object loaded from {object_path}", flush=True)

        self.control_var = shared_np(30, "control", np.float32)
        self.q_var = shared_np(29, "q", np.float32)
        self.dq_var = shared_np(29, "dq", np.float32)
        self.omega_var = shared_np(3, "omega", np.float32)
        self.imu_quat_var = shared_np(4, "imu_quat", np.float32)
        self.root_pos = shared_np(3, "root_pos", np.float32)
        self.root_vel = shared_np(3, "root_vel", np.float32)
        self.torso_pos = shared_np(3, "torso_pos", np.float32)
        self.torso_orn = shared_np(4, "torso_orn", np.float32)

        self.control_lock = mp.Lock()
        self.data_lock = mp.Lock()
        self.config = config
        self.control_dt = config.get("control_dt", 0.02) # default 50 Hz
        self.process = mp.Process(target=run_simulation, args=(self.control_lock, self.data_lock, xml_path_to_load, self.config, ticker_value))
        self.process.start()

    def pd_control(self, target_q):
        with self.control_lock:
            self.control_var[:29] = target_q.copy()

    def release_robot(self):
        with self.control_lock:
            self.control_var[29] = 200.0

    def set_robot_state(self, q):
        total_time = 2
        num_step = int(total_time / self.control_dt)
        with self.data_lock:
            init_q = self.q_var.copy()
        
        target_q = np.zeros(29, dtype=np.float32)
        target_q[:] = q
        
        for i in range(num_step):
            alpha = i / num_step
            self.pd_control(init_q*(1-alpha)+target_q*alpha)
            time.sleep(self.control_dt)

    def maintain_state(self, q):
        self.pd_control(q)
        self.init_q = q.copy()
        input("Press Enter to continue...")
        
    def damping_state(self):
        self.pd_control(DAMPING)

    def zero_torque_state(self):
        self.pd_control(ZERO_TORQUE)

    def get_robot_state(self):
        with self.data_lock:
            q = self.q_var.copy()
            dq = self.dq_var.copy()
            imu_quat = self.imu_quat_var.copy()
            omega_w = self.omega_var.copy()
        omega_l = omega_w #Rotation.from_quat(imu_quat).inv().apply(omega_w)
        return q, dq, imu_quat, omega_l
    
    def get_root_state(self):
        with self.data_lock:
            root_pos = self.root_pos.copy()
            root_vel = self.root_vel.copy()
            root_orn = self.imu_quat_var.copy()
        root_vel_l = Rotation.from_quat(root_orn).inv().apply(root_vel)
        return root_pos, root_orn, root_vel_l

    def get_anchor_state(self):
        with self.data_lock:
            torso_pos = self.torso_pos.copy()
            torso_orn = self.torso_orn.copy()
        return torso_pos, torso_orn

    def step_robot(self, action):
        self.pd_control(action)

    def close(self):
        self.process.terminate()
        self.process.join()
        if self._terrain_temp_file and os.path.exists(self._terrain_temp_file):
            try:
                os.unlink(self._terrain_temp_file)
            except OSError:
                pass

    def get_start_ticker(self):
        return True


if __name__ == "__main__":
    import yaml
    with open("sample_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    robot = MujocoRobot("assets/g1/scene_29dof.xml", config["rl_policy"])

    time.sleep(1)  # wait for the simulation to start

    robot.zero_torque_state()

    input()

    robot.set_robot_state(np.zeros(29, dtype=np.float32))
    for i in range(100):
        print(robot.get_robot_state())
        time.sleep(0.01)
    input()

    robot.damping_state()

    input()
    
    robot.close()