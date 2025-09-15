from typing import Optional

import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation

# ---- Optional Numba import & shim ----
try:
    from numba import njit, types

    JIT_AVAILABLE = True
except Exception:
    JIT_AVAILABLE = False
    types = None

    def njit(*args, **kwargs):
        # no-op decorator when numba isn't available
        def wrap(f):
            return f

        return wrap


# ------------ JIT kernels ------------
@njit(cache=True)
def _predict_inplace(x, P, dt, q_acc):
    """In-place CV predict using block formulas (3x3 position/velocity blocks)."""
    if dt <= 0.0:
        return

    # x_pos += dt * v
    x[0] += dt * x[3]
    x[1] += dt * x[4]
    x[2] += dt * x[5]

    # Blocks of P
    # P = [[Ppp, Ppv],
    #      [Pvp, Pvv]]
    # Update:
    # Ppp' = Ppp + dt*(Ppv+Pvp) + dt^2*Pvv + Q11
    # Ppv' = Ppv + dt*Pvv       + Q12
    # Pvp' = Ppv'^T
    # Pvv' = Pvv                + Q22
    dt2 = dt * dt
    Q11 = (dt2 * dt / 3.0) * q_acc
    Q12 = (dt2 / 2.0) * q_acc
    Q22 = (dt) * q_acc

    # We'll do it via slices and copies to avoid aliasing
    Ppp = P[0:3, 0:3].copy()
    Ppv = P[0:3, 3:6].copy()
    Pvp = P[3:6, 0:3].copy()
    Pvv = P[3:6, 3:6].copy()

    # Ppp'
    Ppp_new = Ppp + dt * (Ppv + Pvp) + (dt2) * Pvv
    for i in range(3):
        Ppp_new[i, i] += Q11

    # Ppv'
    Ppv_new = Ppv + dt * Pvv
    for i in range(3):
        Ppv_new[i, i] += Q12

    # Pvp' = Ppv'^T
    Pvp_new = Ppv_new.T

    # Pvv'
    Pvv_new = Pvv.copy()
    for i in range(3):
        Pvv_new[i, i] += Q22

    # Write back
    P[0:3, 0:3] = Ppp_new
    P[0:3, 3:6] = Ppv_new
    P[3:6, 0:3] = Pvp_new
    P[3:6, 3:6] = Pvv_new

    # Symmetrize
    for i in range(6):
        for j in range(i + 1, 6):
            v = 0.5 * (P[i, j] + P[j, i])
            P[i, j] = v
            P[j, i] = v


@njit(cache=True)
def _update_block_inplace(x, P, z, R, start_idx):
    """
    In-place Kalman update for either position (start_idx=0) or velocity (start_idx=3).
    Joseph-form covariance with block structure to avoid building H explicitly.
    """
    a = start_idx
    b = a + 3

    # Innovation y = z - h(x)
    y = np.empty(3, dtype=np.float64)
    for i in range(3):
        y[i] = z[i] - x[a + i]

    # S = P_aa + R
    Paa = P[a:b, a:b].copy()
    S = Paa + R

    # PH^T = P[:, a:b]
    PHt = P[:, a:b].copy()  # shape (6,3)

    # K = PHt @ inv(S), computed via solve on S^T:  K^T = solve(S^T, PHt^T)
    # This is faster/more stable than explicit inverse.
    Kt = np.linalg.solve(S.T, PHt.T)  # (3,6)
    K = Kt.T  # (6,3)

    # x = x + K y
    # temp = K @ y
    temp = np.empty(6, dtype=np.float64)
    for i in range(6):
        s = 0.0
        for j in range(3):
            s += K[i, j] * y[j]
        temp[i] = s
        x[i] += s

    # Joseph-form P update using block tricks:
    # A = (I - K H); but we avoid forming H:
    # AP = P - K @ (H P) = P - K @ P[a:b, :]
    HP = P[a:b, :].copy()  # (3,6)
    KP = np.empty((6, 6), dtype=np.float64)  # = K @ HP
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(3):
                s += K[i, k] * HP[k, j]
            KP[i, j] = s

    AP = P - KP

    # P' = AP @ (I - H^T K^T) + K R K^T
    # (I - H^T K^T) affects only columns a:b:
    # AP @ (I - H^T K^T) = AP - AP[:, a:b] @ K^T
    APab = AP[:, a:b].copy()  # (6,3)
    AP_corr = np.empty((6, 6), dtype=np.float64)
    # AP_corr = AP - APab @ K^T
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(3):
                s += APab[i, k] * K[j, k]  # (APab) @ (K^T)
            AP_corr[i, j] = AP[i, j] - s

    # Add K R K^T
    KR = np.empty((6, 3), dtype=np.float64)
    for i in range(6):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += K[i, k] * R[k, j]
            KR[i, j] = s

    P_new = np.empty((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(3):
                s += KR[i, k] * K[j, k]
            P_new[i, j] = AP_corr[i, j] + s

    # Symmetrize and write back
    for i in range(6):
        for j in range(i + 1, 6):
            v = 0.5 * (P_new[i, j] + P_new[j, i])
            P_new[i, j] = v
            P_new[j, i] = v
    for i in range(6):
        for j in range(6):
            P[i, j] = P_new[i, j]

    # (Optional) small numerical floor on diagonals
    for i in range(6):
        if P[i, i] < 1e-12:
            P[i, i] = 1e-12


def rotatepoint(q, v):
    # q_v = [v[0], v[1], v[2], 0]
    # return quatmultiply(quatmultiply(q, q_v), quatconj(q))[:-1]
    #
    # https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
    q_r = q[3:4]
    q_xyz = q[:3]
    t = 2 * np.cross(q_xyz, v)
    return v + q_r * t + np.cross(q_xyz, t)


def heading_zup(quat):
    ref_dir = np.zeros_like(quat[:3])
    ref_dir[0] = 1
    ref_dir = rotatepoint(quat, ref_dir)
    return np.arctan2(ref_dir[1], ref_dir[0])


# ------------- High-level filter -------------
class RootPoseFilterSimple:
    def __init__(self, alpha: float = 0.5, dt: float = 0.02, use_float32: bool = False):
        self.alpha = np.float32(alpha) if use_float32 else np.float64(alpha)
        self.last_pos = np.zeros(3, dtype=np.float64)
        self.last_vel = None
        self.last_slam_pos = None
        self.slam_updated = True
        self.dt = dt

    def updateSlam(self, slam_pos):
        if self.last_slam_pos is None:
            self.last_slam_pos = slam_pos
        elif np.linalg.norm(slam_pos - self.last_slam_pos) < 1e-6:
            self.slam_updated = False
        else:
            self.last_slam_pos = slam_pos
            self.slam_updated = True

    def updateOdo(self, vel, z):
        self.last_pos += vel * self.dt
        self.last_pos[2] = z
        if self.slam_updated:
            self.last_pos[:2] = self.last_pos[:2] * self.alpha + self.last_slam_pos[:2] * (
                1 - self.alpha
            )

    def get_state(self):
        return self.last_pos.copy()


class FootOdometer:
    def __init__(self, robot_id, pb_kin, foot_link_names):
        self.robot_id = robot_id
        self.pb_kin = pb_kin
        self.foot_contact_ids = [self.pb_kin.link_names.index(name) for name in foot_link_names]
        self.last_velocity = None
        self.last_id = None

    def estimate_from_vel_continuity(self, link_states, omega):
        low_c_pos = None
        min_delta_vel = None
        for link_state in link_states:
            c_pos, c_quat, _, _, _, _, c_vel, _ = link_state
            c_pos, c_quat, c_vel = np.array(c_pos), np.array(c_quat), np.array(c_vel)
            potential_root_vel = -c_vel - np.cross(omega, c_pos)
            if low_c_pos is None:
                low_c_pos = c_pos
                root_vel = potential_root_vel
                if self.last_velocity is None:
                    self.last_velocity = root_vel.copy()
                min_delta_vel = np.linalg.norm(potential_root_vel - self.last_velocity)
            else:
                delta_vel = np.linalg.norm(
                    potential_root_vel - self.last_velocity * 0.95
                )  # prevent velocity add up unnecessarily
                if delta_vel < min_delta_vel:
                    low_c_pos = c_pos
                    root_vel = potential_root_vel
                    min_delta_vel = delta_vel
        return root_vel, -low_c_pos[2]  # assume flat ground

    def estimate_velocity_height(self, link_states, omega):
        low_c_pos = None
        for link_state in link_states:
            c_pos, c_quat, _, _, _, _, c_vel, _ = link_state
            c_pos, c_quat, c_vel = np.array(c_pos), np.array(c_quat), np.array(c_vel)
            if low_c_pos is None:
                low_c_pos = c_pos
                root_vel = -c_vel - np.cross(omega, c_pos)
            elif c_pos[2] < low_c_pos[2]:
                low_c_pos = c_pos
                root_vel = -c_vel - np.cross(omega, c_pos)
        return root_vel, -low_c_pos[2]  # assume flat ground

    def estimate_velocity(self, q, dq, quat, omega):
        """
        q: joint angles
        quat: root_orientation world frame
        omega: root_angular_velocity
        """
        omega = Rotation.from_quat(quat).apply(omega)  # convert to world frame
        pose = np.zeros(7, dtype=np.float32)
        pose[3:] = quat.copy()
        self.pb_kin.set_robot_state(q, pose, dq, omega * 0.0)
        # Get foot positions

        link_states = pb.getLinkStates(
            self.robot_id, self.foot_contact_ids, computeLinkVelocity=True
        )

        root_vel_continuity, z_continuity = self.estimate_from_vel_continuity(link_states, omega)
        root_vel_height, z_height = self.estimate_velocity_height(link_states, omega)
        if (
            np.linalg.norm(root_vel_continuity - root_vel_height) > 0.2
            and np.abs(z_height - z_continuity) > 0.05
        ):
            self.last_velocity = root_vel_height.copy()
            z = z_height  # assume flat ground
        else:
            self.last_velocity = root_vel_continuity.copy()
            z = z_continuity  # assume flat ground

        return self.last_velocity.copy(), z  # assume flat ground


def _q_norm(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _q_mul(q1, q2):  # (x,y,z,w)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _q_conj(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64)


def _q_inv(q):
    return _q_conj(_q_norm(q))


def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _yaw_from_q(q):  # ZYX yaw from (x,y,z,w)
    x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def _q_from_yaw(psi):
    half = 0.5 * psi
    s = np.sin(half)
    c = np.cos(half)
    return np.array([0.0, 0.0, s, c], dtype=np.float64)


def angle_weighted_mean(angles, weights=None):
    a = np.asarray(angles, dtype=float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    if a.shape != w.shape:
        raise ValueError("angles and weights must have same shape")
    C, S = np.sum(w * np.cos(a)), np.sum(w * np.sin(a))
    return np.arctan2(S, C)


def _rotate_vec_by_q(q, v):
    """
    Rotate vector(s) v by unit quaternion q (x,y,z,w).
    v can be shape (3,) or (...,3). Returns same shape.
    """
    v = np.asarray(v, dtype=np.float64)
    x, y, z, w = _q_norm(q)
    u = np.array([x, y, z], dtype=np.float64)
    vv = v

    # batch-safe math using formula:
    # v' = 2(u·v)u + (w^2 - u·u)v + 2w(u × v)
    u_dot_v = np.sum(vv * u, axis=-1, keepdims=True)
    u_cross_v = np.cross(u, vv)
    uu = np.dot(u, u)
    return 2.0 * u_dot_v * u + (w * w - uu) * vv + 2.0 * w * u_cross_v


def replace_heading(quat, new_heading):
    """
    Replace the heading (yaw) of a quaternion while preserving tilt (roll & pitch).

    Args:
        quat: array-like, shape (4,), quaternion [x,y,z,w]
        new_heading: float, new yaw angle in radians

    Returns:
        np.ndarray, shape (4,), quaternion with replaced heading.
    """
    r = Rotation.from_quat(quat)

    # Extract yaw (Z), pitch (Y), roll (X) in intrinsic ZYX
    yaw, pitch, roll = r.as_euler("ZYX", degrees=False)

    # Construct tilt-only rotation (keep roll & pitch)
    tilt = Rotation.from_euler("YX", [pitch, roll], degrees=False)

    # Construct new heading rotation
    new_head = Rotation.from_euler("Z", new_heading, degrees=False)

    # Combine: new heading first, then tilt
    out = new_head * tilt
    return out.as_quat()


def slerp(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions q1 and q2.
    t: interpolation factor in [0, 1].
    Returns a new quaternion.
    """
    q1 = _q_norm(q1)
    q2 = _q_norm(q2)

    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2  # ensure shortest path

    if dot > 0.9995:  # close to same direction
        return _q_norm((1.0 - t) * q1 + t * q2)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s1 = np.sin((1.0 - t) * theta_0) / sin_theta_0
    s2 = sin_theta_t / sin_theta_0

    return _q_norm(s1 * q1 + s2 * q2)


class QuaternionCollaborativeFilterSimple:
    """
    Fuse IMU (smooth, yaw-drifty) with SLAM (noisy, drift-free) to output smooth,
    drift-free orientations (x,y,z,w).

    World origin:
      - Z axis = inverse gravity (up), i.e., roll/pitch = 0
      - Heading (yaw) = yaw extracted from the FIRST IMU reading
    """

    def __init__(self, tau_yaw=8.0, tau_slam_yaw=1.0, default_dt=0.01):
        self.tau_yaw = float(tau_yaw)
        self.tau_slam_yaw = float(tau_slam_yaw)
        self.default_dt = float(default_dt)

        self._initialized = False
        self._q_origin = None  # world->IMU0 yaw-only
        self._q_imu0 = None
        self._q_slam0 = None
        self._q_extr_slam_to_imu = None  # IMU ≈ extr * SLAM

        self.last_q_imu = None
        self.current_q = None

    def reset(self):
        self.__init__(self.tau_yaw, self.tau_slam_yaw, self.default_dt)

    def _maybe_init(self, q_imu, q_slam):
        q_imu = _q_norm(q_imu)
        q_slam = _q_norm(q_slam)
        if self._initialized:
            return

        # 1) Origin with Z-up, heading from IMU0
        psi0 = _yaw_from_q(q_imu)
        self._q_origin = _q_from_yaw(psi0)  # world->IMU0 (yaw-only)

        # 2) Save first poses
        self._q_imu0 = q_imu.copy()
        self._q_slam0 = q_slam.copy()

        # 3) Fixed SLAM->IMU extrinsic from first pair: extr = IMU0 * inv(SLAM0)
        self._q_extr_slam_to_imu = _q_mul(self._q_imu0, _q_inv(self._q_slam0))

        # 4) Initialize filtered SLAM yaw in origin frame
        # q_s_in_imu = _q_mul(self._q_extr_slam_to_imu, q_slam)
        # q_s_rel = _q_mul(_q_inv(self._q_origin), q_s_in_imu)

        self._initialized = True

    def update(self, q_imu, q_slam):
        q_imu = _q_norm(q_imu)
        q_slam = _q_norm(q_slam)
        self._maybe_init(q_imu, q_slam)

        # IMU & SLAM relative to origin
        q_i_rel = _q_mul(_q_inv(self._q_origin), q_imu)
        q_s_in_imu = _q_mul(self._q_extr_slam_to_imu, q_slam)
        q_s_rel = _q_mul(_q_inv(self._q_origin), q_s_in_imu)

        if self.last_q_imu is None:
            self.last_q_imu = q_i_rel.copy()
            self.current_q = q_i_rel.copy()
            return q_i_rel
        else:
            delta_q_imu = _q_mul(_q_inv(self.last_q_imu), q_i_rel)
            self.current_q = _q_mul(self.current_q, delta_q_imu)
            # Can we only slerp yaw part
            current_heading = heading_zup(self.current_q)
            slam_heading = heading_zup(q_s_rel)
            heading = angle_weighted_mean([current_heading, slam_heading], [0.8, 0.2])
            # construct currrent_q with new heading
            self.current_q = replace_heading(self.current_q, heading)

            # self.current_q = slerp(self.current_q, q_s_rel, 0.2)  # simple SLERP to SLAM orientation
            self.last_q_imu = q_i_rel.copy()
            return self.current_q.copy()

    # NEW: rotate SLAM position(s) into the initial world frame
    def slam_pos_to_world(self, pos_slam):
        """
        Rotate a position vector (or array of vectors) expressed in the SLAM frame
        into the initial world frame (Z up, heading from first IMU).

        Args:
            pos_slam: shape (3,) or (...,3)

        Returns:
            pos_world with same shape
        """
        if not self._initialized:
            raise RuntimeError("Filter not initialized yet. Call update() at least once.")

        # Total rotation: SLAM -> IMU, then IMU -> WORLD
        # Quaternion composition for vectors: apply q_total = (IMU->WORLD) ⊗ (SLAM->IMU)
        q_imu_to_world = _q_inv(self._q_origin)
        q_total = _q_mul(q_imu_to_world, self._q_extr_slam_to_imu)

        return _rotate_vec_by_q(q_total, pos_slam)
