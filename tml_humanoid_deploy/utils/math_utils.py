import numpy as np
from scipy.spatial.transform import Rotation

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

def heading_quat(quat):
    yaw = heading_zup(quat)
    return Rotation.from_euler("z", yaw).as_quat()

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    
    Parameters
    ----------
    q1, q2 : array-like, shape (4,)
        Quaternions in [w, x, y, z] format.
    
    Returns
    -------
    product : ndarray, shape (4,)
        The quaternion product q1 * q2 in [w, x, y, z] format.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def add_random_rotation_noise(quat, noise_std):
    """
    Add random rotation noise to a quaternion.
    
    The function interprets the input quaternion as [x, y, z, w] and adds a random
    rotation by sampling:
      - A random unit axis uniformly on the sphere.
      - A rotation angle from a normal distribution with standard deviation `noise_std` (radians).
      
    The noise rotation is applied by left-multiplying the original quaternion.
    
    Parameters
    ----------
    quat : array-like, shape (4,)
        The input quaternion in [x, y, z, w] format.
    noise_std : float
        Standard deviation of the rotation noise (in radians).
        
    Returns
    -------
    noisy_quat : ndarray, shape (4,)
        The quaternion with added rotation noise, in [x, y, z, w] format.
    """
    # Ensure input is a NumPy array and normalized.
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    
    # Convert from [x, y, z, w] to [w, x, y, z] for computation.
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)
    
    # Sample a random unit axis.
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    
    # Sample a random noise angle from a normal distribution.
    angle = np.random.normal(loc=0.0, scale=noise_std)
    
    # Build the noise quaternion in [w, x, y, z] format.
    half_angle = angle / 2.0
    noise_q = np.concatenate(([np.cos(half_angle)], np.sin(half_angle) * axis))
    
    # Apply the noise by quaternion multiplication: noise * quat.
    noisy_q_wxyz = quaternion_multiply(noise_q, quat_wxyz)
    noisy_q_wxyz /= np.linalg.norm(noisy_q_wxyz)  # Ensure normalization
    
    # Convert back to [x, y, z, w] format.
    noisy_quat = np.array([noisy_q_wxyz[1], noisy_q_wxyz[2], noisy_q_wxyz[3], noisy_q_wxyz[0]])
    return noisy_quat

def normalize(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a batch of quaternions."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (norm + eps)

def yaw_quat(quat: np.ndarray) -> np.ndarray:
    """
    Extract the yaw component of a quaternion (w, x, y, z).
    
    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)
    
    Returns:
        A quaternion with only yaw component. Same shape as input.
    """
    shape = quat.shape
    quat_yaw = quat.reshape(-1, 4)

    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]

    # yaw angle
    yaw = np.arctan2(2 * (qw * qz + qx * qy),
                     1 - 2 * (qy * qy + qz * qz))

    # construct yaw-only quaternion
    quat_yaw_out = np.zeros_like(quat_yaw)
    quat_yaw_out[:, 0] = np.cos(yaw / 2.0)
    quat_yaw_out[:, 3] = np.sin(yaw / 2.0)

    quat_yaw_out = normalize(quat_yaw_out)
    return quat_yaw_out.reshape(shape)