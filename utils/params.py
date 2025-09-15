import numpy as np

ISAAC_TO_MUJOCO = np.array([
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
])

MUJOCO_TO_ISAAC = np.array([
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
])

ACTION_SCALE = np.array(
    [
        0.5475464652142303,  # left_hip_pitch_joint
        0.3506614663788243,  # left_hip_roll_joint
        0.5475464652142303,  # left_hip_yaw_joint
        0.3506614663788243,  # left_knee_joint
        0.43857731392336724,  # left_ankle_pitch_joint
        0.43857731392336724,  # left_ankle_roll_joint
        0.5475464652142303,  # right_hip_pitch_joint
        0.3506614663788243,  # right_hip_roll_joint
        0.5475464652142303,  # right_hip_yaw_joint
        0.3506614663788243,  # right_knee_joint
        0.43857731392336724,  # right_ankle_pitch_joint
        0.43857731392336724,  # right_ankle_roll_joint
        0.5475464652142303,  # waist_yaw_joint
        0.43857731392336724,  # waist_roll_joint
        0.43857731392336724,  # waist_pitch_joint
        0.43857731392336724,  # left_shoulder_pitch_joint
        0.43857731392336724,  # left_shoulder_roll_joint
        0.43857731392336724,  # left_shoulder_yaw_joint
        0.43857731392336724,  # left_elbow_joint
        0.43857731392336724,  # left_wrist_roll_joint
        0.07450087032950714,  # left_wrist_pitch_joint
        0.07450087032950714,  # left_wrist_yaw_joint
        0.43857731392336724,  # right_shoulder_pitch_joint
        0.43857731392336724,  # right_shoulder_roll_joint
        0.43857731392336724,  # right_shoulder_yaw_joint
        0.43857731392336724,  # right_elbow_joint
        0.43857731392336724,  # right_wrist_roll_joint
        0.07450087032950714,  # right_wrist_pitch_joint
        0.07450087032950714,  # right_wrist_yaw_joint
    ]
)

DEFAULT_POSE = np.array([
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        ])

REVOLUTE_JOINTS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
]

FOOT_SPHERE_NAMES = ["l1", "l2", "l3", "l4", "r1", "r2", "r3", "r4"]


ZERO = [0.0] * 29