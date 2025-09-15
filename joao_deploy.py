import argparse
import json
import time
import numpy as np
import mujoco
import torch
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
import os

import onnx
import onnxruntime as ort

from utils.isaac_utils import (
    matrix_from_quat,
    subtract_frame_transforms,
    quat_rotate_inverse,
)

import rerun as rr



def motion_anchor_pos_b(
    robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w
) -> torch.Tensor:
    pos, _ = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        anchor_pos_w,
        anchor_quat_w,
    )
    return pos.view(1, -1)


def motion_anchor_ori_b(
    robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w
) -> torch.Tensor:
    _, ori = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        anchor_pos_w,
        anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def base_lin_vel(root_link_quat_w, root_lin_vel_w) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    return quat_rotate_inverse(root_link_quat_w, root_lin_vel_w)


def base_ang_vel(root_link_quat_w, root_ang_vel_w) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    return quat_rotate_inverse(root_link_quat_w, root_ang_vel_w)


def joint_pos_rel(joint_pos, default_joint_pos) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    return joint_pos - default_joint_pos


def joint_vel_rel(joint_vel):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    return joint_vel


# -------------------------------------------------------------------
# Main low-level policy controller that:
#   - reads mimic obs from Redis
#   - feeds into policy
#   - runs the sim
# -------------------------------------------------------------------


class RealTimePolicyController:
    def __init__(
        self,
        xml_file,
        policy_path,
        device="cuda",
        record_video=False,
        record_proprio=False,
        headless=True,
    ):
        self.device = device
        self.headless = headless

        # Load policy
        self.session = ort.InferenceSession(policy_path)
        print(f"Policy loaded from {policy_path}")

        # Load the metadata
        self.model_metadata = self.session.get_modelmeta().custom_metadata_map
        self.model_metadata["joint_names"] = self.model_metadata["joint_names"].split(
            ","
        )
        self.model_metadata["joint_stiffness"] = [
            float(i) for i in self.model_metadata["joint_stiffness"].split(",")
        ]
        self.model_metadata["joint_damping"] = [
            float(i) for i in self.model_metadata["joint_damping"].split(",")
        ]
        self.model_metadata["default_joint_pos"] = [
            float(i) for i in self.model_metadata["default_joint_pos"].split(",")
        ]
        self.model_metadata["action_scale"] = [
            float(i) for i in self.model_metadata["action_scale"].split(",")
        ]
        #self.model_metadata["body_names"] = self.model_metadata["body_names"].split(",")

        # Create a mapping between joints and metadata
        joint_stiffness = dict(
            zip(
                self.model_metadata["joint_names"],
                self.model_metadata["joint_stiffness"],
            )
        )
        joint_damping = dict(
            zip(
                self.model_metadata["joint_names"], self.model_metadata["joint_damping"]
            )
        )
        default_joint_pos = dict(
            zip(
                self.model_metadata["joint_names"],
                self.model_metadata["default_joint_pos"],
            )
        )
        joint_action_scale = dict(
            zip(self.model_metadata["joint_names"], self.model_metadata["action_scale"])
        )

        root_body = "pelvis"
        self.root_body_id = 0#self.model_metadata["body_names"].index(root_body)
        self.anchor_body_name = "torso_link"
        self.anchor_body_id = 9#self.model_metadata["body_names"].index("torso_link")

        # Get the reference motion length
        model = onnx.load(policy_path)
        for node in model.graph.node:
            if node.name == "/Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        self.length = onnx.numpy_helper.to_array(attr.t)
        print(f"Constant node found: {node.name}, value: {self.length}")

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.002  # the physics simulation runs at this time step
        self.data = mujoco.MjData(self.model)
        self.robot_root_body_id = self.model.body(root_body).id
        self.robot_anchor_body_id = self.model.body(self.anchor_body_name).id

        # Set the parameters in the correct order
        model_joint_names = [
            self.model.joint(i).name for i in range(1, self.model.njnt)
        ]
        self.isaaclab2mujoco = [
            self.model_metadata["joint_names"].index(name) for name in model_joint_names
        ]
        self.mujoco2isaaclab = [
            model_joint_names.index(name) for name in self.model_metadata["joint_names"]
        ]
        # default_dof_pos = np.array(
        #     [default_joint_pos[name] for name in model_joint_names]
        # )
        stiffness = np.array([joint_stiffness[name] for name in model_joint_names])
        damping = np.array([joint_damping[name] for name in model_joint_names])
        # action_scale = np.array(
        #     [joint_action_scale[name] for name in model_joint_names]
        # )
        default_dof_pos = np.array(self.model_metadata["default_joint_pos"])
        # stiffness = np.array(self.model_metadata["joint_stiffness"])
        # damping = np.array(self.model_metadata["joint_damping"])
        action_scale = np.array(self.model_metadata["action_scale"])

        if not self.headless:
            self.viewer = mjv.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
            self.viewer.cam.distance = 2.0

        # Example defaults & placeholders
        self.num_actions = 29
        self.sim_duration = 100000.0
        self.sim_dt = 0.002
        self.sim_decimation = 10  # the policy is called every 10 time steps

        # PD Gains, etc. (adapt as needed)
        self.default_dof_pos = default_dof_pos
        self.mujoco_default_dof_pos = np.concatenate(
            [np.array([0, 0, 0.793]), np.array([1, 0, 0, 0]), default_dof_pos]
        )
        self.stiffness = stiffness
        self.damping = damping
        self.torque_limits = np.array(
            [
                88,
                139,
                88,
                139,
                50,
                50,
                88,
                139,
                88,
                139,
                50,
                50,
                88,
                50,
                50,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
            ]
        )
        self.torque_limits[20] = 5
        self.torque_limits[27] = 5

        self.last_action = np.zeros(self.num_actions)

        self.action_scale = action_scale

        self.record_video = record_video
        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None

    def extract_data(self):
        qpos = self.data.qpos.astype(np.float32)
        qvel = self.data.qvel.astype(np.float32)

        dof_pos = qpos[7:]
        dof_vel = qvel[6:]

        return dof_pos, dof_vel

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, qpos, qvel):
        # body & hand
        self.data.qpos[:7] = qpos[:7]
        # self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def get_state(self, time_step):
        """
        Returns the dof position and velocity at time step time_step.
        """
        # The ONNX policy has a time_step input and outputs joint_pos and joint_vel
        ort_inputs = {
            "obs": np.zeros(self.session.get_inputs()[0].shape, dtype=np.float32),
            "time_step": np.array([[time_step]], dtype=np.float32),
        }
        ort_outs = self.session.run(None, ort_inputs)

        output_names = [output.name for output in self.session.get_outputs()]
        body_pos_output_idx = output_names.index("body_pos_w")
        body_quat_output_idx = output_names.index("body_quat_w")
        joint_pos_output_idx = output_names.index("joint_pos")

        body_vel_output_idx = output_names.index("body_lin_vel_w")
        body_ang_vel_output_idx = output_names.index("body_ang_vel_w")
        joint_vel_output_idx = output_names.index("joint_vel")

        qpos = np.concatenate(
            [
                ort_outs[body_pos_output_idx].squeeze()[self.root_body_id],
                ort_outs[body_quat_output_idx].squeeze()[self.root_body_id],
                ort_outs[joint_pos_output_idx].squeeze()[self.isaaclab2mujoco],
            ]
        )

        qvel = np.concatenate(
            [
                ort_outs[body_vel_output_idx].squeeze()[self.root_body_id],
                ort_outs[body_ang_vel_output_idx].squeeze()[self.root_body_id],
                ort_outs[joint_vel_output_idx].squeeze()[self.isaaclab2mujoco],
            ]
        )
        return (qpos, qvel)

    def get_observation(self, time_step):
        # The ONNX policy has a time_step input and outputs joint_pos and joint_vel
        ort_inputs = {
            "obs": np.zeros(self.session.get_inputs()[0].shape, dtype=np.float32),
            "time_step": np.array([[time_step]], dtype=np.float32),
        }
        ort_outs = self.session.run(None, ort_inputs)
        output_names = [output.name for output in self.session.get_outputs()]
        joint_pos_output_idx = output_names.index("joint_pos")
        joint_vel_output_idx = output_names.index("joint_vel")

        body_pos_output_idx = output_names.index("body_pos_w")
        anchor_pos_w = torch.from_numpy(
            ort_outs[body_pos_output_idx].squeeze()[self.anchor_body_id],
        )
        body_quat_output_idx = output_names.index("body_quat_w")
        anchor_quat_w = torch.from_numpy(
            ort_outs[body_quat_output_idx].squeeze()[self.anchor_body_id],
        )
        anchor_pos_b = motion_anchor_pos_b(
            robot_anchor_pos_w=torch.from_numpy(
                self.data.xpos[self.robot_anchor_body_id]
            ),
            robot_anchor_quat_w=torch.from_numpy(
                self.data.xquat[self.robot_anchor_body_id]
            ),
            anchor_pos_w=anchor_pos_w,
            anchor_quat_w=anchor_quat_w,
        )
        anchor_ori_b = motion_anchor_ori_b(
            robot_anchor_pos_w=torch.from_numpy(
                self.data.xpos[self.robot_anchor_body_id]
            ),
            robot_anchor_quat_w=torch.from_numpy(
                self.data.xquat[self.robot_anchor_body_id]
            ),
            anchor_pos_w=anchor_pos_w,
            anchor_quat_w=anchor_quat_w,
        )

        return torch.cat(
            [
                torch.from_numpy(
                    ort_outs[joint_pos_output_idx].squeeze()
                ),  # command joint position
                torch.from_numpy(
                    ort_outs[joint_vel_output_idx].squeeze()
                ),  # command joint velocity
                anchor_pos_b.squeeze(),  # position error in tracking the anchor
                anchor_ori_b.flatten(),  # orientation error in tracking the anchor
                base_lin_vel(
                    torch.from_numpy(self.data.xquat[1]),
                    torch.from_numpy(self.data.cvel[1, 3:]),  # self.data.qvel[:3]
                ),  # Root velocity in root frame
                base_ang_vel(
                    torch.from_numpy(self.data.xquat[1]),
                    torch.from_numpy(self.data.cvel[1, :3]),  # self.data.qvel[3:6]
                ),  # Root angular velocity in root frame
                joint_pos_rel(
                    torch.from_numpy(self.data.qpos[7:][self.mujoco2isaaclab]),
                    # this one is in the correct order for the policy. For MuJoCo
                    # we have the other one
                    np.array(self.model_metadata["default_joint_pos"]),
                ),
                torch.from_numpy(self.data.qvel[6:][self.mujoco2isaaclab]),
                torch.from_numpy(self.last_action),
            ],
            dim=-1,
        )

    def run(self):
        # Optionally record video
        if self.record_video:
            import imageio

            video_name = "debug_sim.mp4"
            print(f"Saving video to {video_name}")
            mp4_writer = imageio.get_writer(video_name, fps=50)
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(*self.get_state(0))

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating...")

        try:
            for i in pbar:
                rr.set_time_sequence("frame", i)
                t_start = time.time()
                dof_pos, dof_vel = self.extract_data()
                for idx in range(len(dof_pos)):
                    rr.log(f"dof_pos/{idx}", rr.Scalar(dof_pos[idx].item()))

                if i // self.sim_decimation > self.length:
                    print("Reference motion finished.")
                    break

                if i % self.sim_decimation == 0:  # Query the policy
                    # Compute the final observation
                    obs_tensor = (
                        self.get_observation(i // self.sim_decimation)
                        .float()
                        .unsqueeze(0)
                    )

                    # Run inference
                    ort_inputs = {
                        "obs": obs_tensor.numpy(),
                        "time_step": np.array([[0]], dtype=np.float32),
                    }
                    raw_action = self.session.run(["actions"], ort_inputs)[0].squeeze()

                    # Store the raw action and process it
                    self.last_action = raw_action
                    # raw_action = np.clip(raw_action, -10.0, 10.0)
                    scaled_actions = raw_action * self.action_scale
                    pd_target = scaled_actions + self.default_dof_pos

                    # make camera follow the pelvis
                    if not self.headless:
                        pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                        self.viewer.cam.lookat = pelvis_pos
                        self.viewer.sync()
                        if mp4_writer is not None:
                            img = self.viewer.read_pixels()
                            mp4_writer.append_data(img)

                    # Record proprio if enabled
                    if self.record_proprio:
                        proprio_data = {
                            "timestamp": time.time(),
                            "dof_pos": dof_pos.tolist(),
                            "dof_vel": dof_vel.tolist(),
                            # "ang_vel": ang_vel.tolist(),
                        }
                        self.proprio_recordings.append(proprio_data)

                # PD control
                torque = (
                    pd_target[self.isaaclab2mujoco] - dof_pos
                ) * self.stiffness - dof_vel * self.damping
                # # Reindex
                # torque = torque
                # Clip
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)

                # Apply the torques directly after fixing the order
                self.data.qfrc_applied[6:] = torque

                mujoco.mj_step(self.model, self.data)
                # sleep to maintain real-time pace
                # elapsed = time.time() - t_start
                # if elapsed < self.sim_dt:
                #     time.sleep(self.sim_dt - elapsed)
        except Exception as e:
            print(f"Error in run: {e}")
            pass
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved")

            # Save proprio recordings if enabled
            if self.record_proprio:
                name = "reach"
                proprio_file = f"logs/proprio_recordings_sim_{name}.json"
                with open(proprio_file, "w") as f:
                    json.dump(self.proprio_recordings, f)
                print(f"Proprio recordings saved to {proprio_file}")

            if not self.headless:
                self.viewer.close()


def main_low_level_sim(args):
    controller = RealTimePolicyController(
        xml_file=args.xml_file,
        policy_path=args.onnx,
        device="cuda",
        record_video=args.record_video,
        headless=args.headless,
    )
    controller.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    HERE = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--onnx", help="Path to the ONNX model")
    parser.add_argument(
        "--xml_file",
        help="Mujoco XML file",
    )
    parser.add_argument("--record_video", action="store_true", help="Record a video")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    rr.init("BeyondMimic", spawn=True)

    main_low_level_sim(args)