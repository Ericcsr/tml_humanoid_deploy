import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# --- SDK Imports ---
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)

# --- Ankle API and G1 Joint Definitions ---
from tml_humanoid_deploy.ankle.ankle_api import AnkleAPI

# Using the full, robust Kp/Kd arrays for stable holding
Kp = [
    60,
    60,
    60,
    100,
    40,
    40,
    60,
    60,
    60,
    100,
    40,
    40,
    60,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
]
Kd = [
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]


class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


class TestState(Enum):
    PREPARING = 0
    CHIRPING_PR = 1
    PAUSE_BETWEEN_TESTS = 2
    CHIRPING_AB = 3
    FINISHED = 4


class RealRobotChirpTest:
    def __init__(self):
        self.control_dt = 0.02
        self.duration_prepare = 3.0
        self.duration_test = 5.0
        self.duration_pause = 3.0  # Pause between the two tests
        self.ANKLE_KP = 28.0
        self.ANKLE_KD = 1.8
        self.AMP_PITCH = 0.4
        self.FREQ_START_TEST = 0.1
        self.FREQ_END_TEST = 5

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.update_mode_machine_ = False
        self.crc = CRC()
        self.state = TestState.PREPARING

        self.mode_machine_ = 0
        self.first_state_received_ = False
        self.initial_q_positions_ = None

        # --- Expanded Data Logging ---
        self.time_log_pr = []
        self.commanded_pitch_log_pr = []
        self.measured_pitch_log_pr = []
        self.frequency_log_pr = []
        self.time_log_ab = []
        self.commanded_pitch_log_ab = []
        self.measured_pitch_log_ab = []
        self.frequency_log_ab = []

        self.start_time = 0.0
        self.time_since_start = 0.0
        self.current_step = 0

        self.ankle_api = AnkleAPI(model_dir="/home/takaraet/Projects/tml_humanoid_deploy/tml_humanoid_deploy/ankle/trained_models")

    def _generate_trajectories(self, t, amp, f_start, f_end=None):
        if f_end is None or f_start == f_end:
            freq = np.full_like(t, f_start)
            phase = 2 * np.pi * freq * t
        else:
            T = t[-1]
            freq = f_start + (f_end - f_start) * t / T
            phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * T))
        pos = amp * np.sin(phase)
        one_second = int(1 / self.control_dt)
        prepend_pos = np.full((one_second,), pos[0])
        prepend_freq = np.full((one_second,), freq[0])
        pos = np.concatenate((prepend_pos, pos))
        freq = np.concatenate((prepend_freq, freq))
        return pos, freq

    def _plot_results(self):
        fig, axs = plt.subplots(2, 1, figsize=(15, 12), constrained_layout=True)
        fig.suptitle(
            "Real Robot Ankle Tracking: Pitch/Roll vs. A/B Linkage", fontsize=16
        )

        # --- Process and Plot Time Domain ---
        axs[0].set_title("Time Domain Response")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Ankle Pitch (rad)")
        # Commanded (use AB log as it's identical)
        time_vec_cmd = np.array(self.time_log_ab)
        commanded_pos = np.array(self.commanded_pitch_log_ab)
        axs[0].plot(
            time_vec_cmd, commanded_pos, "k--", label="Commanded Position", linewidth=2
        )
        # PR Results
        time_vec_pr = np.array(self.time_log_pr)
        measured_pos_pr = np.array(self.measured_pitch_log_pr)
        axs[0].plot(
            time_vec_pr, measured_pos_pr, "b-", label=f"Pitch/Roll Mode", alpha=0.8
        )
        # AB Results
        time_vec_ab = np.array(self.time_log_ab)
        measured_pos_ab = np.array(self.measured_pitch_log_ab)
        axs[0].plot(
            time_vec_ab, measured_pos_ab, "r-", label=f"A/B Linkage Mode", alpha=0.8
        )
        axs[0].legend()
        axs[0].grid(True)

        # --- Process and Plot Frequency Domain ---
        axs[1].set_title("Absolute Tracking Error vs. Frequency")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Absolute Error (rad)")
        # PR Error
        freq_vec_pr = np.array(self.frequency_log_pr)
        error_pr = np.array(self.commanded_pitch_log_pr) - measured_pos_pr
        axs[1].plot(
            freq_vec_pr, np.abs(error_pr), "b.", markersize=2, label="Pitch/Roll Error"
        )
        # AB Error
        freq_vec_ab = np.array(self.frequency_log_ab)
        error_ab = np.array(self.commanded_pitch_log_ab) - measured_pos_ab
        axs[1].plot(
            freq_vec_ab, np.abs(error_ab), "r.", markersize=2, label="A/B Linkage Error"
        )
        # Moving Averages
        window = 100
        if len(error_pr) > window:
            smooth_error_pr = np.convolve(
                np.abs(error_pr), np.ones(window) / window, mode="valid"
            )
            smooth_freq_pr = np.convolve(
                freq_vec_pr, np.ones(window) / window, mode="valid"
            )
            axs[1].plot(smooth_freq_pr, smooth_error_pr, "b-", linewidth=3)
        if len(error_ab) > window:
            smooth_error_ab = np.convolve(
                np.abs(error_ab), np.ones(window) / window, mode="valid"
            )
            smooth_freq_ab = np.convolve(
                freq_vec_ab, np.ones(window) / window, mode="valid"
            )
            axs[1].plot(smooth_freq_ab, smooth_error_ab, "r-", linewidth=3)
        axs[1].legend()
        axs[1].grid(True)

        plt.show()

    def Init(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
        
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        total_duration = self.duration_test + 1.0
        time_vec = np.arange(0, total_duration, self.control_dt)
        self.chirp_pos_trajectory, self.chirp_freq_trajectory = (
            self._generate_trajectories(
                time_vec, self.AMP_PITCH, self.FREQ_START_TEST, self.FREQ_END_TEST
            )
        )
        self.controlThread = RecurrentThread(self.control_dt, self.ControlLoop)
        while self.update_mode_machine_ == False:
            time.sleep(1)

        if self.update_mode_machine_ == True:
            print(
                f"Initial state received. Captured mode_machine: {self.mode_machine_}. Robot will now hold pose."
            )
            self.start_time = time.time()
            self.controlThread.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True

    def ControlLoop(self):
        if self.low_state is None:
            return
        self.time_since_start = time.time() - self.start_time

        # --- Main State Machine ---
        if self.state == TestState.PREPARING:
            if self.time_since_start > self.duration_prepare:
                self.state = TestState.CHIRPING_PR
                self.current_step = 0
                print("\nPreparation complete. Starting Pitch/Roll (PR) test...")
            for i in range(29):
                ratio = np.clip(self.time_since_start / self.duration_prepare, 0.0, 1.0)
                self.low_cmd.mode_pr = 0
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = 0. 
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.low_state.motor_state[i].q 
                self.low_cmd.motor_cmd[i].dq = 0. 
                self.low_cmd.motor_cmd[i].kp = Kp[i] 
                self.low_cmd.motor_cmd[i].kd = Kd[i]

        elif self.state == TestState.CHIRPING_PR:
            if self.current_step >= len(self.chirp_pos_trajectory):
                self.state = TestState.PAUSE_BETWEEN_TESTS
                self.start_time = time.time()
                print("\nPR test finished. Pausing before AB test...")
                return

            commanded_pitch = self.chirp_pos_trajectory[self.current_step]
            measured_pitch = self.low_state.motor_state[G1JointIndex.LeftAnklePitch].q
            self.time_log_pr.append(self.current_step * self.control_dt)
            self.commanded_pitch_log_pr.append(commanded_pitch)
            self.measured_pitch_log_pr.append(measured_pitch)
            self.frequency_log_pr.append(self.chirp_freq_trajectory[self.current_step])

            self.low_cmd.motor_cmd[G1JointIndex.LeftAnklePitch].q = commanded_pitch
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnklePitch].kp = self.ANKLE_KP
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnklePitch].kd = self.ANKLE_KD
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleRoll].q = (
                0.0  # Keep roll at zero
            )

            self.low_cmd.mode_pr = 0  # Set PR Mode
            self.current_step += 1

        elif self.state == TestState.PAUSE_BETWEEN_TESTS:
            if self.time_since_start > self.duration_pause:
                self.state = TestState.CHIRPING_AB
                self.current_step = 0

                print("Pause complete. Starting A/B Linkage test...")
            # transition to the start position
            alpha = min(self.time_since_start / self.duration_pause, 1.0)

            # Interpolate motor positions towards the first AB target
            first_pitch = self.chirp_pos_trajectory[0]
            current_pitch= self.low_state.motor_state[G1JointIndex.LeftAnklePitch].q
            interp_target = alpha * first_pitch + (1-alpha) * current_pitch
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnklePitch].q = interp_target
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnklePitch].kp = self.ANKLE_KP
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnklePitch].kd = self.ANKLE_KD
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleRoll].q = (
                0.0  # Keep roll at zero
            )

            self.low_cmd.mode_pr = 0  # Set PR Mode

        elif self.state == TestState.CHIRPING_AB:
            if self.current_step >= len(self.chirp_pos_trajectory):
                self.state = TestState.FINISHED
                return

            commanded_pitch = self.chirp_pos_trajectory[self.current_step]
            q_targets = self.ankle_api.get_ik(
                np.array([[commanded_pitch, 0.0]]), warn=False, side="left",
            )[0]
            q_measured_A = self.low_state.motor_state[G1JointIndex.LeftAnkleA].q
            q_measured_B = self.low_state.motor_state[G1JointIndex.LeftAnkleB].q
            measured_pitch = self.ankle_api.get_fk(
                np.array([[q_measured_A, q_measured_B]]), warn=False, side="left"
            )[0, 0]

            self.time_log_ab.append(self.current_step * self.control_dt)
            self.commanded_pitch_log_ab.append(commanded_pitch)
            self.measured_pitch_log_ab.append(measured_pitch)
            self.frequency_log_ab.append(self.chirp_freq_trajectory[self.current_step])

            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleA].q = q_targets[0]
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleA].kp = self.ANKLE_KP
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleA].kd = self.ANKLE_KD
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleB].q = q_targets[1]
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleB].kp = self.ANKLE_KP
            self.low_cmd.motor_cmd[G1JointIndex.LeftAnkleB].kd = self.ANKLE_KD

            self.low_cmd.mode_pr = 1  # Set AB Mode
            self.current_step += 1

        elif self.state == TestState.FINISHED:
            return

        # --- Set common fields for every command ---
        self.low_cmd.mode_machine = self.mode_machine_
        for i in range(29):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


if __name__ == "__main__":
    print("--- Unitree G1 Real Robot Ankle Chirp Test ---")
    print("\n!!! WARNING !!!")
    print("This script will move the robot's left ankle joint in two different modes.")
    print("Ensure the robot is safely supported with no obstacles around the legs.")
    input("Press Enter to continue...")
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
    test = RealRobotChirpTest()
    test.Init()
    test.Start()
    while True:
        if test.state == TestState.FINISHED:
            print("\nTest finished. Plotting comparative results.")
            test._plot_results()
            break
        time.sleep(1)
