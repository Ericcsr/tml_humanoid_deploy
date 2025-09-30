import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseRobotEnv(ABC):
    """Base class for robot environments (both simulation and real)."""
    
    @abstractmethod
    def set_robot_state(self, q: np.ndarray, kp: Optional[np.ndarray] = None, kd: Optional[np.ndarray] = None) -> None:
        """Set the robot joint positions and optionally the control gains."""
        pass
    
    @abstractmethod
    def maintain_state(self, q: np.ndarray, kp: Optional[np.ndarray] = None, kd: Optional[np.ndarray] = None) -> None:
        """Maintain the robot in a specific state."""
        pass
    
    @abstractmethod
    def get_robot_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the current robot state (q, dq, imu_quat, omega)."""
        pass
    
    @abstractmethod
    def get_root_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the root state (position, orientation, velocity)."""
        pass
    
    @abstractmethod
    def get_anchor_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the anchor state (position, orientation)."""
        pass
    
    @abstractmethod
    def step_robot(self, action: np.ndarray) -> None:
        """Execute a control action on the robot."""
        pass
