"""Optional ROS publisher for root pose and body-frame linear velocity."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

_ROS_BACKEND: Optional[str] = None
_Odometry: Any = None
_init_node: Any = None
_now: Any = None


def _load_ros_backend() -> str:
    global _ROS_BACKEND, _Odometry, _init_node, _now
    if _ROS_BACKEND is not None:
        return _ROS_BACKEND

    try:
        import rclpy
        from nav_msgs.msg import Odometry

        def _init(node_name: str) -> None:
            if not rclpy.ok():
                rclpy.init()

        _ROS_BACKEND = "ros2"
        _Odometry = Odometry
        _init_node = _init
        _now = None
        return _ROS_BACKEND
    except ImportError:
        pass

    try:
        import rospy
        from nav_msgs.msg import Odometry

        def _init(node_name: str) -> None:
            if not rospy.core.is_initialized():
                rospy.init_node(node_name, anonymous=True)

        def _stamp():
            return rospy.Time.now()

        _ROS_BACKEND = "ros1"
        _Odometry = Odometry
        _init_node = _init
        _now = _stamp
        return _ROS_BACKEND
    except ImportError as exc:
        raise ImportError(
            "ROS publishing requires rclpy (ROS 2) or rospy (ROS 1) with nav_msgs."
        ) from exc


class RootPoseRosPublisher:
    """Publish root pose and body-frame linear velocity as nav_msgs/Odometry."""

    def __init__(
        self,
        topic: str,
        node_name: str,
        frame_id: str = "map",
        child_frame_id: str = "base_link",
        queue_size: int = 10,
    ) -> None:
        backend = _load_ros_backend()
        _init_node(node_name)

        if backend == "ros2":
            import rclpy
            from rclpy.node import Node

            self._node = Node(node_name)
            self._pub = self._node.create_publisher(_Odometry, topic, queue_size)
            self._stamp = lambda: self._node.get_clock().now().to_msg()
            self._spin = lambda: rclpy.spin_once(self._node, timeout_sec=0.0)
        else:
            import rospy

            self._node = None
            self._pub = rospy.Publisher(topic, _Odometry, queue_size=queue_size)
            self._stamp = _now
            self._spin = lambda: None

        self._frame_id = frame_id
        self._child_frame_id = child_frame_id

    def publish(
        self,
        root_pos: np.ndarray,
        root_quat_xyzw: np.ndarray,
        root_lin_vel_body: np.ndarray,
    ) -> None:
        pos = np.asarray(root_pos, dtype=np.float64).reshape(3)
        quat = np.asarray(root_quat_xyzw, dtype=np.float64).reshape(4)
        vel = np.asarray(root_lin_vel_body, dtype=np.float64).reshape(3)

        msg = _Odometry()
        msg.header.stamp = self._stamp()
        msg.header.frame_id = self._frame_id
        msg.child_frame_id = self._child_frame_id

        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.x = float(quat[0])
        msg.pose.pose.orientation.y = float(quat[1])
        msg.pose.pose.orientation.z = float(quat[2])
        msg.pose.pose.orientation.w = float(quat[3])

        msg.twist.twist.linear.x = float(vel[0])
        msg.twist.twist.linear.y = float(vel[1])
        msg.twist.twist.linear.z = float(vel[2])

        self._pub.publish(msg)
        self._spin()
