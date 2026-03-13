import importlib
import os
import pkgutil
import threading
from typing import List, Dict
from cv_bridge import CvBridge
import logging

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Image, CameraInfo
import message_filters
import rerun as rr


from mm_actions.actions import __path__ as actions_path
from mm_actions.reasoning.gemini_client import GeminiRoboticsClient
from mm_interface.action import TaskCommand


class MmActionsNode(Node):
    def __init__(self) -> None:
        super().__init__('mm_actions')
        self._busy_lock = threading.Lock()
        self._busy = False
        self.bridge = CvBridge()
        self._latest_image = None
        self._latest_joint_state = None
        self._gemini_client: None

        api_key = os.getenv("GOOGLE_API_KEY")
        self._gemini_client = GeminiRoboticsClient(api_key=api_key)

        self.image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw', qos_profile=10)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw', qos_profile=10)
        self.depth_info_sub = message_filters.Subscriber(self, CameraInfo, "/camera/aligned_depth_to_color/camera_info", qos_profile=10)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.depth_info_sub],
            queue_size=30,
            slop=0.05
        )
        self.ts.registerCallback(self._synced_image_cb)

        self.arm_joint_state_sub = self.create_subscription(
            JointState,
            'joint_states_feedback',
            self._arm_joint_state_cb,
            1
        )

        self.arm_cmd_pub = self.create_publisher(JointState, '/joint_states', 10)

        self._dispatch = self._load_actions()

        self._action_server = ActionServer(
            self,
            TaskCommand,
            'task_command',
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
        )
        rr.init("mm_actions", spawn=False)
        rr.connect_grpc()

    def _synced_image_cb(self, img_msg, depth_msg, depth_info_msg) -> None:
        self._latest_image = {
            "rgb": self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8"),
            "depth": self.bridge.imgmsg_to_cv2(depth_msg),
            "intrinsics": {
                'fx': depth_info_msg.k[0],
                'fy': depth_info_msg.k[4],
                'cx': depth_info_msg.k[2],
                'cy': depth_info_msg.k[5],
            }
        }
    
    def _arm_joint_state_cb(self, joint_state_msg):
        self._latest_joint_state = joint_state_msg.position

    def get_image(self) -> Dict:
        return self._latest_image
    
    def get_joint_state(self) -> String:
        return self._latest_joint_state

    def publish_arm_cmd(self, q: List[float], gripper: float = None):
        """Publish an arm joint command with a gripper position.
        Args:
            q: Six joint positions for joints 1-6, in order.
            gripper: Gripper position from 0.0 to 0.1
        """
        msg = JointState()
        q = list(q)
        assert len(q) == 6, f"Expected 6 joint commands for the arm, got {len(q)}"
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        msg.position = q
        if gripper is not None:
            msg.position = q + [gripper]
            msg.name += ["gripper"]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.arm_cmd_pub.publish(msg)

    def goal_cb(self, _goal_request: TaskCommand.Goal) -> GoalResponse:
        with self._busy_lock:
            if self._busy:
                self.get_logger().warn('Rejecting goal: another action is running')
                return GoalResponse.REJECT
            self._busy = True
            return GoalResponse.ACCEPT

    def cancel_cb(self, _goal_handle) -> CancelResponse:
        with self._busy_lock:
            self._busy = False
        return CancelResponse.ACCEPT

    def destroy_node(self) -> bool:
        self._action_server.destroy()
        return super().destroy_node()

    def execute_cb(self, goal_handle) -> TaskCommand.Result:

        action_name = None
        image = self.get_image()
        if image is None or image.get("rgb") is None:
            result = TaskCommand.Result()
            result.success = False
            result.message = "No synchronized RGB image available."
            goal_handle.abort()
            self._finish_action()
            return result

        joint_state_at_image = self.get_joint_state()
        print("Querying Gemini")
        decision = self._gemini_client.decide_task(
            image_rgb=image["rgb"],
            instruction=goal_handle.request.command,
        )
        if decision is None:
            result = TaskCommand.Result()
            result.success = False
            result.message = "Gemini could not determine a task."
            goal_handle.abort()
            self._finish_action()
            return result
        
        action_name = decision.action
        print(f"Success, executing {action_name} action")
        action_cls = self._dispatch.get(action_name)
        if action_cls is None:
            result = TaskCommand.Result()
            result.success = False
            result.message = f'unknown action: {action_name}'
            goal_handle.abort()
            self._finish_action()
            return result
        try:
            action = action_cls(
                self.get_image,
                self.get_joint_state,
                self.publish_arm_cmd,
                image,
                decision.point,
                joint_state_at_image,
            )
            success, message = action.run()
            result = TaskCommand.Result()
            result.success = bool(success)
            result.message = message
            if result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result
        finally:
            self._finish_action()

    def _load_actions(self):
        actions = {}
        for _, name, _ in pkgutil.iter_modules(actions_path):
            module = importlib.import_module(f'mm_actions.actions.{name}')
            action_name = getattr(module, 'ACTION_NAME', None)
            action_cls = getattr(module, 'ACTION_CLASS', None)
            if action_name and action_cls is not None:
                actions[action_name] = action_cls
        return actions

    def _finish_action(self) -> None:
        with self._busy_lock:
            self._busy = False


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    node = MmActionsNode()
    try:
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
