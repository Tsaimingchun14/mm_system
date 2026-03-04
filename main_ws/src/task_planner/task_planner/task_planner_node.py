import os
import threading
import uuid
from pathlib import Path

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dotenv import load_dotenv

from mm_interface.action import TaskCommand
from mm_interface.msg import MotionPlannerCommand, MotionPlannerFeedback
from .gemini_client import GeminiRoboticsClient

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__("task_planner_node")

        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            self.get_logger().error(
                "No Gemini API key found. "
                "Set env var GOOGLE_API_KEY."
            )
            raise RuntimeError("Missing Gemini API key.")

        # Runtime state
        self.latest_image_msg = None
        self.active_goal_handle = None
        self.active_motion_request_id = ""
        self.active_goal_result_status = 0
        self.active_goal_result_message = ""
        self.active_goal_done_event = threading.Event()
        self.state_lock = threading.Lock()
        self.goal_in_progress = False

        # Utilities / clients
        self.bridge = CvBridge()
        self.gemini = GeminiRoboticsClient(api_key=api_key, model=GeminiRoboticsClient.MODEL)

        # Action server
        self.task_action_server = ActionServer(
            self,
            TaskCommand,
            "task_command",
            execute_callback=self.execute_task_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
        )

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, "/image_raw", self.image_cb, 10
        )
        self.motion_planner_feedback_sub = self.create_subscription(
            MotionPlannerFeedback,
            "/motion_planner_feedback",
            self.motion_feedback_cb,
            10,
        )

        # Publications
        self.motion_planner_cmd_pub = self.create_publisher(
            MotionPlannerCommand, "/motion_planner_command", 10
        )

        self.get_logger().info("TaskPlannerNode ready.\n")

    def image_cb(self, msg: Image):
        self.latest_image_msg = msg

    def goal_cb(self, _goal_request: TaskCommand.Goal):
        with self.state_lock:
            if self.goal_in_progress:
                self.get_logger().warn("Task action goal rejected: already processing another goal.")
                return GoalResponse.REJECT
            self.goal_in_progress = True
        return GoalResponse.ACCEPT

    def cancel_cb(self, _goal_handle):
        return CancelResponse.REJECT

    def execute_task_cb(self, goal_handle):
        with self.state_lock:
            self.active_goal_handle = goal_handle
            self.active_motion_request_id = ""
            self.active_goal_result_status = 0
            self.active_goal_result_message = ""
            self.active_goal_done_event.clear()

        command = goal_handle.request.command.strip()
        if not command:
            goal_handle.abort()
            result = TaskCommand.Result()
            result.success = False
            result.message = "Empty command."
            with self.state_lock:
                self.active_goal_handle = None
                self.goal_in_progress = False
            return result

        if self.latest_image_msg is None:
            goal_handle.abort()
            result = TaskCommand.Result()
            result.success = False
            result.message = "No camera image available."
            with self.state_lock:
                self.active_goal_handle = None
                self.goal_in_progress = False
            return result

        self.get_logger().info(f"Received task goal: '{command}'")

        try:
            image = self.bridge.imgmsg_to_cv2(self.latest_image_msg, desired_encoding="rgb8")
        except Exception as exc:
            goal_handle.abort()
            result = TaskCommand.Result()
            result.success = False
            result.message = f"Image decode error: {exc}"
            with self.state_lock:
                self.active_goal_handle = None
                self.goal_in_progress = False
            return result

        self.get_logger().info(
            f"Querying Gemini Robotics-ER…  instruction='{command}'"
        )
        decision = self.gemini.decide_task(
            image_rgb=image,
            instruction=command,
        )
        if decision is None:
            goal_handle.abort()
            result = TaskCommand.Result()
            result.success = False
            result.message = "Gemini could not determine a valid task."
            with self.state_lock:
                self.active_goal_handle = None
                self.goal_in_progress = False
            return result

        motion_request_id = uuid.uuid4().hex
        self.active_motion_request_id = motion_request_id

        out_msg = MotionPlannerCommand()
        out_msg.request_id = motion_request_id
        out_msg.action = decision.action
        out_msg.point = [float(decision.point[0]), float(decision.point[1])]
        out_msg.ref_image = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
        self.motion_planner_cmd_pub.publish(out_msg)

        goal_feedback = TaskCommand.Feedback()
        goal_feedback.feedback = (
            f"Published MotionPlannerCommand(request_id={motion_request_id}, action={out_msg.action})"
        )
        goal_handle.publish_feedback(goal_feedback)

        self.get_logger().info(
            "Published MotionPlannerCommand "
            f"(request_id={out_msg.request_id}, action={out_msg.action}, "
            f"point=[{out_msg.point[0]:.1f}, {out_msg.point[1]:.1f}], "
            f"ref_image=({out_msg.ref_image.width}x{out_msg.ref_image.height}))"
        )

        while rclpy.ok() and not self.active_goal_done_event.wait(timeout=0.1):
            pass

        with self.state_lock:
            status = self.active_goal_result_status
            message = self.active_goal_result_message
            self.active_goal_handle = None
            self.active_motion_request_id = ""
            self.active_goal_result_status = 0
            self.active_goal_result_message = ""
            self.active_goal_done_event.clear()
            self.goal_in_progress = False

        result = TaskCommand.Result()
        result.success = (status == 1)
        result.message = message
        if status == 1:
            goal_handle.succeed()
        else:
            self.get_logger().error(
                f"Task goal failed from MotionPlannerFeedback status={status}: {message}"
            )
            goal_handle.abort()
        return result

    def motion_feedback_cb(self, msg: MotionPlannerFeedback):
        self.get_logger().info(
            f"MotionPlannerFeedback(request_id={msg.request_id}, status={msg.status}): {msg.message}"
        )
        with self.state_lock:
            if self.active_goal_handle is None:
                return
            if msg.request_id != self.active_motion_request_id:
                return

            feedback = TaskCommand.Feedback()
            feedback.feedback = msg.message
            self.active_goal_handle.publish_feedback(feedback)

            if msg.status not in (1, -1):
                return

            self.active_goal_result_status = int(msg.status)
            self.active_goal_result_message = msg.message
            self.active_goal_done_event.set()

def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    rclpy.spin(node)
    node.task_action_server.destroy()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
