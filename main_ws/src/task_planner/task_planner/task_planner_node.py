"""
task_planner_node.py
ROS 2 node that bridges natural-language commands to motion_planner tasks.

Topic interface
---------------
Subscriptions
  /task_command  (std_msgs/String)
      Natural-language operator command, e.g. "grasp the red cup" or
      "hand the bottle to the person on the right".

  /image_raw     (sensor_msgs/Image)
      Live RGB camera stream. The latest frame is used when a command arrives.

Publications
  /motion_task   (std_msgs/String)
      JSON action payload for motion_planner, e.g.
        {"action": "grasp", "point": [423.5, 317.2]}

Parameters (ROS 2 node params)
------------------------------
  gemini_api_key      (string, required unless GOOGLE_API_KEY env var is set)
  gemini_model        (string, default: "gemini-robotics-er-1.5-preview")
  command_topic       (string, default: "/task_command")
  image_topic         (string, default: "/image_raw")
  motion_task_topic   (string, default: "/motion_task")

Flow
----
1.  Operator publishes a string on /task_command.
2.  Node captures the latest /image_raw frame.
3.  Calls GeminiRoboticsClient.decide_task(image, command).
    Gemini decides:
      a) Which ACTION to execute (grasp / hand_over / …).
      b) WHERE the target object is (2D pixel grasp centre).
4.  Publishes {"action": "...", "point": [x, y]} to /motion_task.
"""

import json
import os
import threading

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from .gemini_client import GeminiRoboticsClient


class TaskPlannerNode(Node):

    def __init__(self):
        super().__init__("task_planner_node")

        # ---- Parameters ------------------------------------------------
        self.declare_parameter("gemini_api_key", "")
        self.declare_parameter("gemini_model", GeminiRoboticsClient.MODEL)
        self.declare_parameter("command_topic", "/task_command")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("motion_task_topic", "/motion_task")

        api_key = self.get_parameter("gemini_api_key").value
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            self.get_logger().error(
                "No Gemini API key found. "
                "Set ROS param 'gemini_api_key' or env var GOOGLE_API_KEY."
            )
            raise RuntimeError("Missing Gemini API key.")

        model             = self.get_parameter("gemini_model").value
        command_topic     = self.get_parameter("command_topic").value
        image_topic       = self.get_parameter("image_topic").value
        motion_task_topic = self.get_parameter("motion_task_topic").value

        # ---- Gemini client ---------------------------------------------
        self._gemini = GeminiRoboticsClient(api_key=api_key, model=model)

        # ---- ROS interfaces --------------------------------------------
        self._bridge = CvBridge()
        self._latest_image = None          # numpy RGB uint8
        self._image_lock   = threading.Lock()
        self._busy         = False         # prevent overlapping Gemini calls

        self._image_sub = self.create_subscription(
            Image, image_topic, self._image_cb, 10
        )
        self._command_sub = self.create_subscription(
            String, command_topic, self._command_cb, 10
        )
        self._motion_task_pub = self.create_publisher(
            String, motion_task_topic, 10
        )

        self.get_logger().info(
            f"TaskPlannerNode ready.\n"
            f"  command_topic     : {command_topic}\n"
            f"  image_topic       : {image_topic}\n"
            f"  motion_task_topic : {motion_task_topic}\n"
            f"  gemini_model      : {model}"
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image):
        """Cache the most recent RGB frame (non-blocking)."""
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._image_lock:
                self._latest_image = rgb
        except Exception as exc:
            self.get_logger().warn(f"Image decode error: {exc}")

    def _command_cb(self, msg: String):
        """
        Handle an incoming natural-language command.

        Spawns a background thread for the Gemini API call so the ROS
        executor thread is never blocked.
        """
        command = msg.data.strip()
        if not command:
            return

        if self._busy:
            self.get_logger().warn(
                "Still processing a previous command — new command ignored. "
                f"(Dropped: '{command}')"
            )
            return

        self.get_logger().info(f"Received command: '{command}'")

        with self._image_lock:
            image = (
                self._latest_image.copy()
                if self._latest_image is not None
                else None
            )

        if image is None:
            self.get_logger().error(
                "No camera image available. "
                "Ensure the camera is publishing on the image topic."
            )
            return

        self._busy = True
        thread = threading.Thread(
            target=self._query_and_publish,
            args=(command, image),
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------
    # Core logic  (runs in background thread)
    # ------------------------------------------------------------------

    def _query_and_publish(self, command: str, image):
        try:
            self.get_logger().info(
                f"Querying Gemini Robotics-ER…  instruction='{command}'"
            )

            decision = self._gemini.decide_task(
                image_rgb=image,
                instruction=command,
            )

            if decision is None:
                self.get_logger().error(
                    f"Gemini could not determine a valid task for: '{command}'. "
                    "The target may not be visible or the instruction is ambiguous."
                )
                return

            self.get_logger().info(
                f"Gemini decision → action='{decision.action}', "
                f"target='{decision.label}', "
                f"pixel=({decision.point[0]:.1f}, {decision.point[1]:.1f})"
            )

            # Build the JSON payload expected by motion_planner.
            payload = {
                "action": decision.action,
                "point":  [float(decision.point[0]), float(decision.point[1])],
            }
            out_msg = String()
            out_msg.data = json.dumps(payload)
            self._motion_task_pub.publish(out_msg)

            self.get_logger().info(f"Published → /motion_task: {out_msg.data}")

        finally:
            self._busy = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    try:
        node = TaskPlannerNode()
        rclpy.spin(node)
    except RuntimeError as exc:
        print(f"[task_planner] Fatal: {exc}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
