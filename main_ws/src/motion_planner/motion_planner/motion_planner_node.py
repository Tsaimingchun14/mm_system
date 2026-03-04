import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, JointState
import message_filters
import numpy as np
from cv_bridge import CvBridge
from mm_interface.msg import (
    JointPlannerCommand,
    JointPlannerFeedback,
    MotionPlannerCommand,
    MotionPlannerFeedback,
)
from .perception.perception import PerceptionModel
from .action_types import parse_action
import roboticstoolbox as rtb

class MotionPlannerNode(Node):
    DT = 1/5.0

    def __init__(self):
        super().__init__('motion_planner_node')

        # Subscriptions
        self.motion_command_sub = self.create_subscription(
            MotionPlannerCommand,
            'motion_planner_command',
            self.motion_planner_command_cb,
            10
        )
        self.joint_feedback_sub = self.create_subscription(
            JointPlannerFeedback,
            'joint_planner_feedback',
            self.joint_planner_feedback_cb,
            10
        )

        # Publications
        self.joint_command_pub = self.create_publisher(JointPlannerCommand, 'joint_planner_command', 10)
        self.motion_feedback_pub = self.create_publisher(MotionPlannerFeedback, 'motion_planner_feedback', 10)

        # Timer
        self.timer = self.create_timer(self.DT, self.control_loop)

        # Runtime task state
        self.current_task = None
        self.current_request_id = ""
        self.joint_planner_status = JointPlannerFeedback.WORKING

        # Synced perception + robot state
        self.sync_image = None
        self.sync_depth = None
        self.sync_intrinsics = None
        self.sync_arm_joint_position = None

        # Perception subscriptions
        self.image_sub = message_filters.Subscriber(self, Image, 'image_raw', 1)
        self.depth_sub = message_filters.Subscriber(self, Image, 'depth_image', 1)
        self.depth_info_sub = message_filters.Subscriber(self, CameraInfo, "depth_info", 1)
        self.joint_state_sub = message_filters.Subscriber(
            self, JointState, 'joint_states_feedback', 1
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.depth_info_sub, self.joint_state_sub],
            queue_size=30,
            slop=0.05
        )
        self.ts.registerCallback(self.synced_image_cb)

        # Models / utilities
        self.bridge = CvBridge()
        self.perception_model = PerceptionModel()
        # self.robot = rtb.models.KachakaPiper()
        # self.current_base_joint_position = [0, 0]
        self.robot = rtb.models.Piper()

    def motion_planner_command_cb(self, msg: MotionPlannerCommand):
        # Convert incoming command msg to the action parser input schema.
        ref_image = None
        if msg.ref_image.height > 0 and msg.ref_image.width > 0:
            try:
                ref_image = self.bridge.imgmsg_to_cv2(msg.ref_image, desired_encoding="rgb8")
            except Exception as exc:
                self.get_logger().warn(f"Failed to decode ref_image from MotionPlannerCommand: {exc}")

        d = {
            "action": msg.action,
            "point": [float(msg.point[0]), float(msg.point[1])],
            "ref_image": ref_image,
        }
        action_obj = parse_action(d)
        self.current_task = action_obj
        self.current_request_id = msg.request_id
        self.publish_motion_feedback(0, f"Accepted action '{msg.action}'.")
        self.get_logger().info(f"Received task: {action_obj}")

    def control_loop(self):
        # No active task: nothing to do.
        if self.current_task is None:
            return

        # Wait until synchronized perception + joint state are available.
        if (
            self.sync_image is None
            or self.sync_depth is None
            or self.sync_intrinsics is None
            or self.sync_arm_joint_position is None
        ):
            return

        synced_snapshot = {
            'image': self.sync_image,
            'depth': self.sync_depth,
            'intrinsics': self.sync_intrinsics,
            'joints': self.sync_arm_joint_position
        }

        # Let current action advance one step using the synced snapshot.
        self.current_task.step(self, synced_snapshot)

        # Emit terminal feedback and clear task state once done or failed.
        if self.current_task.is_completed():
            self.get_logger().info(f"Action {self.current_task.action} completed.")
            self.publish_motion_feedback(1, f"Action '{self.current_task.action}' completed.")
            self.current_task = None
            self.current_request_id = ""
        elif self.current_task.failed():
            self.get_logger().error(f"Action {self.current_task.action} failed.")
            self.publish_motion_feedback(-1, f"Action '{self.current_task.action}' failed.")
            self.current_task = None
            self.current_request_id = ""

    def synced_image_cb(self, img_msg, depth_msg, depth_info_msg, joint_msg):
        # Keep the most recent synchronized RGB-D + intrinsics + arm joints.
        self.sync_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        self.sync_depth = self.bridge.imgmsg_to_cv2(depth_msg)
        self.sync_intrinsics = {
            'fx': depth_info_msg.k[0],
            'fy': depth_info_msg.k[4],
            'cx': depth_info_msg.k[2],
            'cy': depth_info_msg.k[5],
        }
        self.sync_arm_joint_position = joint_msg.position

    def joint_planner_feedback_cb(self, msg: JointPlannerFeedback):
        self.joint_planner_status = msg.status

    def publish_ee_goal(self, pose_list):
        """
        pose_list: [x, y, z, qw, qx, qy, qz]
        """
        if not isinstance(pose_list, (list, tuple)) or len(pose_list) != 7:
            self.get_logger().error("publish_ee_goal expects a list of 7 elements: [x, y, z, qw, qx, qy, qz]")
            return
        msg = JointPlannerCommand()
        msg.request_id = self.current_request_id
        msg.target_pose.position.x = pose_list[0]
        msg.target_pose.position.y = pose_list[1]
        msg.target_pose.position.z = pose_list[2]
        msg.target_pose.orientation.w = pose_list[3]
        msg.target_pose.orientation.x = pose_list[4]
        msg.target_pose.orientation.y = pose_list[5]
        msg.target_pose.orientation.z = pose_list[6]
        msg.gripper_width = float(np.nan)
        self.joint_command_pub.publish(msg)

    def publish_gripper_goal(self, width: float):
        # Clamp width to the valid gripper command range.
        msg = JointPlannerCommand()
        msg.request_id = self.current_request_id
        msg.target_pose.position.x = float(np.nan)
        msg.target_pose.position.y = float(np.nan)
        msg.target_pose.position.z = float(np.nan)
        msg.target_pose.orientation.w = float(np.nan)
        msg.target_pose.orientation.x = float(np.nan)
        msg.target_pose.orientation.y = float(np.nan)
        msg.target_pose.orientation.z = float(np.nan)
        msg.gripper_width = float(max(0.0, min(0.08, width)))
        self.joint_command_pub.publish(msg)

    def update_perception_prompt(self, point, image):
        frame = image if image is not None else self.sync_image
        if frame is None:
            self.get_logger().warn("No image available for perception prompt update.")
            return
        self.perception_model.update_point_prompt(frame, [point])

    def get_object_center_3d_camera(self, rgb_image, depth_image, intrinsics):
        # The new perception_model.process_frame now returns (x, y, z) directly
        center_3d = self.perception_model.process_frame(rgb_image, depth_image, intrinsics)
        return center_3d

    def convert_camera_to_base(self, point_camera, arm_joint_position):
        """
        point_camera: [x, y, z] in camera frame
        arm_joint_position: [q1, q2, q3, q4, q5, q6, q7]
        """
        camera_in_ee = np.array([
            [ 0.12045728,  0.99241666,  0.02447911, -0.07102005],
            [-0.99265611,  0.12068956, -0.0082389,   0.02413094],
            [-0.0111308,  -0.0233069,   0.99966639, -0.09727718],
            [ 0.,          0.,          0.,          1.        ]
        ])

        # q[:2] = base_joint_position
        # q[2:8] = arm_joint_position[:6]
        q = arm_joint_position[:6]
        ee_in_base = self.robot.fkine(q, include_base=False).A
        p_cam = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
        p_base = ee_in_base @ camera_in_ee @ p_cam
        return p_base[:3]

    def publish_motion_feedback(self, status: int, message: str):
        feedback = MotionPlannerFeedback()
        feedback.request_id = self.current_request_id
        feedback.status = status
        feedback.message = message
        self.motion_feedback_pub.publish(feedback)

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
