import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Int32
import message_filters
import numpy as np
from cv_bridge import CvBridge
from .perception.perception import PerceptionModel
from .action_types import parse_action
import json
import roboticstoolbox as rtb
from .controller_status import JointControllerStatus

class MotionPlannerNode(Node):
        
    DT = 1/10.0

    def __init__(self):
        super().__init__('motion_planner_node')
        self.ee_pose_pub = self.create_publisher(Pose, 'target_ee_pose', 10)
        self.task_sub = self.create_subscription(String, 'motion_task', self._task_cb, 10)
        self.timer = self.create_timer(self.DT, self._control_loop)
        self.current_task = None  
        
        self.latest_image = None
        self.latest_depth = None
        self.intrinsics = None
        self.image_sub = message_filters.Subscriber(self, Image, 'image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, 'depth_image')
        self.depth_info_sub = message_filters.Subscriber( self, CameraInfo, "depth_info")
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.depth_info_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self._synced_image_cb)

        self.bridge = CvBridge()
        self.perception_model = PerceptionModel()
        
        qos_profile = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.arm_joint_state_sub = self.create_subscription(
            JointState,
            'joint_states_feedback',
            self._arm_joint_state_cb,
            qos_profile
        )
        self.joint_controller_status_sub = self.create_subscription(
            Int32, 
            'joint_controller_status', 
            self._jc_status_cb,
            qos_profile
        )
        self.joint_controller_status = JointControllerStatus.WORKING
        # self.robot = rtb.models.KachakaPiper()
        # self.current_base_joint_position = [0, 0]
        self.robot = rtb.models.Piper()
        self.current_arm_joint_position = None

    def _task_cb(self, msg):
        d = json.loads(msg.data)
        action_obj = parse_action(d)
        self.current_task = action_obj
        self.get_logger().info(f"Received task: {action_obj}")

    def _control_loop(self):
        if self.current_task is None:
            return

        if self.latest_image is None or self.latest_depth is None or self.intrinsics is None:
            return
        
        self.current_task.step(self)
        
        if self.current_task.is_completed():
            self.get_logger().info(f"Action {self.current_task.action} completed.")
            self.current_task = None
        elif self.current_task.failed():
            self.get_logger().error(f"Action {self.current_task.action} failed.")
            self.current_task = None

    def _synced_image_cb(self, img_msg, depth_msg, depth_info_msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(img_msg)
        self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg)
        self.intrinsics = {
            'fx': depth_info_msg.k[0],
            'fy': depth_info_msg.k[4],
            'cx': depth_info_msg.k[2],
            'cy': depth_info_msg.k[5],
        }
        
    def _arm_joint_state_cb(self, msg):
        self.current_arm_joint_position = msg.position
        
    def _jc_status_cb(self, msg):
        self.joint_controller_status = JointControllerStatus(msg.data)
        
    def publish_ee_pose(self, pose_list):
        """
        pose_list: [x, y, z, qw, qx, qy, qz]
        """
        if not isinstance(pose_list, (list, tuple)) or len(pose_list) != 7:
            self.get_logger().error("publish_ee_pose expects a list of 7 elements: [x, y, z, qw, qx, qy, qz]")
            return
        pose_msg = Pose()
        pose_msg.position.x = pose_list[0]
        pose_msg.position.y = pose_list[1]
        pose_msg.position.z = pose_list[2]
        pose_msg.orientation.w = pose_list[3]
        pose_msg.orientation.x = pose_list[4]
        pose_msg.orientation.y = pose_list[5]
        pose_msg.orientation.z = pose_list[6]
        self.ee_pose_pub.publish(pose_msg)
        
    def get_object_center_3d_camera(self):
        mask = self.perception_model.process_frame(self.latest_image)
        center_3d = self.perception_model.calculate_object_center_3d(mask, self.latest_depth, self.intrinsics)
        return center_3d
    
    def convert_camera_to_base(self, point_camera):
        """
        point_camera: [x, y, z] in camera frame
        """
        camera_in_ee = np.array([
            [ 0.12045728,  0.99241666,  0.02447911, -0.07102005],
            [-0.99265611,  0.12068956, -0.0082389,   0.02413094],
            [-0.0111308,  -0.0233069,   0.99966639, -0.09727718],
            [ 0.,          0.,          0.,          1.        ]
        ])
        
        if self.current_arm_joint_position is None:
            self.get_logger().error("No arm joint state for FK")
            return None
        q = np.zeros(self.robot.n)
        # q[:2] = self.current_base_joint_position
        # q[2:8] = self.current_arm_joint_position[:6]
        q = self.current_arm_joint_position[:6]
        ee_in_base = self.robot.fkine(q, include_base=False).A
        p_cam = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
        p_base = ee_in_base @ camera_in_ee @ p_cam
        return p_base[:3]
    
    def update_perception_prompt(self, point_prompt):
        self.perception_model.update_point_prompt(self.latest_image, point_prompt)

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
