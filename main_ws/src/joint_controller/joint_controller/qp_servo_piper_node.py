# qp_servo_node: ROS2 node for QP-based end-effector velocity control of a mobile manipulator.
# Accepts joint states and target end-effector pose (in current base frame), outputs velocity commands at 40 Hz.
# TODO: Consider supporting servo targets for different link combinations (e.g., multiple end-effectors or mobile base).
# Note: Simultaneous goals for multiple body parts is theoretically possible with whole-body control,
# but requires more complex QP formulation and may involve trade-offs or prioritization.
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Int32
import numpy as np
from spatialmath import SE3, UnitQuaternion
import roboticstoolbox as rtb
import qpsolvers as qp
from enum import Enum

class JointControllerStatus(Enum):
    FAIL = -1
    WORKING = 0
    IDLE = 1
    
class QPServoNode(Node):
    ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    DT = 1/40.0  
    INTEGRATION_DT = 0.01  # Fixed integration timestep
    POSE_ERROR_THRESHOLD = 0.02  # Threshold for considering target reached

    def __init__(self):
        super().__init__('qp_servo_node')

        qos_profile = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.arm_joint_state_sub = self.create_subscription(
            JointState,
            'joint_states_feedback',
            self.arm_joint_state_cb,
            qos_profile
        )
        self.target_pose_sub = self.create_subscription(  #always assume in base frame
            Pose,
            'target_ee_pose',
            self.target_pose_cb,
            qos_profile
        )
        
        self.ee_pose_sub = self.create_subscription(
            Pose,
            'end_pose',
            self.ee_pose_cb,
            qos_profile
        )
        self.arm_pos_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.timer = self.create_timer(self.DT, self.control_loop)
        
        self.status_pub = self.create_publisher(Int32, 'joint_controller_status', 10)

        self.current_arm_joint_position = None
        self.target_pose = None
        self.ee_pose = None 

        self.robot = rtb.models.Piper()  # Use arm-only model
        self.q_calc = None  
        self.target_reached = False
        self.status = JointControllerStatus.IDLE
        
        self.debug = True  
        if self.debug:
            self.q_calc_pub = self.create_publisher(JointState, '/debug/q_calc', 10)
            self.q_meas_pub = self.create_publisher(JointState, '/debug/q_meas', 10)
            self.ee_calc_pub = self.create_publisher(PoseStamped, '/debug/ee_calc', 10)
            self.ee_meas_pub = self.create_publisher(PoseStamped, '/debug/ee_meas', 10)
            
    def arm_joint_state_cb(self, msg):
        self.current_arm_joint_position = msg.position
        # Initialize q_calc with measured q if not set
        if self.q_calc is None and self.current_arm_joint_position is not None:
            self.q_calc = np.array(self.current_arm_joint_position[:6])

    def target_pose_cb(self, msg):
        self.target_pose = [msg.position.x, msg.position.y, msg.position.z , msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        
    def ee_pose_cb(self, msg):
        self.ee_pose = [msg.position.x, msg.position.y, msg.position.z , msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]

    def control_loop(self):
        
        if self.target_pose is None:
            print("No target pose received yet")
            self.status = JointControllerStatus.IDLE
            self.publish_status()
            return 
        if self.current_arm_joint_position is None:
            print("No joint state received yet")
            self.status = JointControllerStatus.FAIL
            self.publish_status()
            return
       
        # Use calculated q if available, else initialize with measured q
        if self.q_calc is None:
            q = np.array(self.current_arm_joint_position[:6])
            self.q_calc = q.copy()
        else:
            q = self.q_calc
            
        self.robot.q = q
        wTe = self.robot.fkine(self.robot.q)
        Tep = SE3.Rt(UnitQuaternion(self.target_pose[3:]).SO3(), self.target_pose[:3]).A
        eTep = np.linalg.inv(wTe.A) @ Tep
        et = np.sum(np.abs(eTep[:3, -1]))
        # If target reached, sync q_calc with measured q and stop sending commands
        if et < self.POSE_ERROR_THRESHOLD:
            self.q_calc = np.array(self.current_arm_joint_position[:6])
            if not self.target_reached:
                self.target_reached = True
            self.status = JointControllerStatus.IDLE
            self.publish_status()
            return
        self.target_reached = False
        self.status = JointControllerStatus.WORKING
        self.publish_status()
        
        Y = 0.01
        Q = np.eye(self.robot.n + 6)
        Q[:self.robot.n, :self.robot.n] *= Y
        Q[self.robot.n:, self.robot.n:] = (1.0 / et) * np.eye(6)
        v, _ = rtb.p_servo(wTe, Tep, 1.5)
        v[3:] *= 0.5
        Aeq = np.c_[self.robot.jacobe(self.robot.q), np.eye(6)]
        beq = v.reshape((6,))
        Ain = np.zeros((self.robot.n + 6, self.robot.n + 6))
        bin = np.zeros(self.robot.n + 6)
        ps = 0.1
        pi = 0.9
        Ain[:self.robot.n, :self.robot.n], bin[:self.robot.n] = self.robot.joint_velocity_damper(ps, pi, self.robot.n)
        c = np.concatenate((np.zeros(self.robot.n), np.zeros(6)))
        
        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[self.robot.qdlim[:self.robot.n], 10 * np.ones(6)]
        ub = np.r_[self.robot.qdlim[:self.robot.n], 10 * np.ones(6)]
        
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='quadprog')
        if qd is None:
            self.get_logger().warn('QP solver failed, not sending commands')
            return
        qd = qd[:self.robot.n]
        if et > 0.5:
            qd *= 0.7 / et
        else:
            qd *= 1.4
        # Integrate velocity to position using fixed dt
        self.q_calc = q + qd * self.INTEGRATION_DT
        self.cmd_arm(self.q_calc.tolist())
        
        if self.debug: 
            self.pub_debug_msg(q, wTe)
                    
    
    def cmd_arm(self, q):
        msg = JointState()
        assert len(q) == 6, "Expected 6 joint commands for the arm."
        msg.name = self.ARM_JOINT_NAMES
        msg.position = q
        msg.header.stamp = self.get_clock().now().to_msg()
        self.arm_pos_cmd_pub.publish(msg)
        
    def publish_status(self):
        msg = Int32()
        msg.data = self.status.value
        self.status_pub.publish(msg)
        
    def pub_debug_msg(self, q, wTe):
        q_calc_msg = JointState()
        q_calc_msg.name = self.ARM_JOINT_NAMES
        q_calc_msg.position = q.tolist()
        q_calc_msg.header.stamp = self.get_clock().now().to_msg()
        self.q_calc_pub.publish(q_calc_msg)

        q_meas_msg = JointState()
        q_meas_msg.name = self.ARM_JOINT_NAMES
        q_meas_msg.position = self.current_arm_joint_position[:6]
        q_meas_msg.header.stamp = self.get_clock().now().to_msg()
        self.q_meas_pub.publish(q_meas_msg)

        ee_calc_msg = PoseStamped()
        ee_calc_msg.header.stamp = self.get_clock().now().to_msg()
        pos = wTe.t
        quat = UnitQuaternion(wTe).vec
        ee_calc_msg.pose.position.x = pos[0]
        ee_calc_msg.pose.position.y = pos[1]
        ee_calc_msg.pose.position.z = pos[2]
        ee_calc_msg.pose.orientation.w = quat[0]
        ee_calc_msg.pose.orientation.x = quat[1]
        ee_calc_msg.pose.orientation.y = quat[2]
        ee_calc_msg.pose.orientation.z = quat[3]
        self.ee_calc_pub.publish(ee_calc_msg)

        if self.ee_pose is not None:
            ee_meas_msg = PoseStamped()
            ee_meas_msg.header.stamp = self.get_clock().now().to_msg()
            ee_meas_msg.pose.position.x = self.ee_pose[0]
            ee_meas_msg.pose.position.y = self.ee_pose[1]
            ee_meas_msg.pose.position.z = self.ee_pose[2]
            ee_meas_msg.pose.orientation.w = self.ee_pose[3]
            ee_meas_msg.pose.orientation.x = self.ee_pose[4]
            ee_meas_msg.pose.orientation.y = self.ee_pose[5]
            ee_meas_msg.pose.orientation.z = self.ee_pose[6]
            self.ee_meas_pub.publish(ee_meas_msg)

def main(args=None):
    rclpy.init(args=args)
    node = QPServoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
