# qp_servo_node: ROS2 node for QP-based end-effector velocity control of a mobile manipulator.
# Accepts joint states and target end-effector pose (in current EE frame), outputs velocity commands at 40 Hz.
# TODO: Consider supporting servo targets for different link combinations (e.g., multiple end-effectors or mobile base).
# Note: Simultaneous goals for multiple body parts is theoretically possible with whole-body control,
# but requires more complex QP formulation and may involve trade-offs or prioritization.
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped, Twist
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
        self.base_vel_cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.arm_pos_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.timer = self.create_timer(self.DT, self.control_loop)

        self.status_pub = self.create_publisher(Int32, 'joint_controller_status', 10)

        self.current_base_joint_position = [0, 0]  # virtual joints for mobile base
        self.current_arm_joint_position = None
        self.target_pose = None 
        self.ee_pose = None 

        self.robot = rtb.models.KachakaPiper()
        self.q_calc = None  
        self.target_reached = False
        self.status = JointControllerStatus.IDLE

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
            self.cmd_base(0, 0)
            print("No target pose received yet")
            self.status = JointControllerStatus.IDLE
            self.publish_status()
            return 
        if self.current_arm_joint_position is None:
            self.cmd_base(0, 0)
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
        
        self.robot.q = np.r_[self.current_base_joint_position, q]
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
        Q[: self.robot.n, : self.robot.n] *= Y
        Q[:2, :2] *= 1.0 / et
        Q[self.robot.n :, self.robot.n :] = (1.0 / et) * np.eye(6)
        v, _ = rtb.p_servo(wTe, Tep, 1.5)
        # v[3:] *= 0.5
        Aeq = np.c_[self.robot.jacobe(self.robot.q), np.eye(6)]
        beq = v.reshape((6,))
        Ain = np.zeros((self.robot.n + 6, self.robot.n + 6))
        bin = np.zeros(self.robot.n + 6)
        ps = 0.1
        pi = 0.9
        Ain[: self.robot.n, : self.robot.n], bin[: self.robot.n] = self.robot.joint_velocity_damper(ps, pi, self.robot.n)
        #different from arm only setup
        c = np.concatenate((np.zeros(2), -self.robot.jacobm(start=self.robot.links[4]).reshape((self.robot.n - 2,)), np.zeros(6)))
        
        #Get base to face end-effector
        k_epsilon = 0.5
        bTe = self.robot.fkine(self.robot.q, include_base=False).A
        theta_epsilon = np.arctan2(bTe[1, -1], bTe[0, -1])   # oringianlly using math.atan2
        epsilon = k_epsilon * theta_epsilon
        c[0] = -epsilon
        
        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[self.robot.qdlim[: self.robot.n], 10 * np.ones(6)]
        ub = np.r_[self.robot.qdlim[: self.robot.n], 10 * np.ones(6)]
        
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='quadprog')
        if qd is None:
            self.get_logger().warn('QP solver failed, sending zero commands to base and no arm movement')
            self.cmd_base(0, 0)
            return
        qd = qd[:self.robot.n]
        if et > 0.5:
            qd *= 0.7 / et
        else:
            qd *= 1.4

        # Integrate velocity to position using fixed dt
        self.q_calc = q + qd[2:] * self.INTEGRATION_DT
        base_v = qd[1]
        base_w = qd[0]
        self.cmd_base(base_v, base_w)
        self.cmd_arm(self.q_calc.tolist())
        
    def cmd_base(self, v, w):
          twist = Twist()
          twist.linear.x = v
          twist.angular.z = w
          self.base_vel_cmd_pub.publish(twist)
    
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

def main(args=None):
    rclpy.init(args=args)
    node = QPServoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
