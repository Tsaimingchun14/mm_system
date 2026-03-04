import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
import numpy as np
from spatialmath import SE3, UnitQuaternion
import roboticstoolbox as rtb
import qpsolvers as qp
from mm_interface.msg import JointPlannerCommand, JointPlannerFeedback

class QPServoNode(Node):
    ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    DT = 1/40.0  
    INTEGRATION_DT = 0.01  # Fixed integration timestep
    POSE_ERROR_THRESHOLD = 0.02  # Threshold for considering target reached

    def __init__(self):
        super().__init__('qp_servo_node')

        # Subscriptions
        # recieve command from motion planner, target ee pose always in base frame
        self.command_sub = self.create_subscription( 
            JointPlannerCommand,
            'joint_planner_command',
            self.joint_planner_command_cb,
            1
        )
        # recieve joint state from robot driver
        self.arm_joint_state_sub = self.create_subscription(
            JointState,
            'joint_states_feedback',
            self.arm_joint_state_cb,
            1
        )
        
        # Publications
        # publish feedback to motion planner
        self.feedback_pub = self.create_publisher(JointPlannerFeedback, 'joint_planner_feedback', 10)
        # publish joint commands to robot driver
        self.arm_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)

        self.timer = self.create_timer(self.DT, self.control_loop)

        self.goal_pose = None
        self.goal_gripper_width = 0.0
        self.goal_request_id = ""
        
        self.current_arm_joint_position = None
        self.current_ee_pose = None 
        
        self.robot = rtb.models.Piper()  # Use arm-only model
        self.q_calc = None  
        self.target_reached = False
        self.status = JointPlannerFeedback.IDLE
        
        self.debug = False 
        if self.debug:
            self.ee_pose_sub = self.create_subscription(Pose, 'end_pose', self.ee_pose_cb, 1)
            self.q_calc_pub = self.create_publisher(JointState, '/debug/q_calc', 10)
            self.q_meas_pub = self.create_publisher(JointState, '/debug/q_meas', 10)
            self.ee_calc_pub = self.create_publisher(PoseStamped, '/debug/ee_calc', 10)
            self.ee_meas_pub = self.create_publisher(PoseStamped, '/debug/ee_meas', 10)

    def joint_planner_command_cb(self, msg: JointPlannerCommand):
        # Use NaN as "not provided" sentinel for partial updates.
        pose = msg.target_pose
        pose_values = [ pose.position.x, pose.position.y, pose.position.z, 
                       pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        if np.all(np.isfinite(pose_values)):
            self.goal_pose = pose_values

        if np.isfinite(msg.gripper_width):
            self.goal_gripper_width = msg.gripper_width
        self.goal_request_id = msg.request_id
        
    def arm_joint_state_cb(self, msg):
        self.current_arm_joint_position = msg.position
        # Initialize q_calc with measured q if not set
        if self.q_calc is None and self.current_arm_joint_position is not None:
            self.q_calc = np.array(self.current_arm_joint_position[:6])

    def control_loop(self):
        
        if self.goal_pose is None:
            print("No target pose received yet")
            self.status = JointPlannerFeedback.IDLE
            self.publish_status()
            return 
        if self.current_arm_joint_position is None:
            print("No joint state received yet")
            self.status = JointPlannerFeedback.FAIL
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
        Tep = SE3.Rt(UnitQuaternion(self.goal_pose[3:]).SO3(), self.goal_pose[:3]).A
        eTep = np.linalg.inv(wTe.A) @ Tep
        et = np.sum(np.abs(eTep[:3, -1]))
        # If target reached, sync q_calc with measured q and stop sending commands
        if et < self.POSE_ERROR_THRESHOLD:
            self.q_calc = np.array(self.current_arm_joint_position[:6])
            if not self.target_reached:
                self.target_reached = True
            self.status = JointPlannerFeedback.IDLE
            self.publish_status()
            self.cmd_arm(self.q_calc.tolist())
            return
        self.target_reached = False
        self.status = JointPlannerFeedback.WORKING
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
        assert len(q) == 6, f"Expected 6 joint commands for the arm, got {len(q)}"
        msg.name = self.ARM_JOINT_NAMES
        msg.position = q + [self.goal_gripper_width]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.arm_pos_cmd_pub.publish(msg)
        
    def publish_status(self):
        feedback = JointPlannerFeedback()
        feedback.request_id = self.goal_request_id
        feedback.status = self.status
        self.feedback_pub.publish(feedback)

    # Below are for debuggin only
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
            
    def ee_pose_cb(self, msg):
        self.current_ee_pose = [msg.position.x, msg.position.y, msg.position.z, 
                        msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]

def main(args=None):
    rclpy.init(args=args)
    node = QPServoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
