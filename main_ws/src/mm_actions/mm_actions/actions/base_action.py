import time
from typing import Optional

import numpy as np
from spatialmath import SE3, UnitQuaternion
import roboticstoolbox as rtb

from mm_actions.motion.piper_kinematic import servo


class BaseAction:
    DT = 1 / 40.0
    INTEGRATION_DT = 0.01
    POSE_ERROR_THRESHOLD = 0.001

    def __init__(self, get_image, get_joint_state, publish_arm_cmd, image=None, point=None, joint_state_at_image=None) -> None:
        self._get_image = get_image
        self._get_joint_state = get_joint_state
        self._publish_arm_cmd = publish_arm_cmd
        self._image = image
        self._point = point
        self._joint_state_at_image = joint_state_at_image
        self._robot = rtb.models.Piper()
        self._q_calc: Optional[np.ndarray] = None

    def move_arm_to_pose(self, target_pose, timeout_s=20.0):
        """
        Servo the arm to a target pose.

        Args:
            target_pose: [x, y, z, qw, qx, qy, qz].
            gripper_width: Gripper width to command during motion.
            timeout_s: Max time to attempt before aborting.

        Returns:
            (success: bool, message: str)
        """
        t_start = time.time()
        target_pose = np.asarray(target_pose, dtype=float)
        if target_pose.shape != (7,):
            return False, "invalid target_pose shape"

        while time.time() - t_start < timeout_s:
            joint_state = self._get_joint_state()
            if joint_state is None or len(joint_state) < 6:
                return False, "no valid joint state"

            if self._q_calc is None:
                self._q_calc = np.array(joint_state[:6], dtype=float)

            q = self._q_calc
            self._robot.q = q
            wTe = self._robot.fkine(q)
            Tep = SE3.Rt(UnitQuaternion(target_pose[3:]).SO3(), target_pose[:3]).A
            eTep = np.linalg.inv(wTe.A) @ Tep
            et = np.sum(np.abs(eTep[:3, -1]))

            if et < self.POSE_ERROR_THRESHOLD:
                return True, "target reached"

            qd = servo(self._robot, q, wTe, Tep, et)
            if qd is None:
                return False, "QP solver failed"

            self._q_calc = q + qd * self.INTEGRATION_DT
            self._publish_arm_cmd(self._q_calc.tolist(), self._get_joint_state()[-1])
            time.sleep(self.DT)

        return False, "timeout"

    def move_arm_to_joint_state(
        self,
        target_joint_state,
    ):
        """
        Smoothly move the arm from current joint state to a target joint state.

        Args:
            target_joint_state: [q1..q6, gripper_width].
        """
        target_joint_state = np.asarray(target_joint_state, dtype=float)
        if target_joint_state.shape != (7,):
            raise ValueError("target_joint_state must be [q1..q6, gripper_width]")

        start_state = self._get_joint_state()

        q0 = np.array(start_state[:6], dtype=float)
        qT = target_joint_state[:6]
        gripper_width = float(target_joint_state[6])

        # Compute duration from the largest joint delta and a max speed.
        max_speed = 0.6
        max_delta = float(np.max(np.abs(qT - q0)))
        duration_s = max(0.3, max_delta / max_speed)
        steps = max(2, int(duration_s / self.DT))

        for i in range(steps + 1):
            t = i / steps
            # Minimum-jerk time scaling: s(0)=0, s(1)=1 with zero vel/acc at endpoints.
            s = 10 * t**3 - 15 * t**4 + 6 * t**5
            q_cmd = q0 + s * (qT - q0)

            self._q_calc = q_cmd
            self._publish_arm_cmd(q_cmd.tolist(), gripper_width)
            time.sleep(self.DT)
    
    def set_gripper_width(self, gripper_width: float):
        joint_state = self._get_joint_state()
        self._publish_arm_cmd(joint_state[:6], gripper_width)

    def convert_camera_to_base(self, point_camera, arm_joint_position):
        """
        point_camera: [x, y, z] in camera frame
        arm_joint_position: [q1, q2, q3, q4, q5, q6, q7]
        """
        ee_T_cam = np.array([
            [ 0.12045728,  0.99241666,  0.02447911, -0.07102005],
            [-0.99265611,  0.12068956, -0.0082389,   0.02413094],
            [-0.0111308,  -0.0233069,   0.99966639, -0.09727718],
            [ 0.,          0.,          0.,          1.        ]
        ])

        # q[:2] = base_joint_position
        # q[2:8] = arm_joint_position[:6]
        q = arm_joint_position[:6]
        base_T_ee = self._robot.fkine(q, include_base=False).A
        p_cam = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
        p_base = base_T_ee @ ee_T_cam @ p_cam
        return p_base[:3]

    def get_ee_pose(self, arm_joint_position):
        """Return end-effector pose in base frame as [x, y, z, qw, qx, qy, qz]."""
        q = np.array(arm_joint_position[:6], dtype=float)
        T = self._robot.fkine(q, include_base=False)
        quat = UnitQuaternion(T)
        return np.r_[T.t, quat.vec]
