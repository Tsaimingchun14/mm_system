import time
from typing import Callable, Optional

import numpy as np
from spatialmath import SE3, UnitQuaternion
import roboticstoolbox as rtb

from mm_actions.motion.piper_kinematic import (
    find_reachable_pose,
    piper_point_to_kachaka,
    servo,
    whole_body_servo,
)


class BaseAction:
    DT = 0.01
    POSE_ERROR_THRESHOLD = 0.01
    WAYPOINT_HEIGHT_M = 0.10
    WAYPOINT_APPROACH_M = 0.05
    WAYPOINT_POSE_ERROR_THRESHOLD = 0.03

    def __init__(
        self,
        get_image,
        get_joint_state,
        publish_arm_cmd,
        is_holding_tightly,
        use_force_grasp,
        image=None,
        point=None,
        joint_state_at_image=None,
        publish_base_cmd: Optional[Callable[[float, float], None]] = None,
        get_base_pose: Optional[Callable[[], Optional[np.ndarray]]] = None,
        whole_body: bool = False,
    ) -> None:
        self._get_image = get_image
        self._get_joint_state = get_joint_state
        self._publish_arm_cmd = publish_arm_cmd
        self._is_holding_tightly = is_holding_tightly
        self._use_force_grasp = use_force_grasp
        self._publish_base_cmd = publish_base_cmd
        self._get_base_pose = get_base_pose
        self._whole_body = bool(whole_body)
        self._image = image
        self._point = point
        self._joint_state_at_image = joint_state_at_image
        self._arm_robot = rtb.models.Piper()
        self._whole_body_robot = rtb.models.KachakaPiper() if self._whole_body else None
        self._robot = self._arm_robot
        self._q_calc: Optional[np.ndarray] = None
        self._base_pose_at_image = self._read_base_pose() if self._whole_body else None

    def _read_base_pose(self):
        if self._get_base_pose is None:
            return None
        pose = self._get_base_pose()
        if pose is None:
            return None
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,):
            return None
        return pose

    @staticmethod
    def _planar_pose_matrix(base_pose):
        x, y, yaw = np.asarray(base_pose, dtype=float)
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[:3, :3] = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        T[:3, 3] = [x, y, 0.0]
        return T

    @classmethod
    def _update_target_pose_from_base_motion(cls, target_pose_at_inspection, base_pose_at_inspection, base_pose_now):
        T_odom_base_inspect = cls._planar_pose_matrix(base_pose_at_inspection)
        T_odom_base_now = cls._planar_pose_matrix(base_pose_now)
        T_target_inspect = SE3.Rt(
            UnitQuaternion(target_pose_at_inspection[3:]).SO3(),
            target_pose_at_inspection[:3],
        ).A
        return np.linalg.inv(T_odom_base_now) @ T_odom_base_inspect @ T_target_inspect

    def find_target_pose(self, point_piper_base, arm_q):
        """Find a reachable target pose for the active arm-only/whole-body mode."""
        arm_q = np.asarray(arm_q[:6], dtype=float)
        if self._whole_body:
            if self._whole_body_robot is None:
                return None
            point_kachaka = piper_point_to_kachaka(point_piper_base)
            return find_reachable_pose(self._whole_body_robot, arm_q, point_kachaka)
        return find_reachable_pose(self._arm_robot, arm_q, point_piper_base)

    def move_to_points(self, points_piper_base, arm_q, timeout_s=20.0):
        """Follow one point or a sequence of points in Piper base frame."""
        points = np.asarray(points_piper_base, dtype=float)
        if points.shape == (3,):
            points = points.reshape(1, 3)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            return False, "invalid point sequence"

        ik_seed = np.asarray(arm_q[:6], dtype=float)
        planned_poses = []
        for index, point in enumerate(points):
            is_final = index == len(points) - 1
            target_pose = self.find_target_pose(point, ik_seed)
            if target_pose is None:
                label = "target point" if is_final else f"waypoint {index + 1}"
                return False, f"IK failed for {label}"
            planned_poses.append(target_pose)
            if self._q_calc is not None:
                ik_seed = self._q_calc

        for index, target_pose in enumerate(planned_poses):
            is_final = index == len(planned_poses) - 1
            success, message = self.move_to_pose(
                target_pose,
                timeout_s=timeout_s,
                position_tolerance=None if is_final else self.WAYPOINT_POSE_ERROR_THRESHOLD,
            )
            if not success:
                label = "target point" if is_final else f"waypoint {index + 1}"
                return False, f"{label} failed: {message}"

        return True, "target reached"

    def make_intermediate_point(self, target_point_piper_base):
        """Return a waypoint 10 cm up and 5 cm closer to the robot base."""
        target_point = np.asarray(target_point_piper_base, dtype=float)
        waypoint = target_point.copy()
        xy_norm = float(np.linalg.norm(waypoint[:2]))
        if xy_norm > 1e-6:
            waypoint[:2] -= self.WAYPOINT_APPROACH_M * waypoint[:2] / xy_norm
        waypoint[2] += self.WAYPOINT_HEIGHT_M
        return waypoint

    def move_to_point_with_waypoint(self, target_point_piper_base, arm_q, timeout_s=20.0):
        target_point_piper_base = np.asarray(target_point_piper_base, dtype=float)
        waypoint_point = self.make_intermediate_point(target_point_piper_base)
        return self.move_to_points([waypoint_point, target_point_piper_base], arm_q, timeout_s=timeout_s)

    def move_to_pose(self, target_pose, timeout_s=20.0, position_tolerance=None):
        if self._whole_body:
            return self.move_whole_body_to_pose(
                target_pose,
                timeout_s=timeout_s,
                position_tolerance=position_tolerance,
            )
        return self.move_arm_to_pose(
            target_pose,
            timeout_s=timeout_s,
            position_tolerance=position_tolerance,
        )

    def move_arm_to_pose(self, target_pose, timeout_s=20.0, position_tolerance=None):
        """Servo the arm to a target pose."""
        t_start = time.time()
        tolerance = self.POSE_ERROR_THRESHOLD if position_tolerance is None else float(position_tolerance)
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
            self._arm_robot.q = q
            wTe = self._arm_robot.fkine(q)
            Tep = SE3.Rt(UnitQuaternion(target_pose[3:]).SO3(), target_pose[:3]).A
            eTep = np.linalg.inv(wTe.A) @ Tep
            et = np.sum(np.abs(eTep[:3, -1]))

            if et < tolerance:
                return True, "target reached"

            qd = servo(self._arm_robot, q, wTe, Tep, et)
            if qd is None:
                return False, "QP solver failed"

            self._q_calc = q + qd * self.DT
            self._publish_arm_cmd(self._q_calc.tolist(), self._get_joint_state()[-1])
            time.sleep(self.DT)

        return False, "timeout"

    def move_whole_body_to_pose(self, target_pose_at_inspection, timeout_s=20.0, position_tolerance=None):
        """Servo Kachaka yaw/forward and Piper arm toward a target pose."""
        if self._whole_body_robot is None:
            return False, "whole-body robot is not enabled"
        if self._publish_base_cmd is None:
            return False, "no base command publisher available"

        tolerance = self.POSE_ERROR_THRESHOLD if position_tolerance is None else float(position_tolerance)
        target_pose_at_inspection = np.asarray(target_pose_at_inspection, dtype=float)
        if target_pose_at_inspection.shape != (7,):
            return False, "invalid target_pose shape"

        base_pose_at_inspection = self._base_pose_at_image
        if base_pose_at_inspection is None:
            base_pose_at_inspection = self._read_base_pose()
        if base_pose_at_inspection is None:
            return False, "no base pose available"

        t_start = time.time()
        try:
            while time.time() - t_start < timeout_s:
                joint_state = self._get_joint_state()
                if joint_state is None or len(joint_state) < 6:
                    return False, "no valid joint state"

                base_pose_now = self._read_base_pose()
                if base_pose_now is None:
                    return False, "no base pose available"

                if self._q_calc is None:
                    self._q_calc = np.array(joint_state[:6], dtype=float)

                arm_q = self._q_calc
                q = np.r_[0.0, 0.0, arm_q]
                self._whole_body_robot.q = q
                wTe = self._whole_body_robot.fkine(q)
                Tep = self._update_target_pose_from_base_motion(
                    target_pose_at_inspection,
                    base_pose_at_inspection,
                    base_pose_now,
                )
                eTep = np.linalg.inv(wTe.A) @ Tep
                et = np.sum(np.abs(eTep[:3, -1]))

                if et < tolerance:
                    self._publish_base_cmd(0.0, 0.0)
                    return True, "target reached"

                qd = whole_body_servo(self._whole_body_robot, q, wTe, Tep, et)
                if qd is None:
                    self._publish_base_cmd(0.0, 0.0)
                    return False, "QP solver failed"

                yaw_rate = float(np.clip(qd[0], -0.6, 0.6))
                forward_velocity = float(np.clip(qd[1], -0.25, 0.25))
                self._q_calc = arm_q + qd[2:] * self.DT
                self._publish_base_cmd(yaw_rate, forward_velocity)
                self._publish_arm_cmd(self._q_calc.tolist(), self._get_joint_state()[-1])
                time.sleep(self.DT)
        finally:
            self._publish_base_cmd(0.0, 0.0)

        return False, "timeout"

    def move_arm_to_joint_state(self, target_joint_state):
        """Smoothly move the arm from current joint state to a target joint state."""
        target_joint_state = np.asarray(target_joint_state, dtype=float)
        if target_joint_state.shape != (7,):
            raise ValueError("target_joint_state must be [q1..q6, gripper_width]")

        start_state = self._get_joint_state()

        q0 = np.array(start_state[:6], dtype=float)
        qT = target_joint_state[:6]
        gripper_width = float(target_joint_state[6])

        max_speed = 0.6
        max_delta = float(np.max(np.abs(qT - q0)))
        duration_s = max(0.3, max_delta / max_speed)
        steps = max(2, int(duration_s / self.DT))

        for i in range(steps + 1):
            t = i / steps
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

        q = arm_joint_position[:6]
        base_T_ee = self._arm_robot.fkine(q, include_base=False).A
        p_cam = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
        p_base = base_T_ee @ ee_T_cam @ p_cam
        return p_base[:3]

    def get_ee_pose(self, arm_joint_position):
        """Return end-effector pose in base frame as [x, y, z, qw, qx, qy, qz]."""
        q = np.array(arm_joint_position[:6], dtype=float)
        T = self._arm_robot.fkine(q, include_base=False)
        quat = UnitQuaternion(T)
        return np.r_[T.t, quat.vec]
