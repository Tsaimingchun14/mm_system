#!/usr/bin/env python3
"""Standalone whole-body point-reachability test.

This does not use ROS or sensors, and does not go through the waypointed
approach the real actions use. For each candidate 3D point in the Piper base
frame, it runs whole-body position-only IK once to find a reachable base+arm
pose, then QP-servos base and arm directly to that single point. The fake
base integrates commanded yaw/forward velocities into odom, and the fake arm
perfectly reaches each published joint command before the next tick.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from spatialmath import SE3, UnitQuaternion

THIS = Path(__file__).resolve()
SRC_ROOT = THIS.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mm_actions.actions import base_action as base_action_module  # noqa: E402
from mm_actions.actions.base_action import BaseAction  # noqa: E402
from mm_actions.motion.piper_kinematic import piper_point_to_kachaka  # noqa: E402


DEFAULT_POINTS = [
    [0.45, 0.00, 0.15],
    [0.55, 0.18, 0.16],
    [0.55, -0.18, 0.16],
    [0.18, 0.35, 0.18],
    [0.18, -0.35, 0.18],
]


@dataclass
class FakeWholeBodySystem:
    q: np.ndarray = field(default_factory=lambda: np.r_[np.zeros(6), 0.10])
    base_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    arm_publish_count: int = 0
    base_publish_count: int = 0
    base_path: list = field(default_factory=list)
    base_cmd_history: list = field(default_factory=list)

    def get_image(self):
        return None

    def get_joint_state(self):
        return self.q.copy()

    def publish_arm_cmd(self, arm_q, gripper=None):
        arm_q = np.asarray(arm_q, dtype=float)
        if arm_q.shape != (6,):
            raise AssertionError(f"expected six arm joints, got {arm_q.shape}")
        if gripper is None:
            gripper = self.q[-1]
        self.q = np.r_[arm_q, float(gripper)]
        self.arm_publish_count += 1

    def publish_base_cmd(self, yaw_rate, forward_velocity):
        yaw_rate = float(yaw_rate)
        forward_velocity = float(forward_velocity)
        dt = BaseAction.DT
        self.base_pose[2] += yaw_rate * dt
        self.base_pose[0] += np.cos(self.base_pose[2]) * forward_velocity * dt
        self.base_pose[1] += np.sin(self.base_pose[2]) * forward_velocity * dt
        self.base_publish_count += 1
        self.base_cmd_history.append([yaw_rate, forward_velocity])
        self.base_path.append(self.base_pose.copy())

    def get_base_pose(self):
        return self.base_pose.copy()

    def is_holding_tightly(self, threshold=None):
        return True


def parse_points(args):
    if args.point is not None:
        return [args.point]
    return DEFAULT_POINTS


def pose_from_target_position_and_fk_orientation(robot, q, target_position):
    target_position = np.asarray(target_position, dtype=float)
    fk = robot.fkine(q)
    quat = UnitQuaternion(fk)
    return np.r_[target_position, quat.vec]


def run_case(point, timeout_s, tolerance, realtime):
    if not realtime:
        base_action_module.time.sleep = lambda _seconds: None

    fake = FakeWholeBodySystem()
    action = BaseAction(
        fake.get_image,
        fake.get_joint_state,
        fake.publish_arm_cmd,
        fake.is_holding_tightly,
        True,
        publish_base_cmd=fake.publish_base_cmd,
        get_base_pose=fake.get_base_pose,
        whole_body=True,
    )

    point = np.asarray(point, dtype=float)
    start_q = fake.get_joint_state()[:6]
    success, message = action.move_to_points(point, start_q, timeout_s=timeout_s)

    final_arm_q = fake.get_joint_state()[:6]
    q_model = np.r_[0.0, 0.0, final_arm_q]
    final_T = action._whole_body_robot.fkine(q_model)

    target_kachaka_at_inspection = piper_point_to_kachaka(point)
    target_pose_at_inspection = pose_from_target_position_and_fk_orientation(
        action._whole_body_robot,
        q_model,
        target_kachaka_at_inspection,
    )
    target_now_T = action._update_target_pose_from_base_motion(
        target_pose_at_inspection,
        np.zeros(3),
        fake.get_base_pose(),
    )
    target_now = target_now_T[:3, 3]
    final_error = float(np.linalg.norm(final_T.t - target_now))

    if fake.base_path:
        base_path = np.asarray(fake.base_path)
        base_distance = float(np.sum(np.linalg.norm(np.diff(base_path[:, :2], axis=0), axis=1))) if len(base_path) > 1 else 0.0
    else:
        base_distance = 0.0

    passed = bool(success) and final_error <= tolerance
    return {
        "point": point,
        "success": bool(success),
        "message": message,
        "final_error_m": final_error,
        "arm_publish_count": fake.arm_publish_count,
        "base_publish_count": fake.base_publish_count,
        "base_pose": fake.get_base_pose(),
        "base_distance_m": base_distance,
        "final_q": final_arm_q,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", nargs=3, type=float, metavar=("X", "Y", "Z"), help="single Piper-base target point in meters")
    parser.add_argument("--timeout", type=float, default=12.0, help="timeout per point in seconds")
    parser.add_argument("--tolerance", type=float, default=0.01, help="max final Euclidean position error in meters")
    parser.add_argument("--realtime", action="store_true", help="do not monkeypatch sleep; run at the controller rate")
    args = parser.parse_args()

    results = [run_case(point, args.timeout, args.tolerance, args.realtime) for point in parse_points(args)]

    print("WHOLE-BODY POINT REACHABILITY TEST")
    print(f"cases={len(results)} tolerance={args.tolerance:.3f} m")
    for i, result in enumerate(results, start=1):
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{i}] {status} point={np.round(result['point'], 4).tolist()} "
            f"success={result['success']} err={result['final_error_m']:.4f} m "
            f"arm_cmds={result['arm_publish_count']} base_cmds={result['base_publish_count']} "
            f"base_distance={result['base_distance_m']:.4f} m message='{result['message']}'"
        )
        print(f"    base_pose={np.round(result['base_pose'], 4).tolist()}")
        print(f"    final_q={np.round(result['final_q'], 4).tolist()}")

    passed = sum(1 for result in results if result["passed"])
    print(f"summary: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
