#!/usr/bin/env python3
"""Standalone arm-only point-reachability test.

This does not use ROS or sensors, and does not go through the waypointed
approach the real grasp/place/handover actions use. For each candidate 3D
point in the Piper base frame, it runs position-only IK once to find a
reachable pose, then QP-servos the arm directly to that single point.
The fake robot assumes each published arm command is perfectly reached before
next controller tick.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
SRC_ROOT = THIS.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mm_actions.actions import base_action as base_action_module  # noqa: E402
from mm_actions.actions.base_action import BaseAction  # noqa: E402


DEFAULT_POINTS = [
    [0.34, 0.00, 0.16],
    [0.42, 0.12, 0.18],
    [0.42, -0.12, 0.18],
    [0.50, 0.00, 0.20],
]


@dataclass
class FakeArmOnlySystem:
    q: np.ndarray = field(default_factory=lambda: np.r_[np.zeros(6), 0.10])
    arm_publish_count: int = 0
    gripper_history: list = field(default_factory=list)

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
        self.gripper_history.append(float(gripper))

    def is_holding_tightly(self, threshold=None):
        return True


def parse_points(args):
    if args.point is not None:
        return [args.point]
    return DEFAULT_POINTS


def run_case(point, timeout_s, tolerance, realtime):
    if not realtime:
        base_action_module.time.sleep = lambda _seconds: None

    fake = FakeArmOnlySystem()
    action = BaseAction(
        fake.get_image,
        fake.get_joint_state,
        fake.publish_arm_cmd,
        fake.is_holding_tightly,
        True,
        whole_body=False,
    )

    point = np.asarray(point, dtype=float)
    start_q = fake.get_joint_state()[:6]
    success, message = action.move_to_points(point, start_q, timeout_s=timeout_s)

    final_q = fake.get_joint_state()[:6]
    final_T = action._arm_robot.fkine(final_q, include_base=False)
    final_error = float(np.linalg.norm(final_T.t - point))

    passed = bool(success) and final_error <= tolerance
    return {
        "point": point,
        "success": bool(success),
        "message": message,
        "final_error_m": final_error,
        "arm_publish_count": fake.arm_publish_count,
        "final_q": final_q,
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

    print("ARM-ONLY POINT REACHABILITY TEST")
    print(f"cases={len(results)} tolerance={args.tolerance:.3f} m")
    for i, result in enumerate(results, start=1):
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{i}] {status} point={np.round(result['point'], 4).tolist()} "
            f"success={result['success']} err={result['final_error_m']:.4f} m "
            f"arm_cmds={result['arm_publish_count']} message='{result['message']}'"
        )
        print(f"    final_q={np.round(result['final_q'], 4).tolist()}")

    passed = sum(1 for result in results if result["passed"])
    print(f"summary: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
