#!/usr/bin/env python3
"""Reachability experiment: arm-only vs whole-body across random 3D points.

Reuses the exact same per-point flow as test_arm_only_point_servo.py and
test_whole_body_point_servo.py (one find_reachable_pose IK search + one QP
servo run per point, no ROS/sensors, fake plant with perfect tracking) but
sweeps 300 randomly sampled points instead of a handful of fixed ones, and
reports reachability broken down by distance bucket.

Buckets by xy-plane radius from the Piper base:
  near  (0, 0.20] m   - 100 points
  mid   (0.20, 0.60] m - 100 points
  far   (0.60, 1.50] m - 100 points
Height z is uniform in [0.0, 0.30] m and angle is uniform over the full
circle for every point, independent of bucket.

The same 300 points are used for both modes so the comparison is apples to
apples.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
TEST_DIR = THIS.parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

import test_arm_only_point_servo as arm_test  # noqa: E402
import test_whole_body_point_servo as wb_test  # noqa: E402


BUCKETS = [
    ("near_0-20cm", 0.0, 0.20),
    ("mid_20-60cm", 0.20, 0.60),
    ("far_60-150cm", 0.60, 1.50),
]
Z_MIN_M, Z_MAX_M = 0.0, 0.30


def sample_bucket(rng, n, r_min, r_max):
    r = rng.uniform(r_min, r_max, size=n)
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    z = rng.uniform(Z_MIN_M, Z_MAX_M, size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def is_ik_feasible(message):
    return "IK failed" not in message


TIMEOUT_S = 12.0
TOLERANCE_M = 0.01


def now_str():
    return time.strftime("%H:%M:%S")


def run_mode(module, points, label, progress_every):
    results = []
    ik_ok = 0
    servo_ok = 0
    t0 = time.time()
    for i, point in enumerate(points):
        result = module.run_case(point, TIMEOUT_S, TOLERANCE_M, False)
        results.append(result)
        if is_ik_feasible(result["message"]):
            ik_ok += 1
        if result["passed"]:
            servo_ok += 1
        if (i + 1) % progress_every == 0 or (i + 1) == len(points):
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            print(
                f"[{now_str()}] [{label}] {i + 1}/{len(points)} "
                f"ik_ok={ik_ok} servo_ok={servo_ok} "
                f"elapsed={elapsed:.1f}s ({rate:.2f}s/point)",
                flush=True,
            )
    return results


def summarize(results):
    n = len(results)
    ik_ok = sum(1 for r in results if is_ik_feasible(r["message"]))
    servo_ok = sum(1 for r in results if r["passed"])
    return n, ik_ok, servo_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-bucket", type=int, default=100, help="points sampled per distance bucket")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed, shared across buckets and modes")
    parser.add_argument("--progress-every", type=int, default=10, help="print a progress line every N points")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    bucket_points = {label: sample_bucket(rng, args.n_per_bucket, r_min, r_max) for label, r_min, r_max in BUCKETS}

    print("REACHABILITY EXPERIMENT", flush=True)
    print(
        f"seed={args.seed} n_per_bucket={args.n_per_bucket} total_points={args.n_per_bucket * len(BUCKETS)}",
        flush=True,
    )
    print(f"z in [{Z_MIN_M}, {Z_MAX_M}] m, angle uniform over full circle", flush=True)

    rows = []
    for label, r_min, r_max in BUCKETS:
        points = bucket_points[label]
        print(f"\n=== bucket {label} (radius in ({r_min}, {r_max}] m) ===", flush=True)

        print("  arm-only...", flush=True)
        arm_results = run_mode(arm_test, points, "arm-only", args.progress_every)
        rows.append(("arm-only", label, *summarize(arm_results)))

        print("  whole-body...", flush=True)
        wb_results = run_mode(wb_test, points, "whole-body", args.progress_every)
        rows.append(("whole-body", label, *summarize(wb_results)))

    print("\n" + "=" * 78, flush=True)
    print(f"{'mode':<12}{'bucket':<16}{'n':>5}{'ik_reachable':>14}{'ik_%':>8}{'servo_ok':>10}{'servo_%':>9}")
    for mode, label, n, ik_ok, servo_ok in rows:
        print(
            f"{mode:<12}{label:<16}{n:>5}{ik_ok:>14}{100 * ik_ok / n:>7.1f}%"
            f"{servo_ok:>10}{100 * servo_ok / n:>8.1f}%"
        )

    print("\n" + "-" * 78)
    for mode in ("arm-only", "whole-body"):
        mode_rows = [r for r in rows if r[0] == mode]
        n_total = sum(r[2] for r in mode_rows)
        ik_total = sum(r[3] for r in mode_rows)
        servo_total = sum(r[4] for r in mode_rows)
        print(
            f"{mode:<12}{'ALL':<16}{n_total:>5}{ik_total:>14}{100 * ik_total / n_total:>7.1f}%"
            f"{servo_total:>10}{100 * servo_total / n_total:>8.1f}%"
        )


if __name__ == "__main__":
    raise SystemExit(main())
