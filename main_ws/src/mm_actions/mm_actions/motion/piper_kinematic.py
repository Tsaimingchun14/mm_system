import numpy as np
import roboticstoolbox as rtb
import qpsolvers as qp
from spatialmath import SE3, UnitQuaternion
import logging

logger = logging.getLogger(__name__)

PIPER_MOUNT_IN_KACHAKA = np.array([0.08, 0.0, 0.525], dtype=float)
IK_FEASIBILITY_TOLERANCE_M = 0.001


def servo(robot, q, wTe, Tep, et):
    """
    Compute joint velocities for Piper arm via QP servoing.

    Args:
        robot: roboticstoolbox robot model.
        q: Current joint positions (n,).
        wTe: Current end-effector pose (SE3).
        Tep: Target end-effector pose (4x4 numpy array).
        et: Scalar position error magnitude.

    Returns:
        qd: Joint velocity command (n,) or None if QP fails.
    """
    n = robot.n

    slack_weight = min(1.0 / et, 50.0) if et > 0 else 50.0
    Q = np.eye(n + 6)
    Q[:n, :n] *= 0.01
    Q[n:, n:] = slack_weight * np.eye(6)

    v, _ = rtb.p_servo(wTe, Tep, 1.5)
    v[3:] *= 0.5

    Aeq = np.c_[robot.jacobe(q), np.eye(6)]
    beq = v.reshape((6,))

    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)
    Ain[:n, :n], bin[:n] = robot.joint_velocity_damper(0.1, 0.9, n)

    c = np.zeros(n + 6)
    lb = -np.r_[robot.qdlim[:n], 10 * np.ones(6)]
    ub =  np.r_[robot.qdlim[:n], 10 * np.ones(6)]

    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='quadprog')
    if qd is None:
        return None

    qd = qd[:n]

    if et > 0.5:
        qd *= 0.7 / et
    else:
        qd *= 1.4

    qd_norm = np.linalg.norm(qd)
    if 0 < qd_norm < 0.05:
        qd *= 0.05 / qd_norm

    return qd


def whole_body_servo(robot, q, wTe, Tep, et):
    """Compute [base_yaw, base_forward, arm...] velocities for position servoing."""
    n = robot.n
    if n != 8:
        raise ValueError(f"whole_body_servo expects an 8-DOF robot, got {n}")

    slack_weight = min(1.0 / et, 50.0) if et > 0 else 50.0
    Q = np.eye(n + 3)
    Q[:n, :n] *= 0.01
    if et > 0:
        Q[:2, :2] *= 1.0 / et
    Q[n:, n:] = slack_weight * np.eye(3)

    v, _ = rtb.p_servo(wTe, Tep, 1.5)
    Aeq = np.c_[robot.jacobe(q)[:3, :], np.eye(3)]
    beq = v[:3].reshape((3,))

    Ain = np.zeros((n + 3, n + 3))
    bin = np.zeros(n + 3)
    Ain[:n, :n], bin[:n] = robot.joint_velocity_damper(0.1, 0.9, n)

    c = np.zeros(n + 3)
    try:
        c[2:n] = -robot.jacobm(start=robot.links[4]).reshape((n - 2,))
    except Exception as exc:
        logger.debug("whole_body_servo: manipulability objective skipped: %s", exc)

    bTe = robot.fkine(q, include_base=False).A
    heading_error = 0.5 * np.arctan2(bTe[1, -1], bTe[0, -1])
    c[0] = -heading_error

    lb = -np.r_[robot.qdlim[:n], 10 * np.ones(3)]
    ub = np.r_[robot.qdlim[:n], 10 * np.ones(3)]

    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='quadprog')
    if qd is None:
        return None

    qd = qd[:n]
    if et > 0.5:
        qd *= 0.7 / et
    else:
        qd *= 1.4

    return qd


def piper_point_to_kachaka(point_piper_base):
    point_piper_base = np.asarray(point_piper_base, dtype=float)
    return point_piper_base + PIPER_MOUNT_IN_KACHAKA


ARM_SEEDS = [
    np.zeros(6),
    np.array([0.0, 1.0, -2.0, 0.0, 0.8, 0.0]),
    np.array([0.0, 1.6, -1.3, 0.0, 1.2, 0.0]),
    np.array([0.0, 2.0, -0.8, 0.0, 1.0, 0.0]),
]


def make_seeds(robot, init_arm_q, target_position):
    """Yield candidate q0 seeds for ik_LM, shaped to match robot.n.

    A 6-DOF arm-only model gets the current arm pose plus a handful of
    curated alternate arm postures. An 8-DOF base+arm model additionally
    sweeps candidate base yaw/forward positions, since the base can reach
    the same point from many different places and a local IK solver can't
    discover "drive somewhere else" from a single nearby seed.
    """
    init_arm_q = np.asarray(init_arm_q, dtype=float)
    if init_arm_q.shape != (6,):
        raise ValueError("init_arm_q must contain six Piper joints")

    arm_seeds = [init_arm_q] + ARM_SEEDS

    if robot.n == 6:
        for arm_q in arm_seeds:
            yield arm_q
        return

    target_position = np.asarray(target_position, dtype=float)
    x, y, _ = target_position
    yaw0 = np.arctan2(y, x) if abs(x) + abs(y) > 1e-9 else 0.0
    distance = float(np.hypot(x, y))
    base_forward_seeds = [
        0.0,
        max(0.0, distance - 0.18),
        max(0.0, distance - 0.35),
        min(distance, 1.0),
        -0.2,
    ]
    base_yaw_seeds = [0.0, yaw0, 0.5 * yaw0, -yaw0, np.pi / 2, -np.pi / 2, np.pi, -np.pi]

    for yaw in base_yaw_seeds:
        for forward in base_forward_seeds:
            for arm_q in arm_seeds:
                yield np.r_[yaw, forward, arm_q]


def find_reachable_pose(
    robot,
    init_arm_q,
    target_position,
    max_base_forward_m=1.0,
    max_base_yaw_rad=np.pi,
    mask=None,
) -> np.ndarray | None:
    """Position-only IK probe returning [x, y, z, qw, qx, qy, qz].

    Works for both the 6-DOF arm-only Piper model and the 8-DOF
    KachakaPiper base+arm model: it sweeps every seed from make_seeds,
    keeps the best-scoring feasible solution, and only applies base-travel
    limits when the robot actually has base DOF (robot.n == 8).
    """
    if mask is None:
        mask = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    init_arm_q = np.asarray(init_arm_q, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    has_base = robot.n == 8
    T = SE3.Trans(target_position[0], target_position[1], target_position[2])
    best = None

    for q0 in make_seeds(robot, init_arm_q, target_position):
        sol = robot.ik_LM(
            T,
            q0=q0,
            mask=mask,
            method="chan",
            k=0.1,
            ilimit=1000,
            slimit=1,
            tol=1e-4,
            joint_limits=True,
        )
        q_sol = sol[0]
        T_sol = robot.fkine(q_sol)
        err = float(np.linalg.norm(T_sol.t - target_position))
        success = sol[1] == 1
        feasible = success and err <= IK_FEASIBILITY_TOLERANCE_M
        base_travel_cost = 0.0
        if has_base:
            feasible = (
                feasible
                and abs(float(q_sol[0])) <= max_base_yaw_rad
                and abs(float(q_sol[1])) <= max_base_forward_m
            )
            base_travel_cost = abs(float(q_sol[1])) + 0.2 * abs(float(q_sol[0]))
        score = (
            0 if feasible else 1,
            err,
            base_travel_cost,
            float(np.linalg.norm(q_sol[-6:] - init_arm_q)),
        )
        if best is None or score < best[0]:
            best = (score, feasible, q_sol, T_sol)

    if best is None or not best[1]:
        logger.warning(
            "find_reachable_pose: IK failed for position %s",
            target_position.tolist(),
        )
        return None

    q_sol = best[2]
    T_sol = best[3]
    logger.info(
        "find_reachable_pose: target=%s err=%.4f",
        target_position.tolist(),
        float(best[0][1]),
    )
    new_ori = UnitQuaternion(T_sol)
    return np.r_[target_position, new_ori.vec]
