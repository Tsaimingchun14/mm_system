import numpy as np
import roboticstoolbox as rtb
import qpsolvers as qp
from spatialmath import SE3, UnitQuaternion
import logging

logger = logging.getLogger(__name__)

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

def find_reachable_pose(robot, init_q, target_position, mask=None) -> np.ndarray | None:
    if mask is None:
        mask = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    logger.info(
        "find_reachable_pose: target_position=%s norm=%.4f",
        np.asarray(target_position).tolist(),
        float(np.linalg.norm(target_position)),
    )
    T = SE3.Trans(target_position[0], target_position[1], target_position[2])
    sol = robot.ik_LM(
        T,
        q0=init_q,
        mask=mask,
        method="chan",
        k=0.1,
        ilimit=1000,
        slimit=5,
        tol=1e-4,
        joint_limits=True,
    )

    if sol[1] != 1:
        logger.warning(
            "find_reachable_pose: IK failed for position %s, target unreachable",
            np.asarray(target_position).tolist(),
        )
        return None

    q_sol = sol[0]
    T_sol = robot.fkine(q_sol)
    new_ori = UnitQuaternion(T_sol)

    return np.r_[target_position, new_ori.vec]
