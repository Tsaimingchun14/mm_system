ACTION_NAME = 'handover'

import time
import numpy as np
import rerun as rr

from mm_actions.actions.base_action import BaseAction
from mm_actions.logging.loggin import log_frame, overlay_point_rgb
from mm_actions.motion.piper_kinematic import find_reachable_pose
from mm_actions.perception.utils import camera_2d_to_3d

class HandoverAction(BaseAction):
    def run(self):

        rgb = self._image.get("rgb")
        depth = self._image.get("depth")
        intrinsics = self._image.get("intrinsics")

        rr.log("handover/image/rgb", rr.Image(overlay_point_rgb(rgb, self._point)))
        rr.log("handover/image/depth", rr.DepthImage(depth))

        point_cam = camera_2d_to_3d(self._point, depth, intrinsics)
        if point_cam is None:
            return False, "handover aborted: invalid depth at target point"
        print(
            "point_cam: x={:.4f}, y={:.4f}, z={:.4f}".format(
                float(point_cam[0]),
                float(point_cam[1]),
                float(point_cam[2]),
            )
        )

        joint_state_for_cam = self._joint_state_at_image

        point_base = self.convert_camera_to_base(point_cam, joint_state_for_cam)
        vec = np.asarray(point_base, dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return False, "handover aborted: invalid target point"

        if norm > 0.4:
            point_base = vec / norm * 0.4
        
        base_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
        ee_pose = self.get_ee_pose(joint_state_for_cam) if joint_state_for_cam is not None else None
        log_frame("handover/world/base", base_pose)
        log_frame("handover/world/ee", ee_pose)
        rr.log(
            "handover/world/point_base",
            rr.Points3D([point_base], colors=[[255, 255, 0]], radii=0.01),
        )

        print(
            "point_base: x={:.4f}, y={:.4f}, z={:.4f}".format(
                float(point_base[0]),
                float(point_base[1]),
                float(point_base[2]),
            )
        )

        q = np.array(joint_state_for_cam[:6], dtype=float)
        target_pose = find_reachable_pose(self._robot, q, point_base)
        if target_pose is None:
            return False, "handover aborted: IK failed for target point"

        success, message = self.move_arm_to_pose(target_pose)
        if not success:
            return False, message

        time.sleep(3.0)
        joint_state = self._get_joint_state()
        self._publish_arm_cmd(joint_state[:6], gripper=0.1)
        time.sleep(1.0)
        home_joint_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
        self.move_arm_to_joint_state(home_joint_state)
        
        return True, "handover complete"


ACTION_CLASS = HandoverAction
