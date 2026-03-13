ACTION_NAME = 'grasp'

import numpy as np
import rerun as rr
import time

from mm_actions.actions.base_action import BaseAction
from mm_actions.logging.loggin import log_frame, overlay_point_rgb
from mm_actions.motion.piper_kinematic import find_reachable_pose
from mm_actions.perception.utils import camera_2d_to_3d


class GraspAction(BaseAction):
    def run(self):

        rgb = self._image.get("rgb")
        depth = self._image.get("depth")
        intrinsics = self._image.get("intrinsics")

        rr.log("grasp/image/rgb", rr.Image(overlay_point_rgb(rgb, self._point)))
        rr.log("grasp/image/depth", rr.DepthImage(depth))

        # thickness of object set to 0.06m. this would make tooltip go deeper when approaching
        point_cam = camera_2d_to_3d(self._point, depth, intrinsics, depth_offset_m=0.06) 
        if point_cam is None:
            print("point_cam=None")
            return False, "grasp aborted: invalid depth at target point"
        print(
            "point_cam: x={:.4f}, y={:.4f}, z={:.4f}".format(
                float(point_cam[0]),
                float(point_cam[1]),
                float(point_cam[2]),
            )
        )

        joint_state_for_cam = self._joint_state_at_image
        point_base = self.convert_camera_to_base(point_cam, joint_state_for_cam)

        base_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
        ee_pose = self.get_ee_pose(joint_state_for_cam) if joint_state_for_cam is not None else None
        log_frame("grasp/world/base", base_pose)
        log_frame("grasp/world/ee", ee_pose)
        rr.log(
            "grasp/world/point_base",
            rr.Points3D([point_base], colors=[[255, 255, 0]], radii=0.01),
        )

        print(
            "point_base: x={:.4f}, y={:.4f}, z={:.4f}".format(
                float(point_base[0]),
                float(point_base[1]),
                float(point_base[2]),
            )
        )

        joint_state = self._get_joint_state()
        self._publish_arm_cmd(joint_state[:6], gripper=0.1)

        q = np.array(joint_state[:6], dtype=float)

        # Temporary hack (originally grasping higher than point on picture)
        point_base[2] -= 0.03

        target_pose = find_reachable_pose(self._robot, q, point_base)
        if target_pose is None:
            return False, "grasp aborted: IK failed for target point"

        success, message = self.move_arm_to_pose(target_pose)
        if not success:
            return False, message
        
        joint_state = self._get_joint_state()
        self._publish_arm_cmd(joint_state[:6], gripper=0.06)
        time.sleep(1.0)

        home_joint_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06]
        self.move_arm_to_joint_state(home_joint_state)

        return True, "grasp complete"


ACTION_CLASS = GraspAction
