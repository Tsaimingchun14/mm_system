ACTION_NAME = 'place'

import time
import numpy as np
import rerun as rr

from mm_actions.actions.base_action import BaseAction
from mm_actions.logging.loggin import log_frame, overlay_point_rgb
from mm_actions.perception.utils import camera_2d_to_3d


class PlaceAction(BaseAction):
    PLACE_HEIGHT_OFFSET_M = 0.07

    def run(self):

        rgb = self._image.get("rgb")
        depth = self._image.get("depth")
        intrinsics = self._image.get("intrinsics")

        rr.log("place/image/rgb", rr.Image(overlay_point_rgb(rgb, self._point)))
        rr.log("place/image/depth", rr.DepthImage(depth))

        point_cam = camera_2d_to_3d(self._point, depth, intrinsics)
        if point_cam is None:
            return False, "place aborted: invalid depth at target point"
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
            return False, "place aborted: invalid target point"

        target_point_base = np.asarray(point_base, dtype=float).copy()
        target_point_base[2] += self.PLACE_HEIGHT_OFFSET_M

        base_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
        ee_pose = self.get_ee_pose(joint_state_for_cam) if joint_state_for_cam is not None else None
        log_frame("place/world/base", base_pose)
        log_frame("place/world/ee", ee_pose)
        rr.log(
            "place/world/point_base",
            rr.Points3D([point_base], colors=[[255, 255, 0]], radii=0.01),
        )
        rr.log(
            "place/world/target_point_base",
            rr.Points3D([target_point_base], colors=[[0, 255, 255]], radii=0.01),
        )

        print(
            "point_base: x={:.4f}, y={:.4f}, z={:.4f}".format(
                float(point_base[0]),
                float(point_base[1]),
                float(point_base[2]),
            )
        )
        print(
            "target_point_base: x={:.4f}, y={:.4f}, z={:.4f}".format(
                float(target_point_base[0]),
                float(target_point_base[1]),
                float(target_point_base[2]),
            )
        )

        q = np.array(joint_state_for_cam[:6], dtype=float)
        success, message = self.move_to_point_with_waypoint(target_point_base, q)
        if not success:
            return False, message

        time.sleep(1.0)
        joint_state = self._get_joint_state()
        self._publish_arm_cmd(joint_state[:6], gripper=0.1)
        time.sleep(1.0)
        home_joint_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
        self.move_arm_to_joint_state(home_joint_state)

        return True, "place complete"


ACTION_CLASS = PlaceAction
