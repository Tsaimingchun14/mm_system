from typing import List
from mm_interface.msg import JointPlannerFeedback
from .action_types import BaseAction

class GraspAction(BaseAction):
    action: str = "grasp"

    def __init__(self, point: List[float], ref_image=None):
        if not (isinstance(point, list) and len(point) == 2 and all(isinstance(x, float) for x in point)):
            raise ValueError("point must be a list of two floats")
        self.point = point
        self.ref_image = ref_image
        self.steps = ["prepare", "pregrasp", "grasp", "close_gripper", "retract"]
        self.current_step = 0
        self.failed_flag = False
        
        # State tracking
        self.is_moving = False
        self.target_base = None
        self.step_start_time = None

    def step(self, context, data: dict):
        if self.is_completed() or self.failed():
            return
        
        curr_state = self.steps[self.current_step]
        
        # 1. Check if previously commanded move finished
        if self.is_moving:
            if context.joint_planner_status == JointPlannerFeedback.IDLE:
                print(f"Step {curr_state} reached target.")
                self._advance_step()
            elif context.joint_planner_status == JointPlannerFeedback.FAIL:
                print(f"Step {curr_state} failed in controller.")
                self.failed_flag = True
            return

        # 2. Step Logic
        if curr_state == "prepare":
            # Update prompt with the selected point and reference image.
            context.update_perception_prompt(self.point, self.ref_image)
            self._advance_step()
            
        elif curr_state in ["pregrasp", "grasp"]:
            if self.step_start_time is None:
                self.step_start_time = context.get_clock().now()

            # Perception: Get target center in camera frame
            rgb, depth, K, joints = data['image'], data['depth'], data['intrinsics'], data['joints']
            res = context.get_object_center_3d_camera(rgb, depth, K)
            
            if res is not None:
                self.target_base = context.convert_camera_to_base(res, joints)
                z_offset = 0.05 if curr_state == "pregrasp" else 0.0
                # Orientation: [qw, qx, qy, qz] = [0, 0, 1, 0] for looking down
                pose = [
                    self.target_base[0], 
                    self.target_base[1], 
                    self.target_base[2] + z_offset,
                    0.0, 0.0, 1.0, 0.0
                ]
                self._goto_pose(context, pose)
                context.publish_gripper_goal(0.08)
            else:
                elapsed = (context.get_clock().now() - self.step_start_time).nanoseconds / 1e9
                if elapsed > 3.0:
                    print(f"Action Failed: Perception renewal timeout (3s) in {curr_state}")
                    self.failed_flag = True
            
        elif curr_state == "close_gripper":
            print("Closing gripper...")
            context.publish_gripper_goal(0.05) # Close gripper
            self._advance_step()
            
        elif curr_state == "retract":
            # Fixed pose: position [0.2, 0, 0.2]
            # Orientation: pitch up 45 deg from looking down (135 deg around Y)
            # Resulting quaternion [qw, qx, qy, qz] = [0.38268, 0, 0.92388, 0]
            retract_pose = [0.2, 0.0, 0.2, 0.38268, 0.0, 0.92388, 0.0]
            self._goto_pose(context, retract_pose)

    def _goto_pose(self, context, pose):
        """Sends a pose [x, y, z, qw, qx, qy, qz] to the controller."""
        context.publish_ee_goal(pose)
        self.is_moving = True

    def _advance_step(self):
        self.current_step += 1
        self.is_moving = False
        self.step_start_time = None
                
    def is_completed(self) -> bool:
        return self.current_step >= len(self.steps)

    def failed(self) -> bool:
        return self.failed_flag
