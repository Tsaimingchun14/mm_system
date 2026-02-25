
from typing import List, Dict, Type
from .controller_status import JointControllerStatus

# Action registry
_ACTION_REGISTRY: Dict[str, Type] = {}

class ActionMeta(type):
    def __new__(cls, name, bases, dct):
        klass = super().__new__(cls, name, bases, dct)
        if hasattr(klass, 'action') and isinstance(getattr(klass, 'action'), str):
            _ACTION_REGISTRY[getattr(klass, 'action')] = klass
        return klass

class BaseAction(metaclass=ActionMeta):
    action: str = "base"
    def __init__(self):
        pass
    def step(self, context):
        raise NotImplementedError
    def is_completed(self) -> bool:
        raise NotImplementedError
    def failed(self) -> bool:
        return False


class GraspAction(BaseAction):
    action: str = "grasp"
    def __init__(self, point: List[float]):
        if not (isinstance(point, list) and len(point) == 2 and all(isinstance(x, float) for x in point)):
            raise ValueError("point must be a list of two floats")
        self.point = point
        self.steps = ["prepare", "pregrasp", "grasp", "close_gripper", "retract"]
        self.current_step = 0
        self.failed_flag = False
        self.entered_states = set()

    def step(self, context):
        if self.current_step >= len(self.steps):
            return
        curr_state = self.steps[self.current_step]
        if curr_state == "prepare":
            context.update_perception_prompt(self.point)
            self.current_step += 1
        elif curr_state == "pregrasp":
            if curr_state in self.entered_states and context.joint_controller_status == JointControllerStatus.IDLE:
                self.current_step += 1
                return
            object_center_cam = context.get_object_center_3d_camera()
            if object_center_cam is None:
                print("Object center not found, cannot proceed with pregrasp.")
                return
            else:
                object_center_base = context.convert_camera_to_base(object_center_cam)
                #TODO: generate pregrasp pose based on object_center_base
                ee_pose_base = [object_center_base[0], object_center_base[1], object_center_base[2]+0.05, 0.0, 0.0, 1.0, 0.0]  
                context.publish_ee_pose(ee_pose_base)
        elif curr_state == "grasp":
            if curr_state in self.entered_states and context.joint_controller_status == JointControllerStatus.IDLE:
                self.current_step += 1
                return
            object_center_cam = context.get_object_center_3d_camera()
            if object_center_cam is None:
                print("Object center not found, cannot proceed with grasp.")
                return
            else:
                object_center_base = context.convert_camera_to_base(object_center_cam)
                #TODO: generate grasp pose based on object_center_base
                ee_pose_base = [object_center_base[0], object_center_base[1], object_center_base[2], 0.0, 0.0, 1.0, 0.0]  
                context.publish_ee_pose(ee_pose_base)
        elif curr_state == "close_gripper":
            print("Closing gripper not implemented yet.")
            self.current_step += 1
        elif curr_state == "retract":
            print("Retracting arm not implemented yet.")
            self.current_step += 1
            
            
        self.entered_states.add(curr_state)
                
    def is_completed(self) -> bool:
        return self.current_step >= len(self.steps)

    def failed(self) -> bool:
        return self.failed_flag

class HandOverAction(BaseAction):
    action: str = "hand_over"
    def __init__(self, point: List[float]):
        if not (isinstance(point, list) and len(point) == 2 and all(isinstance(x, float) for x in point)):
            raise ValueError("point must be a list of two floats")
        self.point = point
        self.steps = ["approach", "extend", "release", "retract"]
        self.current_step = 0
        self.failed_flag = False

    def step(self, context):
        if self.current_step < len(self.steps):
            self.current_step += 1
        else:
            pass

    def is_completed(self) -> bool:
        return self.current_step >= len(self.steps)

    def failed(self) -> bool:
        return self.failed_flag

def parse_action(d) -> BaseAction:
    if "action" not in d:
        raise ValueError("Missing action field")
    action_name = d["action"]
    if action_name not in _ACTION_REGISTRY:
        raise ValueError(f"Unsupported action: {action_name}")
    klass = _ACTION_REGISTRY[action_name]
    return klass(**{k: v for k, v in d.items() if k != "action"})