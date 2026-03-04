from typing import List
from .action_types import BaseAction

class HandOverAction(BaseAction):
    action: str = "hand_over"
    def __init__(self, point: List[float], ref_image=None):
        if not (isinstance(point, list) and len(point) == 2 and all(isinstance(x, float) for x in point)):
            raise ValueError("point must be a list of two floats")
        self.point = point
        self.ref_image = ref_image
        self.steps = ["approach", "extend", "release", "retract"]
        self.current_step = 0
        self.failed_flag = False

    def step(self, context, synced_data: dict):
        if self.current_step < len(self.steps):
            self.current_step += 1
        else:
            pass

    def is_completed(self) -> bool:
        return self.current_step >= len(self.steps)

    def failed(self) -> bool:
        return self.failed_flag
