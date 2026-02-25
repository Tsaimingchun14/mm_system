from enum import Enum

class JointControllerStatus(Enum):
    FAIL = -1
    WORKING = 0
    IDLE = 1