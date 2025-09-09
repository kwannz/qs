from enum import IntEnum

class WorkflowStatus(IntEnum):
    PENDING = 0
    RUNNING = 1
    SUCCESS = 2
    FAILED = 3
    MANUAL_STOP = 4