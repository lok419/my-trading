from enum import Enum

class GRID_STATUS(Enum):
    IDLE = 1
    ACTIVE = 2
    NEUTRAL = 3

class GRID_TYPE(Enum):
    MEAN_REVERT = 0
    MOMENTUM_UP = 1
    MOMENTUM_DOWN = 2

class TS_PROP(Enum):
    RANDOM = 0
    MEAN_REVERT = 1
    MOMENTUM = 2

class STATUS(Enum):
    RUN = 0
    PAUSE = 1
    TERMINATE = 2
    STOP = 4