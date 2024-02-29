from enum import Enum

class Status(Enum):
    IDLE = 1
    ACTIVE = 2
    NEUTRAL = 3

class TS_PROP(Enum):
    RANDOM = 0
    MEAN_REVERT = 1
    MOMENTUM = 2

class GRID_TYPE(Enum):
    MEAN_REVERT = 0
    MOMENTUM_UP = 1
    MOMENTUM_DOWN = 2