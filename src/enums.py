from enum import Enum


class QueueType(Enum):
    CLASSIC = 0
    SNAKE = 1


class AgentPhase(Enum):
    SHOPPING = 0
    REACHING_QUEUE = 1
    IN_QUEUE = 2
    SNAKE_REACHING_CASHIER = 3
    PAYING = 4
