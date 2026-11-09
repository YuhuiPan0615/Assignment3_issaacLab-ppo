from enum import IntEnum
import numpy as np


class EndReason(IntEnum):
    RUNNING = 0
    COLLISION = 1
    SUCCESS = 2
    OUT_OF_BOUNDS = 3
    TIMEOUT = 4


def classify_end(collision, distance, position, steps, cfg):
    outside = ((np.abs(position[:, 0]) > cfg.room_width / 2 + cfg.boundary_tolerance)
               | (np.abs(position[:, 1]) > cfg.room_length / 2 + cfg.boundary_tolerance))
    reason = np.zeros(len(distance), dtype=np.int8)
    # Assign from lowest to highest priority so there is only one outcome.
    reason[steps >= cfg.max_steps] = EndReason.TIMEOUT
    reason[outside] = EndReason.OUT_OF_BOUNDS
    reason[distance <= cfg.goal_radius] = EndReason.SUCCESS
    reason[collision] = EndReason.COLLISION
    terminated = (reason != EndReason.RUNNING) & (reason != EndReason.TIMEOUT)
    return terminated, reason == EndReason.TIMEOUT, reason
