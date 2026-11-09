import numpy as np
from .terminations import EndReason


def reward_components(previous_distance, distance, previous_action, applied_action,
                      sectors, body_forward_speed, reason, cfg):
    front = sectors[:, [15, 0, 1]].min(axis=1)
    proximity = np.clip((cfg.safe_distance - front) / cfg.safe_distance, 0, 1)
    forward = np.clip(body_forward_speed / cfg.forward_speed, 0, 1)
    return {
        "progress": 2.0 * np.clip(previous_distance - distance, -0.05, 0.05),
        "step": np.full(len(distance), -0.005),
        "smoothness": -0.02 * np.square(applied_action - previous_action).sum(axis=1),
        "safety": -0.1 * proximity * forward,
        "success": 20.0 * (reason == EndReason.SUCCESS),
        "collision": -25.0 * (reason == EndReason.COLLISION),
        "out_of_bounds": -25.0 * (reason == EndReason.OUT_OF_BOUNDS),
        "timeout": -3.0 * (reason == EndReason.TIMEOUT),
    }
