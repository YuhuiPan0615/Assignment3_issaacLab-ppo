import numpy as np
from ..robot.differential_drive import normalize_velocity


def build_observation(sectors, distance, sine, cosine, velocity, previous_action, cfg):
    observation = np.column_stack((sectors / cfg.lidar_range,
                                   np.clip(distance / cfg.goal_distance_scale, 0, 1),
                                   sine, cosine, normalize_velocity(velocity, cfg),
                                   previous_action)).astype(np.float32)
    if observation.shape[1] != 23 or not np.isfinite(observation).all():
        raise ValueError("observation must be finite and exactly 23 dimensional")
    return observation
