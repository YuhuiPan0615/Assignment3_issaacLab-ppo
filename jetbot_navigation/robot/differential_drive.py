import numpy as np
from ..environment.navigation_env_cfg import NavigationEnvCfg


def normalize_velocity(velocity, cfg: NavigationEnvCfg):
    result = np.array(velocity, dtype=np.float64, copy=True)
    scale = np.where(result[..., 0] >= 0, cfg.forward_speed, cfg.reverse_speed)
    result[..., 0] /= scale
    result[..., 1] /= cfg.angular_speed
    return np.clip(result, -1, 1)


def process_actions(actions, previous_velocity, cfg: NavigationEnvCfg):
    """Returns wheel targets, achievable twist, and the applied normalized action.

    Joint limits scale both wheel speeds together, preserving curvature. The
    reported action is reconstructed after acceleration AND wheel saturation.
    """
    actions = np.asarray(actions, dtype=np.float64)
    previous_velocity = np.asarray(previous_velocity, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2 or actions.shape != previous_velocity.shape:
        raise ValueError("actions and previous_velocity must have shape (N, 2)")
    if not np.isfinite(actions).all() or not np.isfinite(previous_velocity).all():
        raise ValueError("actions and previous_velocity must be finite")
    clipped = np.clip(actions, -1, 1)
    desired = clipped.copy()
    desired[:, 0] *= np.where(desired[:, 0] >= 0, cfg.forward_speed, cfg.reverse_speed)
    desired[:, 1] *= cfg.angular_speed
    limit = np.array([cfg.linear_acceleration, cfg.angular_acceleration]) * cfg.control_dt
    twist = previous_velocity + np.clip(desired - previous_velocity, -limit, limit)
    wheels = np.stack((twist[:, 0] - twist[:, 1] * cfg.wheel_base / 2,
                       twist[:, 0] + twist[:, 1] * cfg.wheel_base / 2), axis=1) / cfg.wheel_radius
    saturation = np.maximum(1, np.abs(wheels).max(axis=1) / cfg.wheel_speed_limit)
    wheels /= saturation[:, None]
    twist /= saturation[:, None]
    wheels *= [cfg.left_wheel_sign, cfg.right_wheel_sign]
    return wheels, twist, normalize_velocity(twist, cfg)
