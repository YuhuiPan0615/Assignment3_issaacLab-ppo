"""Torch-native navigation math used by the real Isaac Lab environment.

This module deliberately contains no NumPy conversion.  Every function keeps
the leading environment dimension and executes on the input tensor's device.
"""
from __future__ import annotations

import torch


def process_actions(actions, previous_velocity, cfg):
    """Map normalized twist actions to saturated wheel velocity targets."""
    clipped = actions.clamp(-1.0, 1.0)
    forward_scale = torch.where(
        clipped[:, 0] >= 0.0,
        torch.full_like(clipped[:, 0], cfg.forward_speed),
        torch.full_like(clipped[:, 0], cfg.reverse_speed),
    )
    desired = torch.stack(
        (clipped[:, 0] * forward_scale, clipped[:, 1] * cfg.angular_speed), dim=-1
    )
    limits = torch.tensor(
        [cfg.linear_acceleration, cfg.angular_acceleration],
        dtype=actions.dtype,
        device=actions.device,
    ) * cfg.control_dt
    twist = previous_velocity + (desired - previous_velocity).clamp(-limits, limits)
    left = (twist[:, 0] - twist[:, 1] * cfg.wheel_base / 2.0) / cfg.wheel_radius
    right = (twist[:, 0] + twist[:, 1] * cfg.wheel_base / 2.0) / cfg.wheel_radius
    wheels = torch.stack((left, right), dim=-1)
    saturation = torch.maximum(
        torch.ones_like(wheels[:, 0]), wheels.abs().amax(dim=-1) / cfg.wheel_speed_limit
    )
    wheels = wheels / saturation.unsqueeze(-1)
    twist = twist / saturation.unsqueeze(-1)
    signs = torch.tensor(
        [cfg.left_wheel_sign, cfg.right_wheel_sign],
        dtype=actions.dtype,
        device=actions.device,
    )
    wheels = wheels * signs
    linear_scale = torch.where(
        twist[:, 0] >= 0.0,
        torch.full_like(twist[:, 0], cfg.forward_speed),
        torch.full_like(twist[:, 0], cfg.reverse_speed),
    )
    applied = torch.stack(
        (twist[:, 0] / linear_scale, twist[:, 1] / cfg.angular_speed), dim=-1
    ).clamp(-1.0, 1.0)
    return wheels, twist, applied


def lidar_sectors(distances, cfg):
    """Sanitize 64 ranges and reduce contiguous rays into 16 minima."""
    clean = torch.nan_to_num(
        distances, nan=0.0, posinf=cfg.lidar_range, neginf=0.0
    ).clamp(0.0, cfg.lidar_range)
    return clean.reshape(clean.shape[0], cfg.lidar_sectors, -1).amin(dim=-1)


def goal_state(position, yaw, goals):
    delta = goals - position
    distance = torch.linalg.vector_norm(delta, dim=-1)
    bearing = torch.atan2(delta[:, 1], delta[:, 0]) - yaw
    return distance, torch.sin(bearing), torch.cos(bearing)


def normalize_velocity(velocity, cfg):
    linear_scale = torch.where(
        velocity[:, 0] >= 0.0,
        torch.full_like(velocity[:, 0], cfg.forward_speed),
        torch.full_like(velocity[:, 0], cfg.reverse_speed),
    )
    return torch.stack(
        (velocity[:, 0] / linear_scale, velocity[:, 1] / cfg.angular_speed), dim=-1
    ).clamp(-1.0, 1.0)


def build_observation(sectors, distance, sine, cosine, velocity, previous_action, cfg):
    observation = torch.cat(
        (
            sectors / cfg.lidar_range,
            (distance / cfg.goal_distance_scale).clamp(0.0, 1.0).unsqueeze(-1),
            sine.unsqueeze(-1),
            cosine.unsqueeze(-1),
            normalize_velocity(velocity, cfg),
            previous_action,
        ),
        dim=-1,
    )
    if observation.shape[1] != 23:
        raise RuntimeError(f"expected a 23D observation, received {observation.shape}")
    return observation


def reward_components(previous_distance, distance, previous_action, applied_action,
                      sectors, body_forward_speed, reason, cfg):
    """Return the eight Stage 2 reward terms without device synchronization."""
    front = sectors[:, [15, 0, 1]].amin(dim=-1)
    proximity = ((cfg.safe_distance - front) / cfg.safe_distance).clamp(0.0, 1.0)
    forward = (body_forward_speed / cfg.forward_speed).clamp(0.0, 1.0)
    return {
        "progress": 2.0 * (previous_distance - distance).clamp(-0.05, 0.05),
        "step": torch.full_like(distance, -0.005),
        "smoothness": -0.02 * (applied_action - previous_action).square().sum(dim=-1),
        "safety": -0.1 * proximity * forward,
        "success": 20.0 * (reason == 2).to(distance.dtype),
        "collision": -25.0 * (reason == 1).to(distance.dtype),
        "out_of_bounds": -25.0 * (reason == 3).to(distance.dtype),
        "timeout": -3.0 * (reason == 4).to(distance.dtype),
    }
