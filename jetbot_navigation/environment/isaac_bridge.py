"""Adapter for configured Isaac Lab articulation and sensors.

Not an environment factory: asset/sensor configuration remains version-specific.
The CPU reference core incurs device transfers; large GPU training requires a
Torch-native port after the fixed Isaac versions and assets are confirmed.
"""
import numpy as np
from .navigation_env import SensorFrame


def to_numpy(value):
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class IsaacSensorBridge:
    def __init__(self, robot, lidar, contacts, env_origins, wheel_joint_ids,
                 ray_order, ray_origins_w, cfg):
        """contacts: one filtered ContactSensor for EACH chassis/wheel body.

        Their filters must contain only obstacles. ray_order explicitly maps the
        configured pattern to observations.ray_angles(), checked during sensor
        setup rather than inferred from a sensor's undocumented output ordering.
        ray_origins_w is a callable returning the actual world ray starts (N,R,3).
        Sensor data.pos_w alone is insufficient when the ray pattern or mounting
        offset displaces the origins.
        """
        if not contacts:
            raise ValueError("filtered chassis and wheel contact sensors are required")
        if len(wheel_joint_ids) != 2 or len(set(wheel_joint_ids)) != 2:
            raise ValueError("provide distinct [left, right] wheel joint indices")
        order = np.asarray(ray_order)
        if order.shape != (cfg.lidar_rays,) or order.dtype.kind not in 'iu' or not np.array_equal(
                np.sort(order), np.arange(cfg.lidar_rays)):
            raise ValueError("ray_order must be a permutation of every LiDAR ray")
        self.robot, self.lidar, self.contacts = robot, lidar, contacts
        self.env_origins = to_numpy(env_origins).copy()
        self.wheel_joint_ids, self.ray_order, self.cfg = list(wheel_joint_ids), order, cfg
        if not callable(ray_origins_w):
            raise ValueError("provide a callable returning actual world-space ray starts")
        self.ray_origins_w = ray_origins_w

    def frame(self):
        data = self.robot.data
        pose = to_numpy(data.root_pos_w)
        quaternion = to_numpy(data.root_quat_w)  # Isaac Lab 3: x, y, z, w
        x, y, z, w = quaternion.T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        velocity = np.column_stack((to_numpy(data.root_lin_vel_b)[:, 0],
                                    to_numpy(data.root_ang_vel_b)[:, 2]))
        hits = to_numpy(self.lidar.data.ray_hits_w)
        origins = to_numpy(self.ray_origins_w())
        if hits.shape != (len(pose), self.cfg.lidar_rays, 3) or origins.shape != hits.shape:
            raise ValueError("hit positions and ray starts must have shape (N, R, 3)")
        if not np.isfinite(origins).all():
            raise ValueError("ray starts contain invalid sensor state")
        distances = np.linalg.norm(hits - origins, axis=2)
        return SensorFrame(pose[:, :2] - self.env_origins[:, :2], yaw, velocity,
                           distances[:, self.ray_order])

    def latch_contacts(self, task):
        matrices = []
        for sensor in self.contacts:
            data = sensor.data
            value = getattr(data, "normal_force_matrix_w", None)
            if value is None:
                raise RuntimeError("filtered contact matrix unavailable; check single-body sensor and filters")
            matrix = to_numpy(value)
            if matrix.shape[0] != task.num_envs or matrix.shape[-1] != 3 or matrix.size == 0:
                raise RuntimeError("contact sensor has empty or incompatible obstacle filters")
            matrices.append(matrix.reshape(task.num_envs, -1, 3))
        task.latch_contacts(np.concatenate(matrices, axis=1))

    def apply_wheels(self, wheel_targets):
        import torch  # loaded only inside an initialized Isaac runtime
        targets = torch.as_tensor(wheel_targets, dtype=torch.float32, device=self.robot.device)
        self.robot.actuators.target_command.set_velocity_index(
            value=targets, joint_ids=self.wheel_joint_ids
        )

    def latch_control_history(self, task):
        """DirectRLEnv integration: consume four automatically updated samples.

        SensorBase.update() eagerly updates sensors with history_length > 0.
        Call this after the control loop, before finish_step/_reset_idx. History
        index 0 is newest; never call both this and per-substep latch_contacts in
        the same cycle. All sensors must be registered in the InteractiveScene.
        """
        if task.contact_samples:
            raise RuntimeError("do not mix contact history and individual substep sampling")
        histories = []
        for sensor in self.contacts:
            if sensor.cfg.update_period != 0.0 or sensor.cfg.history_length < task.cfg.decimation:
                raise ValueError("contacts require per-physics-step updates and a full control history")
            history = to_numpy(sensor.data.normal_force_matrix_w_history)
            if (history.ndim != 5 or history.shape[0] != task.num_envs
                    or history.shape[1] < task.cfg.decimation or history.shape[-1] != 3
                    or history.size == 0):
                raise RuntimeError("invalid or unavailable filtered contact history")
            histories.append(history[:, :task.cfg.decimation].reshape(
                task.num_envs, task.cfg.decimation, -1, 3))
        if not all(np.isfinite(value).all() for value in histories):
            raise ValueError("invalid contact history")
        combined = np.concatenate(histories, axis=2)
        for index in reversed(range(task.cfg.decimation)):
            task.latch_contacts(combined[:, index])
