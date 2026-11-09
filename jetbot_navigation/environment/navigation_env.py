from dataclasses import dataclass
import numpy as np
from ..robot.differential_drive import process_actions
from ..sensors.lidar import lidar_sectors
from ..sensors.target_state import goal_state
from .navigation_env_cfg import NavigationEnvCfg
from .observations import build_observation
from .rewards import reward_components
from .terminations import classify_end


@dataclass(frozen=True)
class SensorFrame:
    position: np.ndarray  # (N, 2), environment-local metres, reward/reset only
    yaw: np.ndarray  # (N,), reward/relative-goal computation only
    velocity: np.ndarray  # (N, 2), measured body-forward speed and yaw rate
    lidar_distances: np.ndarray  # (N, 64), body-relative angular order


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    reason: np.ndarray
    components: dict[str, np.ndarray]


class NavigationTask:
    """Vectorized task state machine, independent of physics and algorithm.

    Reset simulator first, then call reset with fresh sensor data. For each control
    cycle call begin_step, latch filtered contacts after EVERY physics substep,
    then finish_step with the post-physics frame. No automatic reset: terminal
    observations remain available for timeout bootstrapping.
    """

    def __init__(self, num_envs=1, cfg=None):
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.cfg = cfg or NavigationEnvCfg()
        self.num_envs = num_envs
        self.goals = np.zeros((num_envs, 2))
        self.previous_action = np.zeros((num_envs, 2))
        self.applied_action = np.zeros((num_envs, 2))
        self.command_velocity = np.zeros((num_envs, 2))
        self.previous_distance = np.zeros(num_envs)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.collision = np.zeros(num_envs, dtype=bool)
        self.ready = np.zeros(num_envs, dtype=bool)
        self.pending = False
        self.contact_samples = 0
        self.episode_components = {name: np.zeros(num_envs) for name in (
            "progress", "step", "smoothness", "safety", "success", "collision",
            "out_of_bounds", "timeout")}

    def _frame(self, frame):
        expected = {"position": (self.num_envs, 2), "yaw": (self.num_envs,),
                    "velocity": (self.num_envs, 2),
                    "lidar_distances": (self.num_envs, self.cfg.lidar_rays)}
        for name, shape in expected.items():
            value = np.asarray(getattr(frame, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if name != "lidar_distances" and not np.isfinite(value).all():
                raise ValueError(f"{name} contains invalid physical state")

    def observe(self, frame):
        self._frame(frame)
        sectors = lidar_sectors(frame.lidar_distances, self.cfg)
        distance, sine, cosine = goal_state(frame.position, frame.yaw, self.goals)
        return build_observation(sectors, distance, sine, cosine, frame.velocity,
                                 self.applied_action, self.cfg)

    def reset(self, frame, goals, env_ids=None):
        if self.pending:
            raise RuntimeError("cannot reset during an unfinished control cycle")
        self._frame(frame)
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any((ids < 0) | (ids >= self.num_envs)):
            raise ValueError("env_ids must contain valid integer indices")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("env_ids must be unique")
        goals = np.asarray(goals, dtype=np.float64)
        if goals.shape != (len(ids), 2) or not np.isfinite(goals).all():
            raise ValueError("goals must have shape (len(env_ids), 2) and be finite")
        if not np.allclose(frame.velocity[ids], 0, atol=1e-6):
            raise ValueError("simulator must clear root and joint velocities before reset")
        self.goals[ids] = goals
        for value in (self.previous_action, self.applied_action, self.command_velocity):
            value[ids] = 0
        self.previous_distance[ids] = np.linalg.norm(goals - frame.position[ids], axis=1)
        self.steps[ids] = 0
        self.collision[ids] = False
        self.ready[ids] = True
        for value in self.episode_components.values():
            value[ids] = 0
        return self.observe(frame)

    def begin_step(self, actions):
        if self.pending or not self.ready.all():
            raise RuntimeError("reset every ended environment and finish the previous cycle first")
        wheels, velocity, applied = process_actions(actions, self.command_velocity, self.cfg)
        if wheels.shape != (self.num_envs, 2):
            raise ValueError("one action is required per environment")
        self.previous_action[:] = self.applied_action
        self.applied_action[:] = applied
        self.command_velocity[:] = velocity
        self.collision[:] = False
        self.contact_samples = 0
        self.pending = True
        return wheels

    def latch_contacts(self, obstacle_forces):
        """Input (N, body/partner..., 3) must exclude all ground contacts."""
        if not self.pending:
            raise RuntimeError("contacts must be latched within a control cycle")
        if self.contact_samples >= self.cfg.decimation:
            raise RuntimeError("too many contact samples in a control cycle")
        forces = np.asarray(obstacle_forces, dtype=np.float64)
        if forces.ndim < 3 or forces.shape[0] != self.num_envs or forces.shape[-1] != 3:
            raise ValueError("filtered contact forces require shape (N, ..., 3)")
        if not np.isfinite(forces).all():
            raise ValueError("invalid contact readings must not silently count as no collision")
        magnitude = np.linalg.norm(forces, axis=-1).reshape(self.num_envs, -1)
        self.collision |= np.any(magnitude > self.cfg.contact_threshold, axis=1)
        self.contact_samples += 1

    def finish_step(self, frame):
        if not self.pending:
            raise RuntimeError("begin_step must precede finish_step")
        if self.contact_samples != self.cfg.decimation:
            raise RuntimeError("sample filtered contacts after every physics substep")
        self._frame(frame)
        sectors = lidar_sectors(frame.lidar_distances, self.cfg)
        distance, _, _ = goal_state(frame.position, frame.yaw, self.goals)
        next_steps = self.steps + 1
        terminated, truncated, reason = classify_end(self.collision, distance, frame.position,
                                                      next_steps, self.cfg)
        components = reward_components(self.previous_distance, distance, self.previous_action,
                                       self.applied_action, sectors, frame.velocity[:, 0],
                                       reason, self.cfg)
        observation = self.observe(frame)
        self.steps[:] = next_steps
        self.previous_distance[:] = distance
        for name, value in components.items():
            self.episode_components[name] += value
        self.ready[:] = ~(terminated | truncated)
        self.pending = False
        return StepResult(observation, sum(components.values()).astype(np.float32),
                          terminated, truncated, reason, components)
