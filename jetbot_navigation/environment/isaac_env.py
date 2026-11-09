"""Isaac Lab 3 DirectRLEnv shell for vectorized Jetbot navigation.

Import this module only after ``AppLauncher`` has created the simulation app.
It intentionally contains environment/physics integration but no PPO
implementation.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import omni.usd
import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from ..robot.jetbot_cfg import JetbotAssetCfg
from ..scene.navigation_scene_cfg import FixedScene, define_parallel_scan_mesh
from ..sensors.contact import make_obstacle_filter_paths
from ..sensors.factory import make_sensor_configs
from .navigation_env_cfg import NavigationEnvCfg
from .torch_mdp import (
    build_observation,
    goal_state,
    lidar_sectors,
    process_actions,
    reward_components,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAVIGATION_USD = PROJECT_ROOT / "assets" / "navigation_scene.usda"
SCAN_MESH_PATH = "/World/NavigationScanMesh"


@configclass
class JetbotNavigationIsaacEnvCfg(DirectRLEnvCfg):
    """Real-simulator configuration; callers may override num_envs/device."""

    decimation = 4
    episode_length_s = 120.0
    action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
    )
    observation_space = 23
    state_space = 0
    seed = 42

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=8.0,
        replicate_physics=True,
        filter_collisions=True,
    )

    navigation: NavigationEnvCfg = NavigationEnvCfg()
    layout: FixedScene = FixedScene()
    robot_usd: str | None = None
    lidar_debug_vis: bool = False
    contact_debug_vis: bool = False


class JetbotNavigationIsaacEnv(DirectRLEnv):
    """GPU-vectorized navigation environment for PPO training and evaluation."""

    cfg: JetbotNavigationIsaacEnvCfg

    def __init__(self, cfg: JetbotNavigationIsaacEnvCfg, render_mode: str | None = None, **kwargs):
        if abs(cfg.sim.dt - cfg.navigation.physics_dt) > 1.0e-12:
            raise ValueError("SimulationCfg.dt must match NavigationEnvCfg.physics_dt")
        if cfg.decimation != cfg.navigation.decimation:
            raise ValueError("DirectRLEnv decimation must match the navigation contract")
        cfg.layout.validate()
        super().__init__(cfg, render_mode, **kwargs)

        self._wheel_joint_ids = JetbotAssetCfg().resolve_wheel_joint_ids(self.robot.joint_names)
        self._command_velocity = torch.zeros((self.num_envs, 2), device=self.device)
        self._previous_action = torch.zeros_like(self._command_velocity)
        self._applied_action = torch.zeros_like(self._command_velocity)
        self._wheel_targets = torch.zeros_like(self._command_velocity)
        self._goals_local = torch.tensor(
            self.cfg.layout.goal, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._last_step_end_reason = torch.zeros(
            self.num_envs, dtype=torch.int8, device=self.device
        )
        self._last_obstacle_force = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_step_position = torch.zeros((self.num_envs, 2), device=self.device)
        self._last_step_yaw = torch.zeros(self.num_envs, device=self.device)
        self._last_step_velocity = torch.zeros((self.num_envs, 2), device=self.device)
        self._last_lidar_distances = torch.zeros(
            (self.num_envs, self.cfg.navigation.lidar_rays), device=self.device
        )
        self._last_lidar_sectors = torch.zeros(
            (self.num_envs, self.cfg.navigation.lidar_sectors), device=self.device
        )

    @staticmethod
    def _env_regex_path(path: str) -> str:
        prefix = "/World/envs/env_0"
        if not path.startswith(prefix):
            raise RuntimeError(f"expected body below {prefix}, received {path}")
        return path.replace(prefix, "/World/envs/env_.*", 1)

    def _setup_scene(self):
        layout = self.cfg.layout
        if not NAVIGATION_USD.is_file():
            raise FileNotFoundError(
                f"missing {NAVIGATION_USD}; run scripts.build_scene before launching Isaac Lab"
            )

        asset_cfg = JetbotAssetCfg()
        robot_cfg = asset_cfg.make_articulation_cfg(
            prim_path="/World/envs/env_.*/Robot",
            start=layout.start,
            start_yaw=layout.start_yaw,
            robot_usd=self.cfg.robot_usd,
        )
        self.robot = Articulation(robot_cfg)

        room_cfg = sim_utils.UsdFileCfg(usd_path=str(NAVIGATION_USD))
        room_cfg.func("/World/envs/env_0/Navigation", room_cfg)
        goal_cfg = sim_utils.SphereCfg(
            radius=self.cfg.navigation.goal_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.85, 0.15), emissive_color=(0.02, 0.25, 0.02)
            ),
        )
        goal_cfg.func(
            "/World/envs/env_0/Goal",
            goal_cfg,
            translation=(*layout.goal, self.cfg.navigation.goal_radius),
        )

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        rigid_paths = [
            prim.GetPath().pathString
            for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        resolved_local = asset_cfg.resolve_body_paths(rigid_paths)
        self._resolved_body_paths = {
            name: self._env_regex_path(path) for name, path in resolved_local.items()
        }

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/Ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        source_path, destination_path = "/World/envs/env_0", "/World/envs/env_{}"
        positions = cloner.grid_transforms(
            self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device
        )[0]
        clone_plan = cloner.clone_plan_from_env_0(
            source_path,
            destination_path,
            self.scene.num_envs,
            self.device,
            positions,
            global_paths=("/World/Ground",),
        )
        cloner.replicate(clone_plan, stage=self.scene.stage)
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=["/World/Ground"])

        # RayCaster uses one static mesh, so bake every translated room into it.
        define_parallel_scan_mesh(stage, SCAN_MESH_PATH, self.scene.env_origins, layout)
        obstacle_filters = make_obstacle_filter_paths(
            "/World/envs/env_.*/Navigation/Obstacles",
            (box.name for box in layout.boxes),
        )
        lidar_cfg, contact_cfgs = make_sensor_configs(
            self.cfg.navigation,
            self._resolved_body_paths["chassis"],
            self._resolved_body_paths,
            SCAN_MESH_PATH,
            obstacle_filters,
            debug_vis=self.cfg.lidar_debug_vis,
        )
        self.lidar = RayCaster(lidar_cfg)
        self.contacts = {
            name: ContactSensor(contact_cfg.replace(debug_vis=self.cfg.contact_debug_vis))
            for name, contact_cfg in contact_cfgs.items()
        }

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["lidar"] = self.lidar
        for name, sensor in self.contacts.items():
            self.scene.sensors[f"contact_{name}"] = sensor

        self.sim.set_camera_view(eye=[6.0, -8.0, 8.0], target=[0.0, 0.0, 0.0])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 2):
            raise ValueError(f"actions must have shape ({self.num_envs}, 2)")
        self._previous_action.copy_(self._applied_action)
        wheels, velocity, applied = process_actions(
            actions, self._command_velocity, self.cfg.navigation
        )
        self._wheel_targets.copy_(wheels)
        self._command_velocity.copy_(velocity)
        self._applied_action.copy_(applied)

    def _apply_action(self) -> None:
        self.robot.actuators.target_command.set_velocity_index(
            value=self._wheel_targets, joint_ids=self._wheel_joint_ids
        )

    def _lidar_distances(self) -> torch.Tensor:
        hits = self.lidar.data.ray_hits_w.torch
        starts_b = self.lidar.ray_starts.torch
        quaternions = self.lidar.data.quat_w.torch[:, None, :].expand(
            -1, starts_b.shape[1], -1
        )
        starts_w = quat_apply(
            quaternions.reshape(-1, 4), starts_b.reshape(-1, 3)
        ).reshape_as(starts_b)
        starts_w = starts_w + self.lidar.data.pos_w.torch[:, None, :]
        return torch.linalg.vector_norm(hits - starts_w, dim=-1)

    def _navigation_state(self):
        local_position = self.robot.data.root_pos_w.torch[:, :2] - self.scene.env_origins[:, :2]
        quaternion = self.robot.data.root_quat_w.torch
        x, y, z, w = quaternion.unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        velocity = torch.stack(
            (
                self.robot.data.root_lin_vel_b.torch[:, 0],
                self.robot.data.root_ang_vel_b.torch[:, 2],
            ),
            dim=-1,
        )
        distances = self._lidar_distances()
        sectors = lidar_sectors(distances, self.cfg.navigation)
        distance, sine, cosine = goal_state(local_position, yaw, self._goals_local)
        return local_position, yaw, velocity, distances, sectors, distance, sine, cosine

    def _contact_state(self):
        body_force = []
        for sensor in self.contacts.values():
            history_proxy = sensor.data.normal_force_matrix_w_history
            if history_proxy is None:
                raise RuntimeError(
                    "filtered contact history is unavailable; verify obstacle rigid bodies and filters"
                )
            history = history_proxy.torch
            if history.numel() == 0:
                raise RuntimeError(
                    "filtered contact history is empty; verify obstacle rigid bodies and filters"
                )
            magnitude = torch.linalg.vector_norm(history, dim=-1)
            body_force.append(magnitude.reshape(self.num_envs, -1).amax(dim=-1))
        force = torch.stack(body_force, dim=-1)
        collision = force.amax(dim=-1) > self.cfg.navigation.contact_threshold
        return force, collision

    def _get_observations(self) -> dict:
        _, _, velocity, _, sectors, distance, sine, cosine = self._navigation_state()
        observation = build_observation(
            sectors,
            distance,
            sine,
            cosine,
            velocity,
            self._applied_action,
            self.cfg.navigation,
        )
        return {"policy": observation}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        position, yaw, velocity, distances, sectors, distance, _, _ = self._navigation_state()
        force, collision = self._contact_state()
        outside = (
            (position[:, 0].abs() > self.cfg.navigation.room_width / 2.0
             + self.cfg.navigation.boundary_tolerance)
            | (position[:, 1].abs() > self.cfg.navigation.room_length / 2.0
               + self.cfg.navigation.boundary_tolerance)
        )
        success = distance <= self.cfg.navigation.goal_radius
        raw_timeout = self.episode_length_buf >= self.max_episode_length - 1
        reason = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        reason = torch.where(raw_timeout, torch.full_like(reason, 4), reason)
        reason = torch.where(outside, torch.full_like(reason, 3), reason)
        reason = torch.where(success, torch.full_like(reason, 2), reason)
        reason = torch.where(collision, torch.full_like(reason, 1), reason)

        self._step_velocity = velocity
        self._step_sectors = sectors
        self._step_distance = distance
        self._step_reason = reason
        self._last_step_end_reason.copy_(reason)
        self._last_obstacle_force.copy_(force)
        self._last_collision.copy_(collision)
        self._last_step_position.copy_(position)
        self._last_step_yaw.copy_(yaw)
        self._last_step_velocity.copy_(velocity)
        self._last_lidar_distances.copy_(distances)
        self._last_lidar_sectors.copy_(sectors)
        terminated = collision | success | outside
        time_out = raw_timeout & ~terminated
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        components = reward_components(
            self._previous_distance,
            self._step_distance,
            self._previous_action,
            self._applied_action,
            self._step_sectors,
            self._step_velocity[:, 0],
            self._step_reason,
            self.cfg.navigation,
        )
        self._previous_distance.copy_(self._step_distance)
        return torch.stack(tuple(components.values()), dim=0).sum(dim=0)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        super()._reset_idx(env_ids)

        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        joint_position = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_velocity = self.robot.data.default_joint_vel.torch[env_ids].clone()
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(
            position=joint_position, env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=joint_velocity, env_ids=env_ids
        )

        self._command_velocity[env_ids] = 0.0
        self._previous_action[env_ids] = 0.0
        self._applied_action[env_ids] = 0.0
        self._wheel_targets[env_ids] = 0.0
        start = torch.tensor(self.cfg.layout.start, dtype=torch.float32, device=self.device)
        self._previous_distance[env_ids] = torch.linalg.vector_norm(
            self._goals_local[env_ids] - start, dim=-1
        )

    def diagnostic_tensors(self) -> dict[str, torch.Tensor | dict[str, str]]:
        """Snapshot for tests/UI only; callers decide if/when to copy to CPU."""
        position, yaw, velocity, distances, sectors, distance, _, _ = self._navigation_state()
        force, collision = self._contact_state()
        return {
            "position_local_m": position.detach().clone(),
            "yaw_rad": yaw.detach().clone(),
            "velocity": velocity.detach().clone(),
            "lidar_distance_m": distances.detach().clone(),
            "lidar_sector_m": sectors.detach().clone(),
            "goal_distance_m": distance.detach().clone(),
            "obstacle_force_N": force.detach().clone(),
            "collision": collision.detach().clone(),
            "last_obstacle_force_N": self._last_obstacle_force.detach().clone(),
            "last_collision": self._last_collision.detach().clone(),
            "last_position_local_m": self._last_step_position.detach().clone(),
            "last_yaw_rad": self._last_step_yaw.detach().clone(),
            "last_velocity": self._last_step_velocity.detach().clone(),
            "last_lidar_distance_m": self._last_lidar_distances.detach().clone(),
            "last_lidar_sector_m": self._last_lidar_sectors.detach().clone(),
            "applied_action": self._applied_action.detach().clone(),
            "wheel_target_radps": self._wheel_targets.detach().clone(),
            "last_end_reason": self._last_step_end_reason.detach().clone(),
            "resolved_body_paths": dict(self._resolved_body_paths),
        }
