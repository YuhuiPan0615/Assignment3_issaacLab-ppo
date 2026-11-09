"""Version-independent identifiers for the official Isaac Sim Jetbot asset."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class JetbotAssetCfg:
    """Asset defaults; actual body and joint names are resolved after USD load."""

    nucleus_path: str = "Robots/NVIDIA/Jetbot/jetbot.usd"
    spawn_height: float = 0.05
    chassis_tokens: tuple[str, ...] = ("chassis",)
    left_wheel_tokens: tuple[str, ...] = ("left", "wheel")
    right_wheel_tokens: tuple[str, ...] = ("right", "wheel")

    @staticmethod
    def _resolve_one(names, label, tokens):
        names = list(names)
        candidates = [name for name in names if all(
            token in name.rsplit("/", 1)[-1].lower() for token in tokens
        )]
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot uniquely resolve {label}: {candidates}; available={names}")
        return candidates[0]

    def resolve_body_paths(self, rigid_paths):
        return {
            "chassis": self._resolve_one(rigid_paths, "chassis", self.chassis_tokens),
            "left_wheel": self._resolve_one(rigid_paths, "left wheel", self.left_wheel_tokens),
            "right_wheel": self._resolve_one(rigid_paths, "right wheel", self.right_wheel_tokens),
        }

    def resolve_wheel_joint_ids(self, joint_names):
        joint_names = list(joint_names)
        left = self._resolve_one(joint_names, "left wheel joint", self.left_wheel_tokens)
        right = self._resolve_one(joint_names, "right wheel joint", self.right_wheel_tokens)
        return [joint_names.index(left), joint_names.index(right)]

    def make_articulation_cfg(self, prim_path, start, start_yaw, robot_usd=None):
        """Build the Isaac Lab 3 articulation config after AppLauncher starts."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        usd_path = robot_usd or f"{ISAAC_NUCLEUS_DIR}/{self.nucleus_path}"
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(*start, self.spawn_height),
                # Isaac Lab 3 uses quaternion order (x, y, z, w).
                rot=(0.0, 0.0, math.sin(start_yaw / 2), math.cos(start_yaw / 2)),
            ),
            actuators={
                "wheels": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=None,
                    damping=None,
                )
            },
        )
