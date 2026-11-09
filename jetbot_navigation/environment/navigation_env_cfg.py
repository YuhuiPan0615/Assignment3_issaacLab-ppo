from dataclasses import dataclass
import math


@dataclass
class NavigationEnvCfg:
    """Provisional SI-unit parameters, to be calibrated against the Jetbot USD."""

    physics_dt: float = 1 / 120
    decimation: int = 4
    episode_seconds: float = 120.0
    lidar_rays: int = 64
    lidar_sectors: int = 16
    lidar_range: float = 6.0
    goal_distance_scale: float = 10.0
    goal_radius: float = 0.20
    forward_speed: float = 0.20
    reverse_speed: float = 0.10
    angular_speed: float = 1.5
    linear_acceleration: float = 0.40
    angular_acceleration: float = 3.0
    wheel_radius: float = 0.0325
    wheel_base: float = 0.1125
    wheel_speed_limit: float = 12.0
    left_wheel_sign: int = 1
    right_wheel_sign: int = 1
    contact_threshold: float = 0.1
    safe_distance: float = 0.25
    room_width: float = 4.0
    room_length: float = 6.0
    boundary_tolerance: float = 0.10

    def __post_init__(self):
        positive = (
            "physics_dt", "episode_seconds", "lidar_range", "goal_distance_scale",
            "goal_radius", "forward_speed", "reverse_speed", "angular_speed",
            "linear_acceleration", "angular_acceleration", "wheel_radius",
            "wheel_base", "wheel_speed_limit", "contact_threshold", "safe_distance",
            "room_width", "room_length",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("decimation", "lidar_rays", "lidar_sectors"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.lidar_sectors != 16 or self.lidar_rays % self.lidar_sectors:
            raise ValueError("23D contract requires 16 sectors and equally sized ray groups")
        if self.reverse_speed > self.forward_speed:
            raise ValueError("reverse speed must not exceed forward speed")
        if self.left_wheel_sign not in (-1, 1) or self.right_wheel_sign not in (-1, 1):
            raise ValueError("wheel joint signs must be -1 or +1")
        if not math.isfinite(self.boundary_tolerance) or self.boundary_tolerance < 0:
            raise ValueError("boundary tolerance must be finite and nonnegative")

    @property
    def control_dt(self):
        return self.physics_dt * self.decimation

    @property
    def max_steps(self):
        return math.ceil(self.episode_seconds / self.control_dt)


# Compatibility alias for the original public name.
NavigationConfig = NavigationEnvCfg
