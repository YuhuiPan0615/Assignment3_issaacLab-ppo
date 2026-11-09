"""LiDAR preprocessing and Isaac Lab 3 RayCaster configuration."""
from functools import partial
import numpy as np


def _to_numpy(value):
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def ray_angles(cfg):
    """CCW order, sector 0 centered on body +X; no duplicate 360-degree ray."""
    width = 2 * np.pi / cfg.lidar_sectors
    return -width / 2 + np.arange(cfg.lidar_rays) * 2 * np.pi / cfg.lidar_rays


def lidar_sectors(distances, cfg):
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] != cfg.lidar_rays:
        raise ValueError(f"LiDAR must have shape (N, {cfg.lidar_rays})")
    # No return (+inf) means maximum range; invalid samples are conservative.
    clean = np.nan_to_num(distances, nan=0, posinf=cfg.lidar_range, neginf=0)
    clean = np.clip(clean, 0, cfg.lidar_range)
    return clean.reshape(len(clean), cfg.lidar_sectors, -1).min(axis=2)


def horizontal_pattern(pattern_cfg, device, navigation_cfg):
    import torch
    angles = torch.as_tensor(ray_angles(navigation_cfg), dtype=torch.float32, device=device)
    starts = torch.zeros((navigation_cfg.lidar_rays, 3), device=device)
    directions = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=1)
    return starts, directions


def make_lidar_config(cfg, chassis_path, scan_mesh_path, mount_offset=(0.0, 0.0, 0.20),
                      debug_vis=False):
    """Create the single-static-mesh RayCaster used by the vectorized task."""
    from isaaclab.sensors import RayCasterCfg
    from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg

    if not chassis_path or not scan_mesh_path:
        raise ValueError("chassis and a single static scan mesh are required")
    if len(mount_offset) != 3 or not np.isfinite(mount_offset).all():
        raise ValueError("mount offset must be a finite 3D vector")
    return RayCasterCfg(
        prim_path=chassis_path,
        update_period=cfg.control_dt,
        offset=RayCasterCfg.OffsetCfg(pos=tuple(mount_offset)),
        ray_alignment="base",
        mesh_prim_paths=[scan_mesh_path],
        pattern_cfg=PatternBaseCfg(func=partial(horizontal_pattern, navigation_cfg=cfg)),
        max_distance=cfg.lidar_range,
        drift_range=(0.0, 0.0),
        ray_cast_drift_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        debug_vis=bool(debug_vis),
    )


def actual_ray_origins(lidar):
    """Reconstruct world ray starts from Isaac Lab 3 buffers, including offset."""
    if lidar.cfg.ray_alignment != "base":
        raise ValueError("origin reconstruction is verified for base alignment only")
    if any(tuple(value) != (0.0, 0.0) for value in lidar.cfg.ray_cast_drift_range.values()):
        raise ValueError("ray_cast_drift requires a separate origin and hit correction")
    data = lidar.data
    starts = _to_numpy(lidar.ray_starts)
    quaternion = _to_numpy(data.quat_w)
    position = _to_numpy(data.pos_w)
    # Isaac Lab 3 quaternion convention is (x, y, z, w).
    vector = quaternion[:, None, :3]
    scalar = quaternion[:, None, 3:]
    rotated = starts + 2 * np.cross(vector, np.cross(vector, starts) + scalar * starts)
    return rotated + position[:, None, :]
