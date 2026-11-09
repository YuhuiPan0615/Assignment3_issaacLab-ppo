"""Isaac Lab 3 sensor factory shared by the environment and acceptance tests."""

from .contact import make_contact_configs
from .lidar import actual_ray_origins, make_lidar_config


def make_sensor_configs(cfg, chassis_path, body_paths, scan_mesh_path, obstacle_filters,
                        mount_offset=(0.0, 0.0, 0.20), debug_vis=False):
    lidar = make_lidar_config(cfg, chassis_path, scan_mesh_path, mount_offset, debug_vis)
    contacts = make_contact_configs(cfg, body_paths, chassis_path, obstacle_filters, debug_vis)
    return lidar, contacts
