"""Obstacle-only contact configuration for chassis and both wheels."""


def make_obstacle_filter_paths(obstacle_root, obstacle_names):
    """Return one filter expression per obstacle rigid body.

    A single regular expression that resolves to several rigid bodies is not a
    valid filtered-contact partner. Environment wildcards
    may remain in the root, but every obstacle name must have its own entry.
    """
    root = str(obstacle_root).rstrip("/")
    names = tuple(str(name) for name in obstacle_names)
    if not root.startswith("/World/") or not names:
        raise ValueError("an absolute obstacle root and at least one name are required")
    if len(set(names)) != len(names) or any(not name or "/" in name for name in names):
        raise ValueError("obstacle names must be unique, non-empty prim names")
    return [f"{root}/{name}" for name in names]


def make_contact_configs(cfg, body_paths, chassis_path, obstacle_filters, debug_vis=False):
    from isaaclab.sensors import ContactSensorCfg

    required = {"chassis", "left_wheel", "right_wheel"}
    if set(body_paths) != required:
        raise ValueError("three explicitly named chassis/wheel bodies are required")
    if len(set(body_paths.values())) != 3 or body_paths["chassis"] != chassis_path:
        raise ValueError("contact bodies must be distinct and match the LiDAR chassis")
    if not obstacle_filters:
        raise ValueError("obstacle-only contact filters are required")
    return {
        name: ContactSensorCfg(
            prim_path=path,
            update_period=0.0,
            history_length=cfg.decimation,
            filter_prim_paths_expr=list(obstacle_filters),
            track_air_time=False,
            debug_vis=bool(debug_vis),
        )
        for name, path in body_paths.items()
    }
