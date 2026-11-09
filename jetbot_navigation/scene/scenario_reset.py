"""Scenario reset policy kept separate from fixed scene geometry."""


def sample_reset_poses(scene, count, rng, position_jitter=0.0, yaw_jitter=0.0, attempts=100):
    """Sample collision-free reset poses using a validated fixed scene."""
    scene.validate()
    return scene.sample_reset(count, rng, position_jitter, yaw_jitter, attempts)
