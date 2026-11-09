"""Explicit Gym registration for the Isaac environment.

Registration is a function rather than an import side effect so the lightweight
NumPy package can still be imported on machines without Isaac Sim.
"""

TASK_ID = "Isaac-Jetbot-Navigation-Direct-v0"


def register_isaac_env():
    """Register once after AppLauncher has initialized Isaac Sim."""
    import gymnasium as gym

    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=(
                "jetbot_navigation.environment.isaac_env:JetbotNavigationIsaacEnv"
            ),
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "jetbot_navigation.environment.isaac_env:"
                    "JetbotNavigationIsaacEnvCfg"
                )
            },
        )
    return TASK_ID


__all__ = ["TASK_ID", "register_isaac_env"]
