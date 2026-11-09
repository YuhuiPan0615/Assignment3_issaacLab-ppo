"""Train or resume a PPO policy for Jetbot navigation in Isaac Lab."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import traceback


ROOT = Path(__file__).resolve().parents[2]


def run(args) -> Path:
    """Build the environment and trainer, then return the final checkpoint path."""
    # AppLauncher must start Kit before importing the environment module.
    from jetbot_navigation.agents.ppo import PPOConfig, PPOTrainer
    from jetbot_navigation.environment.isaac_env import (
        JetbotNavigationIsaacEnv,
        JetbotNavigationIsaacEnvCfg,
    )

    cfg = JetbotNavigationIsaacEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    # Timeout targets need the final observation before the automatic reset.
    cfg.compute_final_obs = True
    env = JetbotNavigationIsaacEnv(cfg=cfg, render_mode=None)
    run_name = args.run_name or datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / run_name
    ppo_cfg = PPOConfig(
        steps_per_env=args.steps_per_env,
        update_epochs=args.update_epochs,
        minibatches=args.minibatches,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coefficient=args.entropy_coefficient,
        initial_log_std=args.initial_log_std,
        curiosity_coefficient=args.curiosity_coefficient,
        curiosity_decay=args.curiosity_decay,
        recurrent=args.recurrent,
        recurrent_hidden_size=args.recurrent_hidden_size,
    )
    trainer = PPOTrainer(env, ppo_cfg, run_dir, seed=args.seed)
    try:
        if args.resume:
            trainer.load(Path(args.resume).resolve())
        checkpoint = trainer.train(args.iterations, args.save_interval)
        print(f"[PPO] checkpoint={checkpoint}", flush=True)
        return checkpoint
    finally:
        env.close()


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps_per_env", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--entropy_coefficient", type=float, default=0.005)
    parser.add_argument("--initial_log_std", type=float, default=-0.5)
    parser.add_argument("--curiosity_coefficient", type=float, default=0.0)
    parser.add_argument("--curiosity_decay", type=float, default=0.999)
    parser.add_argument("--recurrent", action="store_true")
    parser.add_argument("--recurrent_hidden_size", type=int, default=128)
    parser.add_argument("--save_interval", type=int, default=25)
    parser.add_argument("--run_name")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    positive = (
        args.num_envs, args.iterations, args.steps_per_env, args.update_epochs,
        args.minibatches, args.save_interval,
    )
    if any(value < 1 for value in positive):
        parser.error("environment and training counts must be positive")
    launcher = AppLauncher(args)
    failed = False
    try:
        run(args)
    except Exception:
        failed = True
        traceback.print_exc()
    finally:
        launcher.app.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
