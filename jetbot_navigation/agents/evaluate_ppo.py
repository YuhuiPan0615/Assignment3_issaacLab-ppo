"""Evaluate saved PPO policies with deterministic actions in Isaac Lab."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import re
import traceback

import torch


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATTERN = re.compile(r"ppo_(\d+)\.pt$")


def _checkpoint_iteration(path: Path) -> int:
    match = CHECKPOINT_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"checkpoint name must be ppo_NNNNNN.pt: {path}")
    return int(match.group(1))


def _load_policy(path: Path, device: torch.device):
    """Rebuild the saved feedforward or GRU policy and load its weights."""
    from jetbot_navigation.agents.ppo import ActorCritic, PPOConfig, RecurrentActorCritic

    data = torch.load(path, map_location=device, weights_only=False)
    if data.get("algorithm") != "PPO" or data.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint: {path}")
    raw_config = data["config"]
    accepted = {field.name for field in fields(PPOConfig)}
    config = {name: value for name, value in raw_config.items() if name in accepted}
    config["hidden_dims"] = tuple(config["hidden_dims"])
    cfg = PPOConfig(**config)
    policy_type = RecurrentActorCritic if cfg.recurrent else ActorCritic
    policy = policy_type(cfg).to(device)
    policy.load_state_dict(data["policy"])
    policy.eval()
    return policy, int(data["iteration"]), int(data["frames"])


@torch.no_grad()
def evaluate_checkpoint(env, checkpoint: Path, seed: int) -> dict[str, float | int | str | bool]:
    """Run one policy and report the first environment to finish its episode."""
    policy, iteration, frames = _load_policy(checkpoint, torch.device(env.device))
    observation = env.reset(seed=seed)[0]["policy"]
    recurrent_state = (
        policy.initial_state(env.num_envs, env.device)
        if hasattr(policy, "initial_state") else None
    )
    episode_return = torch.zeros(env.num_envs, device=env.device)
    minimum_distance = observation[:, 16].clone()
    action_abs_sum = torch.zeros(2, device=env.device)
    reason = 0
    length = 0
    for length in range(1, env.max_episode_length + 1):
        if recurrent_state is None:
            actions = policy.deterministic(observation)
        else:
            actions, recurrent_state = policy.deterministic(
                observation, recurrent_state,
                torch.ones(env.num_envs, device=env.device),
            )
        action_abs_sum += actions.abs().mean(dim=0)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        observation = observations["policy"]
        episode_return += rewards
        minimum_distance = torch.minimum(minimum_distance, observation[:, 16])
        ended = terminated | truncated
        if ended.any():
            first = ended.nonzero(as_tuple=False)[0, 0]
            reason = int(env._last_step_end_reason[first].item())
            result_return = float(episode_return[first].item())
            result_minimum = float(minimum_distance[first].item())
            break
    else:
        raise RuntimeError("evaluation exceeded the environment episode limit")
    reason_name = {1: "collision", 2: "success", 3: "out_of_bounds", 4: "timeout"}.get(
        reason, "unknown"
    )
    scale = float(env.cfg.navigation.goal_distance_scale)
    return {
        "checkpoint": checkpoint.name,
        "iteration": iteration,
        "frames": frames,
        "success": reason == 2,
        "end_reason": reason_name,
        "episode_return": result_return,
        "episode_length": length,
        "minimum_goal_distance_m": result_minimum * scale,
        "mean_abs_linear_action": float((action_abs_sum[0] / length).item()),
        "mean_abs_angular_action": float((action_abs_sum[1] / length).item()),
    }


@torch.no_grad()
def evaluate_checkpoints(
    env, checkpoints: list[Path], seed: int, *, start_paused: bool = False,
) -> list[dict]:
    """Evaluate one episode per checkpoint in separate parallel environments."""
    if env.num_envs != len(checkpoints):
        raise ValueError("parallel evaluation needs one environment per checkpoint")
    if start_paused and not env.sim.has_gui:
        raise ValueError("--start_paused requires a GUI; use --viz kit")
    loaded = [_load_policy(path, torch.device(env.device)) for path in checkpoints]
    policies = [item[0] for item in loaded]
    recurrent_states = [
        policy.initial_state(1, env.device) if hasattr(policy, "initial_state") else None
        for policy in policies
    ]
    observation = env.reset(seed=seed)[0]["policy"]
    if start_paused:
        # Show the reset scene for camera adjustment before stepping physics.
        env.sim.pause()
        env.sim.render()
        print("[PPO-EVAL] Paused. Adjust the camera, then click Play (>) to start.", flush=True)
        env.sim.physics_manager.wait_for_playing()
    episode_returns = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    minimum_distances = observation[:, 16].clone()
    action_abs_sums = torch.zeros((env.num_envs, 2), device=env.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    results: list[dict | None] = [None] * env.num_envs
    scale = float(env.cfg.navigation.goal_distance_scale)
    reason_names = {1: "collision", 2: "success", 3: "out_of_bounds", 4: "timeout"}

    for _ in range(env.max_episode_length):
        # Measure distance from the observation used to select the action.
        minimum_distances[active] = torch.minimum(
            minimum_distances[active], observation[active, 16]
        )
        actions = torch.zeros((env.num_envs, 2), device=env.device)
        for index, policy in enumerate(policies):
            if recurrent_states[index] is None:
                actions[index:index + 1] = policy.deterministic(
                    observation[index:index + 1]
                )
            else:
                action, recurrent_states[index] = policy.deterministic(
                    observation[index:index + 1], recurrent_states[index],
                    torch.ones(1, device=env.device),
                )
                actions[index:index + 1] = action
        actions[~active] = 0.0
        action_abs_sums[active] += actions[active].abs()
        observations, rewards, terminated, truncated, _ = env.step(actions)
        episode_returns[active] += rewards[active]
        episode_lengths[active] += 1
        # Record each clone once; its automatic reset must not add another episode.
        ended = active & (terminated | truncated)
        for index in ended.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist():
            _, iteration, frames = loaded[index]
            reason = int(env._last_step_end_reason[index].item())
            length = int(episode_lengths[index].item())
            results[index] = {
                "checkpoint": checkpoints[index].name,
                "iteration": iteration,
                "frames": frames,
                "success": reason == 2,
                "end_reason": reason_names.get(reason, "unknown"),
                "episode_return": float(episode_returns[index].item()),
                "episode_length": length,
                "minimum_goal_distance_m": float(minimum_distances[index].item()) * scale,
                "mean_abs_linear_action": float((action_abs_sums[index, 0] / length).item()),
                "mean_abs_angular_action": float((action_abs_sums[index, 1] / length).item()),
            }
        active &= ~ended
        observation = observations["policy"]
        if not active.any():
            break
    if any(result is None for result in results):
        raise RuntimeError("one or more policies exceeded the environment episode limit")
    return [result for result in results if result is not None]


def summarize_progress(results: list[dict], distance_margin: float = 0.05,
                       return_margin: float = 1.0) -> dict[str, int | float | bool | str]:
    """Flag a plateau after three non-improving results and no successful episode."""
    if not results:
        raise ValueError("at least one checkpoint result is required")
    best_distance = float("inf")
    best_return = -float("inf")
    consecutive_without_progress = 0
    best_checkpoint = results[0]["checkpoint"]
    for result in results:
        improved = bool(result["success"])
        distance = float(result["minimum_goal_distance_m"])
        episode_return = float(result["episode_return"])
        if distance <= best_distance - distance_margin or episode_return >= best_return + return_margin:
            improved = True
        if improved:
            best_distance = min(best_distance, distance)
            best_return = max(best_return, episode_return)
            best_checkpoint = result["checkpoint"]
            consecutive_without_progress = 0
        else:
            consecutive_without_progress += 1
    successes = sum(bool(result["success"]) for result in results)
    return {
        "evaluated_checkpoints": len(results),
        "successes": successes,
        "best_checkpoint": str(best_checkpoint),
        "best_minimum_goal_distance_m": min(float(r["minimum_goal_distance_m"]) for r in results),
        "best_episode_return": max(float(r["episode_return"]) for r in results),
        "consecutive_without_progress": consecutive_without_progress,
        "plateau_detected": successes == 0 and consecutive_without_progress >= 3,
    }


def run(args) -> Path:
    """Select checkpoints and write their evaluation results next to the run."""
    from jetbot_navigation.environment.isaac_env import (
        JetbotNavigationIsaacEnv,
        JetbotNavigationIsaacEnvCfg,
    )

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoints = sorted(checkpoint_dir.glob("ppo_*.pt"), key=_checkpoint_iteration)
    checkpoints = [path for path in checkpoints if _checkpoint_iteration(path) % args.every == 0]
    if args.max_checkpoints:
        checkpoints = checkpoints[-args.max_checkpoints:]
    if not checkpoints:
        raise FileNotFoundError(f"no matching checkpoints in {checkpoint_dir}")

    cfg = JetbotNavigationIsaacEnvCfg()
    cfg.scene.num_envs = len(checkpoints)
    cfg.sim.device = args.device
    env = JetbotNavigationIsaacEnv(cfg=cfg, render_mode=None)
    try:
        results = evaluate_checkpoints(
            env, checkpoints, args.seed, start_paused=args.start_paused,
        )
        for result in results:
            print(
                f"[PPO-EVAL] {result['checkpoint']} reason={result['end_reason']} "
                f"return={result['episode_return']:+.3f} "
                f"min_goal={result['minimum_goal_distance_m']:.3f}m "
                f"steps={result['episode_length']}", flush=True,
            )
    finally:
        env.close()
    summary = summarize_progress(results)
    output = checkpoint_dir.parent / "checkpoint_evaluation.json"
    output.write_text(
        json.dumps({"results": results, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[PPO-EVAL] summary={json.dumps(summary, ensure_ascii=False)}", flush=True)
    print(f"[PPO-EVAL] output={output}", flush=True)
    return output


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--max_checkpoints", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start_paused", action="store_true",
        help="Load the initial scene, then wait for the GUI Play button before evaluation.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.every < 1 or args.max_checkpoints < 0:
        parser.error("evaluation counts must be valid positive values")
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
