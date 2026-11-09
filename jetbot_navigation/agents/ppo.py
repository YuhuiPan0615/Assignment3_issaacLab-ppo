"""PPO policies and training for Jetbot navigation.

The actor maps a 23-element observation to two normalized velocity commands.
Policy and update code can run independently of Isaac Lab.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class PPOConfig:
    """Policy architecture, rollout size, and PPO, GRU, and RND settings."""

    observation_dim: int = 23
    action_dim: int = 2
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "elu"
    steps_per_env: int = 64
    update_epochs: int = 5
    minibatches: int = 8
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.005
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    initial_log_std: float = -0.5
    curiosity_coefficient: float = 0.0
    curiosity_decay: float = 0.999
    curiosity_loss_coefficient: float = 1.0
    recurrent: bool = False
    recurrent_hidden_size: int = 128

    def __post_init__(self) -> None:
        if self.observation_dim != 23 or self.action_dim != 2:
            raise ValueError("Jetbot PPO requires the fixed 23D observation and 2D action contract")
        if not self.hidden_dims or any(size <= 0 for size in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive layer sizes")
        if self.steps_per_env < 2 or self.update_epochs < 1 or self.minibatches < 1:
            raise ValueError("rollout and update counts must be positive")
        for name in ("learning_rate", "gamma", "gae_lambda", "clip_ratio", "value_clip",
                     "value_coefficient", "max_grad_norm", "target_kl"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")
        if self.entropy_coefficient < 0 or not math.isfinite(self.entropy_coefficient):
            raise ValueError("entropy_coefficient must be finite and nonnegative")
        if self.curiosity_coefficient < 0 or not math.isfinite(self.curiosity_coefficient):
            raise ValueError("curiosity_coefficient must be finite and nonnegative")
        if not 0 < self.curiosity_decay <= 1:
            raise ValueError("curiosity_decay must be in (0, 1]")
        if self.curiosity_loss_coefficient <= 0:
            raise ValueError("curiosity_loss_coefficient must be positive")
        if not isinstance(self.recurrent, bool) or self.recurrent_hidden_size < 1:
            raise ValueError("recurrent must be boolean and recurrent_hidden_size positive")


def _activation(name: str) -> type[nn.Module]:
    activations = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported activation: {name}") from error


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int,
         activation: str, output_gain: float) -> nn.Sequential:
    activation_type = _activation(activation)
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        linear = nn.Linear(previous, width)
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, activation_type()))
        previous = width
    output = nn.Linear(previous, output_dim)
    nn.init.orthogonal_(output.weight, gain=output_gain)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Feedforward Gaussian actor and separate state-value critic."""

    _ACTION_EPSILON = 1.0e-6

    def __init__(self, cfg: PPOConfig):
        super().__init__()
        self.cfg = cfg
        self.actor = _mlp(
            cfg.observation_dim, cfg.hidden_dims, cfg.action_dim, cfg.activation, 0.01
        )
        self.critic = _mlp(
            cfg.observation_dim, cfg.hidden_dims, 1, cfg.activation, 1.0
        )
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), cfg.initial_log_std))

    def _distribution(self, observations: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(observations)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @classmethod
    def _squashed_log_prob(
        cls, distribution: torch.distributions.Normal, raw_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the tanh change-of-variables correction to the action density."""
        correction = torch.log(1.0 - actions.square() + cls._ACTION_EPSILON)
        return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)

    def act(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample bounded actions and return their log probabilities and values."""
        distribution = self._distribution(observations)
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_prob = self._squashed_log_prob(distribution, raw_actions, actions)
        values = self.critic(observations).squeeze(-1)
        return actions, log_prob, values

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute rollout action log probabilities, Gaussian entropy, and values."""
        bounded = actions.clamp(-1.0 + self._ACTION_EPSILON, 1.0 - self._ACTION_EPSILON)
        raw_actions = torch.atanh(bounded)
        distribution = self._distribution(observations)
        log_prob = self._squashed_log_prob(distribution, raw_actions, bounded)
        entropy = distribution.entropy().sum(dim=-1)
        values = self.critic(observations).squeeze(-1)
        return log_prob, entropy, values

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        """Use the squashed Gaussian mean for evaluation without sampling."""
        return torch.tanh(self.actor(observations))


class RecurrentActorCritic(nn.Module):
    """Actor and critic with separate GRU states for each environment."""

    _ACTION_EPSILON = 1.0e-6

    def __init__(self, cfg: PPOConfig):
        super().__init__()
        self.cfg = cfg
        size = cfg.recurrent_hidden_size
        self.actor_memory = nn.GRUCell(cfg.observation_dim, size)
        self.critic_memory = nn.GRUCell(cfg.observation_dim, size)
        for memory in (self.actor_memory, self.critic_memory):
            for name, parameter in memory.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(parameter)
                else:
                    nn.init.zeros_(parameter)
        self.actor = _mlp(size, cfg.hidden_dims, cfg.action_dim, cfg.activation, 0.01)
        self.critic = _mlp(size, cfg.hidden_dims, 1, cfg.activation, 1.0)
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), cfg.initial_log_std))

    def initial_state(self, count: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (count, self.cfg.recurrent_hidden_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    def _distribution(self, actor_features: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(actor_features)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @classmethod
    def _log_prob(cls, distribution, raw_actions, actions):
        correction = torch.log(1.0 - actions.square() + cls._ACTION_EPSILON)
        return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)

    def _step_memory(self, observations, state, masks):
        """Advance both GRUs, clearing old state where an episode has reset."""
        mask = masks.unsqueeze(-1)
        actor_state = self.actor_memory(observations, state[0] * mask)
        critic_state = self.critic_memory(observations, state[1] * mask)
        return actor_state, critic_state

    def act(self, observations, state, masks):
        next_state = self._step_memory(observations, state, masks)
        distribution = self._distribution(next_state[0])
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_prob = self._log_prob(distribution, raw_actions, actions)
        values = self.critic(next_state[1]).squeeze(-1)
        return actions, log_prob, values, next_state

    def value(self, observations, state, masks):
        next_state = self._step_memory(observations, state, masks)
        return self.critic(next_state[1]).squeeze(-1), next_state

    def bootstrap_value(self, observations, critic_state, masks):
        """Estimate next-state values from critic memory without advancing the actor."""
        next_critic = self.critic_memory(
            observations, critic_state * masks.unsqueeze(-1)
        )
        return self.critic(next_critic).squeeze(-1)

    def deterministic(self, observations, state, masks):
        next_state = self._step_memory(observations, state, masks)
        return torch.tanh(self.actor(next_state[0])), next_state

    def evaluate_sequence(self, observations, actions, initial_state, masks):
        """Replay a time-ordered rollout from its saved GRU state for a PPO update."""
        actor_state, critic_state = initial_state
        actor_features, critic_features = [], []
        for step in range(observations.shape[0]):
            actor_state, critic_state = self._step_memory(
                observations[step], (actor_state, critic_state), masks[step]
            )
            actor_features.append(actor_state)
            critic_features.append(critic_state)
        actor_features = torch.stack(actor_features)
        critic_features = torch.stack(critic_features)
        distribution = self._distribution(actor_features)
        bounded = actions.clamp(-1.0 + self._ACTION_EPSILON, 1.0 - self._ACTION_EPSILON)
        raw_actions = torch.atanh(bounded)
        log_prob = self._log_prob(distribution, raw_actions, bounded)
        entropy = distribution.entropy().sum(dim=-1)
        values = self.critic(critic_features).squeeze(-1)
        return log_prob, entropy, values


class RandomNetworkDistillation(nn.Module):
    """Measure observation novelty by predicting a fixed random network's output."""

    def __init__(self, observation_dim: int, feature_dim: int = 32):
        super().__init__()
        self.target = _mlp(observation_dim, (128, 128), feature_dim, "relu", 1.0)
        self.predictor = _mlp(observation_dim, (128, 128), feature_dim, "relu", 1.0)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def error(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one mean squared prediction error per observation."""
        with torch.no_grad():
            targets = self.target(observations)
        predictions = self.predictor(observations)
        return (predictions - targets).square().mean(dim=-1)


class RolloutBuffer:
    """Store one rollout on the simulator device, indexed by step and environment."""

    def __init__(self, steps: int, num_envs: int, observation_dim: int, action_dim: int,
                 device: torch.device | str, recurrent_hidden_size: int = 0):
        shape = (steps, num_envs)
        self.observations = torch.empty((*shape, observation_dim), device=device)
        self.actions = torch.empty((*shape, action_dim), device=device)
        self.log_probs = torch.empty(shape, device=device)
        self.values = torch.empty(shape, device=device)
        self.rewards = torch.empty(shape, device=device)
        self.next_observations = torch.empty((*shape, observation_dim), device=device)
        self.terminated = torch.empty(shape, dtype=torch.bool, device=device)
        self.truncated = torch.empty(shape, dtype=torch.bool, device=device)
        self.next_values = torch.empty(shape, device=device)
        self.advantages = torch.empty(shape, device=device)
        self.returns = torch.empty(shape, device=device)
        self.recurrent_masks = (
            torch.empty(shape, device=device) if recurrent_hidden_size else None
        )
        self.initial_actor_state = (
            torch.empty((num_envs, recurrent_hidden_size), device=device)
            if recurrent_hidden_size else None
        )
        self.initial_critic_state = (
            torch.empty((num_envs, recurrent_hidden_size), device=device)
            if recurrent_hidden_size else None
        )
        self.index = 0

    def set_recurrent_start(self, state: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Snapshot GRU states before the rollout so updates can replay its history."""
        if self.initial_actor_state is None or self.initial_critic_state is None:
            raise RuntimeError("this rollout is not recurrent")
        self.initial_actor_state.copy_(state[0])
        self.initial_critic_state.copy_(state[1])

    @property
    def steps(self) -> int:
        return self.observations.shape[0]

    @property
    def num_envs(self) -> int:
        return self.observations.shape[1]

    def add(self, observations: torch.Tensor, actions: torch.Tensor, log_probs: torch.Tensor,
            values: torch.Tensor, rewards: torch.Tensor, terminated: torch.Tensor,
            truncated: torch.Tensor, next_observations: torch.Tensor,
            next_values: torch.Tensor, recurrent_masks: torch.Tensor | None = None) -> None:
        """Store one transition per environment, including the behavior policy estimates."""
        if self.index >= self.steps:
            raise RuntimeError("rollout buffer is full")
        tensors = (observations, actions, log_probs, values, rewards, terminated,
                   truncated, next_observations, next_values)
        targets = (self.observations, self.actions, self.log_probs, self.values,
                   self.rewards, self.terminated, self.truncated,
                   self.next_observations, self.next_values)
        for target, value in zip(targets, tensors, strict=True):
            target[self.index].copy_(value)
        if self.recurrent_masks is not None:
            if recurrent_masks is None:
                raise ValueError("recurrent rollout requires masks")
            self.recurrent_masks[self.index].copy_(recurrent_masks)
        self.index += 1

    def finish(self, gamma: float, gae_lambda: float) -> None:
        """Compute generalized advantage estimates and value targets for the rollout."""
        if self.index != self.steps:
            raise RuntimeError("rollout must be full before computing advantages")
        gae = torch.zeros(self.num_envs, device=self.observations.device)
        for step in reversed(range(self.steps)):
            # A timeout still uses the final state's value. Both kinds of episode
            # ending stop the GAE recursion before it crosses into the next episode.
            bootstrap = (~self.terminated[step]).float()
            continue_episode = (~(self.terminated[step] | self.truncated[step])).float()
            delta = (
                self.rewards[step]
                + gamma * bootstrap * self.next_values[step]
                - self.values[step]
            )
            gae = delta + gamma * gae_lambda * continue_episode * gae
            self.advantages[step] = gae
        self.returns.copy_(self.advantages + self.values)

    def flattened(self) -> dict[str, torch.Tensor]:
        """Flatten time and environment into a batch for feedforward updates."""
        if self.index != self.steps:
            raise RuntimeError("rollout is incomplete")
        return {
            "observations": self.observations.flatten(0, 1),
            "actions": self.actions.flatten(0, 1),
            "next_observations": self.next_observations.flatten(0, 1),
            "log_probs": self.log_probs.flatten(),
            "values": self.values.flatten(),
            "advantages": self.advantages.flatten(),
            "returns": self.returns.flatten(),
        }

    def recurrent_batch(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor | tuple]:
        """Select complete time sequences and their starting states by environment."""
        if self.recurrent_masks is None or self.initial_actor_state is None:
            raise RuntimeError("this rollout is not recurrent")
        return {
            "observations": self.observations[:, env_ids],
            "next_observations": self.next_observations[:, env_ids],
            "actions": self.actions[:, env_ids],
            "log_probs": self.log_probs[:, env_ids],
            "values": self.values[:, env_ids],
            "advantages": self.advantages[:, env_ids],
            "returns": self.returns[:, env_ids],
            "masks": self.recurrent_masks[:, env_ids],
            "initial_state": (
                self.initial_actor_state[env_ids], self.initial_critic_state[env_ids]
            ),
        }


class RunLogger:
    """Append training metrics to JSON Lines and TensorBoard using the frame count."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._stream = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self.tensorboard = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    def write(self, iteration: int, frames: int, metrics: dict[str, float]) -> None:
        record = {"iteration": iteration, "frames": frames, "time": time.time(), **metrics}
        self._stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()
        for name, value in metrics.items():
            self.tensorboard.add_scalar(name, value, frames)
        self.tensorboard.flush()

    def close(self) -> None:
        self.tensorboard.close()
        self._stream.close()


class PPOTrainer:
    """Collect rollouts, update PPO, report outcomes, and persist checkpoints."""

    END_REASON = {1: "collision", 2: "success", 3: "out_of_bounds", 4: "timeout"}

    def __init__(self, env: Any, cfg: PPOConfig, run_dir: Path, seed: int = 42):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(env.device)
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.logger = RunLogger(self.run_dir)
        policy_type = RecurrentActorCritic if cfg.recurrent else ActorCritic
        self.policy = policy_type(cfg).to(self.device)
        self.curiosity = (
            RandomNetworkDistillation(cfg.observation_dim).to(self.device)
            if cfg.curiosity_coefficient > 0 else None
        )
        parameters = list(self.policy.parameters())
        if self.curiosity is not None:
            parameters.extend(self.curiosity.predictor.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate, eps=1e-5)
        self._intrinsic_second_moment = torch.tensor(0.0, device=self.device)
        self._intrinsic_initialized = False
        self.iteration = 0
        self.frames = 0
        self._episode_returns = torch.zeros(env.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(env.num_envs, device=self.device)
        self._recent_returns: list[float] = []
        self._recent_lengths: list[float] = []
        self._recent_reasons = {name: 0 for name in self.END_REASON.values()}
        if cfg.recurrent:
            self._recurrent_state = self.policy.initial_state(env.num_envs, self.device)
            self._recurrent_masks = torch.zeros(env.num_envs, device=self.device)

    def _terminal_values(self, next_observations: torch.Tensor, terminated: torch.Tensor,
                         truncated: torch.Tensor, extras: dict[str, Any],
                         recurrent_state=None) -> torch.Tensor:
        """Use the pre-reset observation and critic history to bootstrap timeouts."""
        done = terminated | truncated
        if self.cfg.recurrent:
            next_values = self.policy.bootstrap_value(
                next_observations, recurrent_state[1], (~done).float()
            )
        else:
            next_values = self.policy.value(next_observations)
        if truncated.any():
            final = extras.get("final_obs")
            if not isinstance(final, dict) or "policy" not in final:
                raise RuntimeError("timeout bootstrapping requires cfg.compute_final_obs=True")
            terminal_observations = final["policy"]
            next_values = next_values.clone()
            if self.cfg.recurrent:
                timeout_masks = torch.ones(int(truncated.sum().item()), device=self.device)
                next_values[truncated] = self.policy.bootstrap_value(
                    terminal_observations[truncated], recurrent_state[1][truncated], timeout_masks
                )
            else:
                next_values[truncated] = self.policy.value(terminal_observations[truncated])
        return next_values

    def _record_episodes(self, rewards: torch.Tensor, done: torch.Tensor) -> None:
        """Accumulate environment rewards and record outcomes only for completed episodes."""
        self._episode_returns.add_(rewards)
        self._episode_lengths.add_(1)
        if not done.any():
            return
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        self._recent_returns.extend(self._episode_returns[ids].detach().cpu().tolist())
        self._recent_lengths.extend(self._episode_lengths[ids].detach().cpu().tolist())
        reasons = self.env._last_step_end_reason[ids].detach().cpu().tolist()
        for reason in reasons:
            name = self.END_REASON.get(int(reason))
            if name is not None:
                self._recent_reasons[name] += 1
        self._episode_returns[ids] = 0.0
        self._episode_lengths[ids] = 0.0

    @torch.no_grad()
    def collect(self, observations: torch.Tensor) -> tuple[RolloutBuffer, torch.Tensor, dict[str, float]]:
        """Collect a rollout and return the observations needed to continue sampling."""
        storage = RolloutBuffer(
            self.cfg.steps_per_env, self.env.num_envs, self.cfg.observation_dim,
            self.cfg.action_dim, self.device,
            self.cfg.recurrent_hidden_size if self.cfg.recurrent else 0,
        )
        if self.cfg.recurrent:
            storage.set_recurrent_start(self._recurrent_state)
        reward_sum = 0.0
        intrinsic_sum = 0.0
        started = time.perf_counter()
        for _ in range(self.cfg.steps_per_env):
            if self.cfg.recurrent:
                rollout_masks = self._recurrent_masks
                actions, log_probs, values, next_recurrent_state = self.policy.act(
                    observations, self._recurrent_state, rollout_masks
                )
            else:
                rollout_masks = None
                actions, log_probs, values = self.policy.act(observations)
                next_recurrent_state = None
            next_obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
            next_observations = next_obs_dict["policy"]
            next_values = self._terminal_values(
                next_observations, terminated, truncated, extras, next_recurrent_state
            )
            learning_rewards = rewards
            if self.curiosity is not None:
                # Scale RND error by its running RMS. The bonus affects training
                # targets, while episode returns use the environment reward alone.
                intrinsic = self.curiosity.error(next_observations)
                moment = intrinsic.square().mean()
                if not self._intrinsic_initialized:
                    self._intrinsic_second_moment.copy_(moment)
                    self._intrinsic_initialized = True
                else:
                    self._intrinsic_second_moment.lerp_(moment, 0.01)
                normalized_intrinsic = intrinsic / (
                    self._intrinsic_second_moment.sqrt() + 1.0e-8
                )
                coefficient = self.cfg.curiosity_coefficient * (
                    self.cfg.curiosity_decay ** self.iteration
                )
                learning_rewards = rewards + coefficient * normalized_intrinsic.clamp(max=5.0)
                intrinsic_sum += normalized_intrinsic.mean().item()
            storage.add(observations, actions, log_probs, values, learning_rewards,
                        terminated, truncated, next_observations, next_values,
                        recurrent_masks=rollout_masks)
            self._record_episodes(rewards, terminated | truncated)
            reward_sum += rewards.mean().item()
            observations = next_observations
            if self.cfg.recurrent:
                self._recurrent_state = tuple(state.detach() for state in next_recurrent_state)
                self._recurrent_masks = (~(terminated | truncated)).float()
        storage.finish(self.cfg.gamma, self.cfg.gae_lambda)
        frames = self.cfg.steps_per_env * self.env.num_envs
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        metrics = {
            "rollout/mean_step_reward": reward_sum / self.cfg.steps_per_env,
            "performance/simulation_fps": frames / elapsed,
        }
        if self.curiosity is not None:
            metrics["curiosity/normalized_reward"] = intrinsic_sum / self.cfg.steps_per_env
            metrics["curiosity/coefficient"] = self.cfg.curiosity_coefficient * (
                self.cfg.curiosity_decay ** self.iteration
            )
        return storage, observations, metrics

    def update(self, storage: RolloutBuffer) -> dict[str, float]:
        """Update PPO with clipped policy and value losses on the collected rollout."""
        if self.cfg.recurrent:
            return self._update_recurrent(storage)
        batch = storage.flattened()
        advantages = batch["advantages"]
        batch["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        batch_size = advantages.numel()
        minibatch_size = max(1, batch_size // self.cfg.minibatches)
        totals = {"loss/policy": 0.0, "loss/value": 0.0, "policy/entropy": 0.0,
                  "policy/approx_kl": 0.0, "policy/clip_fraction": 0.0,
                  "loss/curiosity": 0.0}
        updates = 0
        stop = False
        for _ in range(self.cfg.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start:start + minibatch_size]
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    batch["observations"][indices], batch["actions"][indices]
                )
                log_ratio = new_log_probs - batch["log_probs"][indices]
                ratio = log_ratio.exp()
                advantage = batch["advantages"][indices]
                policy_loss = -torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * advantage,
                ).mean()
                old_values = batch["values"][indices]
                returns = batch["returns"][indices]
                clipped_values = old_values + (new_values - old_values).clamp(
                    -self.cfg.value_clip, self.cfg.value_clip
                )
                value_loss = 0.5 * torch.maximum(
                    (new_values - returns).square(), (clipped_values - returns).square()
                ).mean()
                entropy_mean = entropy.mean()
                curiosity_loss = torch.zeros((), device=self.device)
                if self.curiosity is not None:
                    curiosity_loss = self.curiosity.error(
                        batch["next_observations"][indices]
                    ).mean()
                loss = (
                    policy_loss + self.cfg.value_coefficient * value_loss
                    - self.cfg.entropy_coefficient * entropy_mean
                    + self.cfg.curiosity_loss_coefficient * curiosity_loss
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.cfg.clip_ratio).float().mean()
                values = (
                    policy_loss, value_loss, entropy_mean, approx_kl,
                    clip_fraction, curiosity_loss,
                )
                for key, value in zip(totals, values, strict=True):
                    totals[key] += value.item()
                updates += 1
                # Skip remaining minibatches when the policy has moved too far.
                if approx_kl.item() > 1.5 * self.cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        return {key: value / updates for key, value in totals.items()} | {
            "policy/log_std_mean": self.policy.log_std.mean().item(),
            "policy/update_minibatches": float(updates),
        }

    def _update_recurrent(self, storage: RolloutBuffer) -> dict[str, float]:
        """Update on whole sequences so gradients can flow through the GRUs."""
        normalized = (storage.advantages - storage.advantages.mean()) / (
            storage.advantages.std() + 1.0e-8
        )
        storage.advantages.copy_(normalized)
        environments_per_batch = max(1, self.env.num_envs // self.cfg.minibatches)
        totals = {"loss/policy": 0.0, "loss/value": 0.0, "policy/entropy": 0.0,
                  "policy/approx_kl": 0.0, "policy/clip_fraction": 0.0,
                  "loss/curiosity": 0.0}
        updates = 0
        stop = False
        for _ in range(self.cfg.update_epochs):
            permutation = torch.randperm(self.env.num_envs, device=self.device)
            for start in range(0, self.env.num_envs, environments_per_batch):
                env_ids = permutation[start:start + environments_per_batch]
                batch = storage.recurrent_batch(env_ids)
                new_log_probs, entropy, new_values = self.policy.evaluate_sequence(
                    batch["observations"], batch["actions"],
                    batch["initial_state"], batch["masks"],
                )
                log_ratio = new_log_probs - batch["log_probs"]
                ratio = log_ratio.exp()
                advantage = batch["advantages"]
                policy_loss = -torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * advantage,
                ).mean()
                old_values = batch["values"]
                returns = batch["returns"]
                clipped_values = old_values + (new_values - old_values).clamp(
                    -self.cfg.value_clip, self.cfg.value_clip
                )
                value_loss = 0.5 * torch.maximum(
                    (new_values - returns).square(), (clipped_values - returns).square()
                ).mean()
                entropy_mean = entropy.mean()
                curiosity_loss = torch.zeros((), device=self.device)
                if self.curiosity is not None:
                    curiosity_loss = self.curiosity.error(
                        batch["next_observations"].flatten(0, 1)
                    ).mean()
                loss = (
                    policy_loss + self.cfg.value_coefficient * value_loss
                    - self.cfg.entropy_coefficient * entropy_mean
                    + self.cfg.curiosity_loss_coefficient * curiosity_loss
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite recurrent PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.cfg.clip_ratio).float().mean()
                values = (
                    policy_loss, value_loss, entropy_mean, approx_kl,
                    clip_fraction, curiosity_loss,
                )
                for key, value in zip(totals, values, strict=True):
                    totals[key] += value.item()
                updates += 1
                if approx_kl.item() > 1.5 * self.cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        return {key: value / updates for key, value in totals.items()} | {
            "policy/log_std_mean": self.policy.log_std.mean().item(),
            "policy/update_minibatches": float(updates),
        }

    def _episode_metrics(self) -> dict[str, float]:
        """Summarize and clear episode outcomes accumulated since the last log entry."""
        completed = len(self._recent_returns)
        total_reasons = sum(self._recent_reasons.values())
        metrics = {
            "episode/completed": float(completed),
            "episode/mean_return": float(np.mean(self._recent_returns)) if completed else 0.0,
            "episode/mean_length": float(np.mean(self._recent_lengths)) if completed else 0.0,
        }
        for reason, count in self._recent_reasons.items():
            metrics[f"outcome/{reason}_count"] = float(count)
            metrics[f"outcome/{reason}_rate"] = count / total_reasons if total_reasons else 0.0
        self._recent_returns.clear()
        self._recent_lengths.clear()
        for reason in self._recent_reasons:
            self._recent_reasons[reason] = 0
        return metrics

    def save(self, name: str | None = None) -> Path:
        """Save model and optimizer state, then replace the checkpoint in one step."""
        path = self.checkpoint_dir / (name or f"ppo_{self.iteration:06d}.pt")
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "format_version": 1,
                "algorithm": "PPO",
                "iteration": self.iteration,
                "frames": self.frames,
                "config": asdict(self.cfg),
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "curiosity": self.curiosity.state_dict() if self.curiosity is not None else None,
                "intrinsic_second_moment": self._intrinsic_second_moment,
                "intrinsic_initialized": self._intrinsic_initialized,
            }, temporary,
        )
        temporary.replace(path)
        return path

    def load(self, checkpoint: Path) -> None:
        """Restore learning state and counters from a compatible PPO checkpoint."""
        data = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if data.get("algorithm") != "PPO" or data.get("format_version") != 1:
            raise ValueError("unsupported checkpoint")
        expected = asdict(self.cfg)
        loaded = data["config"]
        loaded["hidden_dims"] = tuple(loaded["hidden_dims"])
        if loaded != expected:
            raise ValueError("checkpoint PPO configuration does not match this run")
        self.policy.load_state_dict(data["policy"])
        if self.curiosity is not None:
            if data.get("curiosity") is None:
                raise ValueError("checkpoint is missing curiosity state")
            self.curiosity.load_state_dict(data["curiosity"])
        self.optimizer.load_state_dict(data["optimizer"])
        self._intrinsic_second_moment.copy_(
            data.get("intrinsic_second_moment", torch.tensor(0.0, device=self.device))
        )
        self._intrinsic_initialized = bool(data.get("intrinsic_initialized", False))
        self.iteration = int(data["iteration"])
        self.frames = int(data["frames"])

    def train(self, iterations: int, save_interval: int = 25) -> Path:
        """Reset the simulator and train until the total iteration target is reached."""
        if iterations < 1 or save_interval < 1:
            raise ValueError("iterations and save_interval must be positive")
        observations = self.env.reset()[0]["policy"]
        try:
            while self.iteration < iterations:
                started = time.perf_counter()
                storage, observations, metrics = self.collect(observations)
                metrics.update(self.update(storage))
                self.iteration += 1
                self.frames += self.cfg.steps_per_env * self.env.num_envs
                metrics.update(self._episode_metrics())
                metrics["performance/iteration_seconds"] = time.perf_counter() - started
                self.logger.write(self.iteration, self.frames, metrics)
                print(
                    f"[PPO] iter={self.iteration:05d}/{iterations} frames={self.frames:,} "
                    f"return={metrics['episode/mean_return']:+.3f} "
                    f"success={metrics['outcome/success_rate']:.1%} "
                    f"collision={metrics['outcome/collision_rate']:.1%} "
                    f"fps={metrics['performance/simulation_fps']:.0f}", flush=True,
                )
                if self.iteration % save_interval == 0:
                    self.save()
                    self.save("latest.pt")
            return self.save("latest.pt")
        finally:
            self.logger.close()


__all__ = [
    "ActorCritic", "PPOConfig", "PPOTrainer", "RandomNetworkDistillation",
    "RecurrentActorCritic", "RolloutBuffer",
]
