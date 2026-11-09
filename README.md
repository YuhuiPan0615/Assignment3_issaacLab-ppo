# Jetbot Navigation with PPO

Train and evaluate Jetbot navigation policies in Isaac Lab 3. The project supports feedforward Proximal Policy Optimization (PPO), optional Random Network Distillation (RND) for exploration, and recurrent PPO with a GRU.

Training and evaluation share the same navigation task. The policy receives 23 observation values: 16 LiDAR sectors from a 64-ray scan, goal distance and bearing, robot velocity, and the previous action. It outputs two normalized velocity commands, `[linear, angular]`, each in `[-1, 1]`. The simulator runs at 120 Hz and applies a new action every four physics steps (30 Hz).

## Runtime

The project is configured for Windows 11 x64 and an NVIDIA RTX 5090. The reference runtime uses Python 3.12.15, Isaac Sim 6.1.0.0, Isaac Lab `v3.0.0-EA`, PyTorch 2.11.0 with CUDA 12.8, and Warp 1.16.0.

Run the commands below in PowerShell from the repository root. `scripts/python.ps1` uses the project's `.venv`. When setting up a new environment, use [constraints-isaac30.txt](constraints-isaac30.txt) as the runtime version reference. [requirements-core.txt](requirements-core.txt) lists the core Python dependency; Isaac Sim, Isaac Lab, and the training stack, including TensorBoard, must also be installed.

## Evaluate a saved policy

Run a deterministic episode using the newest numbered checkpoint in the bundled GRU PPO + RND run:

```powershell
.\scripts\python.ps1 -m jetbot_navigation.agents.evaluate_ppo --viz none --device cuda:0 --checkpoint_dir .\runs\ppo_gru_rnd_50m\checkpoints --every 1 --max_checkpoints 1
```

The evaluator reads the saved policy configuration and writes `checkpoint_evaluation.json` in the run directory. To watch the episode, replace `--viz none` with `--viz kit --start_paused`, then click **Play** in the simulator.

## Train a policy

Run a short training job with 32 parallel environments:

```powershell
.\scripts\python.ps1 -m jetbot_navigation.agents.train_ppo --viz none --device cuda:0 --num_envs 32 --iterations 2 --steps_per_env 16 --save_interval 1
```

Training creates a timestamped directory under `runs/` and saves the final checkpoint as `checkpoints/latest.pt`. Add `--recurrent` to use a GRU or set `--curiosity_coefficient` above zero to enable RND. To resume a run, use `--resume` with a checkpoint path and the matching PPO configuration. `--iterations` is the total iteration target, including iterations already completed.

## Scene and project layout

To regenerate the USD scene from the scene configuration:

```powershell
.\scripts\python.ps1 -m scripts.build_scene
```

| Path | Contents |
| --- | --- |
| `jetbot_navigation/agents/` | PPO policies, training, and deterministic evaluation |
| `jetbot_navigation/environment/` | Isaac Lab task, observations, rewards, and episode termination |
| `jetbot_navigation/robot/` | Jetbot assets and differential-drive control |
| `jetbot_navigation/scene/` | Scene configuration and reset logic |
| `jetbot_navigation/sensors/` | LiDAR, contact sensors, and goal observations |
| `assets/` | Scene manifest and generated USD |
| `scripts/` | Python launcher and scene builder |
| `runs/` | Saved PPO checkpoints organized by run |
| `figures/` | Training plots, architecture diagrams, and scene images |
