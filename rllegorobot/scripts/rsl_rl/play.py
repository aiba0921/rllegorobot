# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import argparse
import sys
import os
import time
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# 1. コマンドライン引数の設定
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# RSL-RL用の引数を追加 (cli_argsはローカルのファイルを想定)
import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

# 2. 引数の解析
args_cli, hydra_args = parser.parse_known_args()

# ★重要★ Hydraが混乱しないように、sys.argvを掃除する
sys.argv = [sys.argv[0]] + hydra_args

# 3. シミュレーターの起動
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""この後に他のインポートを行う"""

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper 
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rllegorobot.tasks  # noqa: F401

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Play with RSL-RL agent."""
    
    # CLI引数で設定を上書き
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed

    # ログディレクトリとチェックポイントの特定
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # 環境の作成
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ポリシーのロード
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 再生ループ
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    
    print(f"[INFO] 再生を開始します。目標環境数: {env_cfg.scene.num_envs}")
    
    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            
        if args_cli.real_time:
            time_spent = time.time() - start_time
            sleep_time = dt - time_spent
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
