# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import argparse
import math
import sys
import os
import time
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter

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

    # メトリクス用TensorBoardライターの設定
    metrics_log_dir = os.path.join(log_dir, "play_metrics")
    os.makedirs(metrics_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=metrics_log_dir)

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

    # メトリクス追跡の初期化
    total_agents = env.num_envs
    removed_agents = set()          # 転倒・停止したエージェントを追跡
    agent_last_fall_time = [None] * total_agents   # 各エージェントの最終転倒時刻
    fall_time_buffer = 4.0                          # 転倒直前4秒分のデータを除外
    previous_velocities = torch.zeros((total_agents, 3), device=env.unwrapped.device)
    no_movement_start_time = [None] * total_agents  # 動きが停止した開始時刻

    sim_start_time = time.time()
    last_log_time = sim_start_time
    timestep = 0

    print(f"[INFO] 再生を開始します。目標環境数: {env_cfg.scene.num_envs}")
    
    while simulation_app.is_running():
        step_start_time = time.time()
        
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, done, info = env.step(actions)

        current_time = time.time()
        elapsed_time = current_time - sim_start_time

        # --- 転倒エージェントの記録 ---
        for idx, is_done in enumerate(done):
            if is_done and idx not in removed_agents:
                removed_agents.add(idx)
                agent_last_fall_time[idx] = current_time

        # --- 長時間停止したエージェントを除外 ---
        for idx in range(total_agents):
            if idx not in removed_agents:
                actual_velocity = obs[idx, 0:3]
                velocity_change = torch.abs(actual_velocity - previous_velocities[idx])
                if velocity_change[0].item() < 0.1 and velocity_change[2].item() < 0.1:
                    if no_movement_start_time[idx] is None:
                        no_movement_start_time[idx] = current_time
                    elif current_time - no_movement_start_time[idx] >= 15.0:
                        removed_agents.add(idx)
                        agent_last_fall_time[idx] = current_time
                else:
                    no_movement_start_time[idx] = None
                previous_velocities[idx] = actual_velocity

        # --- 転倒エージェント数をログに記録 ---
        writer.add_scalar("Metrics/Fallen_Agents", len(removed_agents), elapsed_time)

        # --- 転倒率の計算とログ記録 ---
        fall_rate = len(removed_agents) / total_agents * 100
        writer.add_scalar("Metrics/Fall_Rate", fall_rate, elapsed_time)

        # --- 追従率と速度差の計算 ---
        step_follow_ratios_x = []
        step_follow_ratios_z = []
        step_speed_differences = []
        policy_tensor = info.get("observations", {}).get("policy", obs)
        for agent_id, observation in enumerate(policy_tensor):
            # 転倒・停止したエージェントや転倒直前4秒分のデータを除外
            if agent_id in removed_agents or (
                agent_last_fall_time[agent_id] is not None
                and current_time - agent_last_fall_time[agent_id] < fall_time_buffer
            ):
                continue

            # 実際の速度と目標速度の取得
            # observation[0:2]: 実際の速度 (vx, vy)
            # observation[5]:   実際の角速度 (wz)
            # observation[9:11]: 目標速度 (vx, vy)
            # observation[11]:  目標角速度 (wz)
            actual_velocity = observation[0:2]
            target_velocity = observation[9:11]
            actual_speed = torch.norm(actual_velocity)
            target_speed = torch.norm(target_velocity)

            # X軸追従率の計算
            follow_ratio_x = (actual_speed / target_speed * 100).item() if target_speed > 0 else 0.0
            step_follow_ratios_x.append(follow_ratio_x)

            # 速度差の計算
            speed_difference = (actual_speed - target_speed).item()
            step_speed_differences.append(speed_difference)

            # Z軸追従率の計算 (周期的な角速度差を考慮)
            actual_z = observation[5].item()
            target_z = observation[11].item()
            z_range = math.pi
            z_deviation = abs((actual_z - target_z + z_range) % (2 * z_range) - z_range)
            z_follow_ratio = max((1 - z_deviation / z_range) * 100, 0.0)
            step_follow_ratios_z.append(z_follow_ratio)

        # --- 各ステップの平均メトリクスをTensorBoardに記録 ---
        if step_follow_ratios_x:
            avg_follow_ratio_x = sum(step_follow_ratios_x) / len(step_follow_ratios_x)
            writer.add_scalar("Metrics/Average_Follow_Ratio_X", avg_follow_ratio_x, elapsed_time)

        if step_follow_ratios_z:
            avg_follow_ratio_z = sum(step_follow_ratios_z) / len(step_follow_ratios_z)
            writer.add_scalar("Metrics/Average_Follow_Ratio_Z", avg_follow_ratio_z, elapsed_time)

        if step_speed_differences:
            avg_speed_difference = sum(step_speed_differences) / len(step_speed_differences)
            writer.add_scalar("Metrics/Average_Speed_Difference", avg_speed_difference, elapsed_time)

        # --- 10ステップごとに目標値と実際の値を時系列で記録 ---
        if timestep % 10 == 0:
            active_observations = [
                policy_tensor[i] for i in range(total_agents) if i not in removed_agents
            ]
            if active_observations:
                actual_values_x = [obs_i[0].item() for obs_i in active_observations]
                target_values_x = [obs_i[9].item() for obs_i in active_observations]
                actual_values_z = [obs_i[5].item() for obs_i in active_observations]
                target_values_z = [obs_i[11].item() for obs_i in active_observations]

                writer.add_scalar("TimeSeries/Actual_X", sum(actual_values_x) / len(actual_values_x), elapsed_time)
                writer.add_scalar("TimeSeries/Target_X", sum(target_values_x) / len(target_values_x), elapsed_time)
                writer.add_scalar("TimeSeries/Actual_Z", sum(actual_values_z) / len(actual_values_z), elapsed_time)
                writer.add_scalar("TimeSeries/Target_Z", sum(target_values_z) / len(target_values_z), elapsed_time)

        timestep += 1

        # --- 全エージェントが転倒・停止した場合はシミュレーション終了 ---
        if len(removed_agents) >= total_agents:
            print(f"[INFO] 全エージェントが転倒または停止しました ({elapsed_time:.2f}秒)。")
            break

        # --- 10秒ごとに経過時間をコンソールに出力 ---
        if current_time - last_log_time >= 10:
            print(f"[INFO] 経過時間: {elapsed_time:.2f}秒 | 転倒エージェント数: {len(removed_agents)}/{total_agents} | 転倒率: {fall_rate:.1f}%")
            last_log_time = current_time

        # --- 転倒したエージェントのアクションをゼロに設定して再ステップ ---
        for idx in removed_agents:
            actions[idx].zero_()
        obs, _, _, _ = env.step(actions)

        if args_cli.real_time:
            time_spent = time.time() - step_start_time
            sleep_time = dt - time_spent
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    writer.close()
    print(f"[INFO] メトリクスを {metrics_log_dir} に保存しました。")

if __name__ == "__main__":
    main()
    simulation_app.close()
