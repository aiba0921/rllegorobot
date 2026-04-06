# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  # G1に合わせ、データを少し多めに収集して安定化
    max_iterations = 2000   # 学習回数を確保
    save_interval = 50
    experiment_name = "rl_legorobot"
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,   # 入力値を自動で正規化してNaN耐性を高める
        critic_obs_normalization=True,
        # G1 Flatに近い構成。32,32より圧倒的に表現力が上がり、ノイズに強くなります。
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,         # 探索を促すためG1(0.008)よりわずかに高く設定
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,      # 急激な変化を防ぐため、1e-3から少し下げて慎重に学習
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
