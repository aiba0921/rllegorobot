# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)
def track_joint_vel_l2(
    env: ManagerBasedRLEnv, target: float, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """目標とする関節速度との差に基づいた報酬を計算します（ガウス関数形式）。"""
    # アセット（ロボット）を取得
    asset: Articulation = env.scene[asset_cfg.name]
    # 指定された関節（__05など）の現在の速度を取得
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    # 目標速度との二乗誤差を計算
    vel_error = torch.square(joint_vel - target)
    # ガウスカーネルを用いて、目標に近いほど報酬を高くする (exp(-error / std^2))
    return torch.exp(-torch.sum(vel_error, dim=1) / (std**2))
