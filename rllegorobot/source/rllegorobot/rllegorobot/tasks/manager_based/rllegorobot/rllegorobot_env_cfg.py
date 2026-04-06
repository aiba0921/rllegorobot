# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.

import math
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

import isaaclab.envs.mdp as mdp

##
# 1. カスタム観測関数・便利関数の定義（完全防具版）
##

def get_safe_quat(quat):
    """NaNやInfを含む不正なクォータニオンを安全な無回転 [1, 0, 0, 0] に置き換える"""
    is_invalid = ~torch.isfinite(quat).all(dim=-1, keepdim=True)
    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device).repeat(quat.shape[0], 1)
    safe_quat = torch.where(is_invalid, default_quat, quat)
    return torch.nn.functional.normalize(safe_quat, p=2, dim=-1)

def safe_action_rate_l2(env):
    val = env.action_manager.action - env.action_manager.prev_action
    val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    penalty = torch.sum(torch.square(val), dim=1)
    return torch.clamp(penalty, max=2000.0)

def safe_joint_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    val = torch.nan_to_num(asset.data.joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
    penalty = torch.sum(torch.square(val), dim=1)
    return torch.clamp(penalty, max=20000.0)

def safe_joint_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    val = torch.nan_to_num(asset.data.joint_acc, nan=0.0, posinf=0.0, neginf=0.0)
    penalty = torch.sum(torch.square(val), dim=1)
    return torch.clamp(penalty, max=100000.0)

def safe_joint_pos_rel(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    val = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.clamp(torch.nan_to_num(val, nan=0.0, posinf=10.0, neginf=-10.0), min=-10.0, max=10.0)

def safe_joint_vel_rel(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    val = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.clamp(torch.nan_to_num(val, nan=0.0, posinf=50.0, neginf=-50.0), min=-50.0, max=50.0)


##
# 2. カスタム報酬・終了関数の定義
##

def reward_track_command_spin(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    body_ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0]]
    current_spin_vel = torch.nan_to_num(body_ang_vel[:, 2], nan=0.0, posinf=100.0, neginf=-100.0)
    target_vel = env.command_manager.get_command("base_velocity")[:, 2] 
    error = torch.square(current_spin_vel - target_vel)
    return torch.exp(-error / 0.5)

def reward_arm_horizontal_maintaining(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    
    body_quat = get_safe_quat(body_quat)
    
    z_unit = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    up_vec = math_utils.quat_apply(body_quat, z_unit)
    
    error = torch.square(up_vec[:, 2] - 1.0)
    return torch.exp(-error / 0.05)

def penalty_low_posture(env, target_height: float, asset_cfg: SceneEntityCfg):
    """Part_1の「Y軸（高さ）」が目標値より低い場合にペナルティを与える"""
    asset = env.scene[asset_cfg.name]
    
    # 【変更】インデックスを 2(Z) から 1(Y) に変更
    current_h = torch.nan_to_num(asset.data.body_pos_w[:, asset_cfg.body_ids[0], 1], nan=0.0, posinf=0.0, neginf=0.0)
    
    error = torch.clamp(target_height - current_h, min=0.0, max=1.0)
    return torch.square(error)
        
def penalty_leg_stretch_in_air(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    """
    【統合版】空中にいる時のみ、Part_1からPart_3およびPart_3_01までの距離が
    それぞれ12cm(0.12m)から外れることを罰する（脚の伸び＋ねじれ防止）
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    
    # 3つのパーツのIDを取得 (設定ファイルで Part_1, Part_3, Part_3_01 の順に渡す前提)
    idx_1 = asset_cfg.body_ids[0]
    idx_3 = asset_cfg.body_ids[1]
    idx_3_01 = asset_cfg.body_ids[2]
    
    # ワールド座標の取得
    pos_1 = asset.data.body_pos_w[:, idx_1, :]
    pos_3 = asset.data.body_pos_w[:, idx_3, :]
    pos_3_01 = asset.data.body_pos_w[:, idx_3_01, :]
    
    # Part_1 から各足先への距離を計算
    dist_3 = torch.norm(pos_3 - pos_1, dim=-1)
    dist_3_01 = torch.norm(pos_3_01 - pos_1, dim=-1)
    
    # それぞれの 0.12m からのズレ（絶対値）を計算
    stretch_error_3 = torch.abs(dist_3 - 0.08)
    stretch_error_3_01 = torch.abs(dist_3_01 - 0.08)
    
    # 両方のズレの合計をエラーとする（両方が同時に12cmにならないとペナルティが消えない）
    total_stretch_error = stretch_error_3 + stretch_error_3_01
    
    # 空中判定（Z軸方向の力が0.1N以下なら空中）
    safe_forces = torch.nan_to_num(sensor.data.net_forces_w[:, :, 2], nan=0.0, posinf=1000.0, neginf=-1000.0)
    is_in_air = torch.max(safe_forces, dim=-1)[0] < 0.1
    
    # 空中フラグを掛ける
    return total_stretch_error * is_in_air.float()
    
def reward_jump_push_off(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]
    
    y_vel = torch.nan_to_num(asset.data.body_lin_vel_w[:, body_idx, 1], nan=0.0, posinf=0.0, neginf=0.0)
    
    # 【修正】max=3.0 を追加（現実的な最大上向き速度を例えば 3.0 m/s に制限）
    reward = torch.clamp(y_vel, min=0.0, max=0.15) 
    
    return reward
            
def penalty_feet_ground_time(env, asset_cfg: SceneEntityCfg):
    contact_sensor = env.scene[asset_cfg.name]
    safe_forces = torch.nan_to_num(contact_sensor.data.net_forces_w[:, :, 2], nan=0.0, posinf=1000.0, neginf=-1000.0)
    is_contact = safe_forces > 1.0

    if not hasattr(env, "feet_ground_time"):
        env.feet_ground_time = torch.zeros_like(is_contact, dtype=torch.float)

    env.feet_ground_time += env.step_dt * is_contact.float()
    env.feet_ground_time *= is_contact.float()
    penalty = torch.clamp(env.feet_ground_time - 0.5, min=0.0)
    return torch.sum(penalty, dim=1)

def penalty_foot_slip(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]

    safe_forces = torch.nan_to_num(contact_sensor.data.net_forces_w[:, :, 2], nan=0.0, posinf=1000.0, neginf=-1000.0)
    is_contact = safe_forces > 1.0
    
    foot_velocities = torch.nan_to_num(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], nan=0.0, posinf=0.0, neginf=0.0)
    foot_slip_speed = torch.norm(foot_velocities, dim=2)

    penalty = torch.sum(foot_slip_speed * is_contact.float(), dim=1)
    return torch.clamp(penalty, max=50.0)

def reward_termination_penalty(env):
    term_man = env.termination_manager
    is_bad_termination = term_man.terminated & ~term_man.get_term("time_out")
    return is_bad_termination.float() * -100.0

def termination_bad_tilt_arm_advanced(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    
    body_quat = get_safe_quat(body_quat)
    
    up_vec = math_utils.quat_apply(body_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1))
    return (torch.abs(up_vec[:, 2]) < 0.7)

def termination_part1_contact(env, threshold_h: float, robot_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]
    
    raw_forces = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    safe_forces = torch.clamp(torch.nan_to_num(raw_forces, nan=0.0, posinf=1000.0, neginf=0.0), max=1000.0)
    is_contact_force = torch.max(safe_forces, dim=1)[0] > 0.1
    
    raw_h = robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]
    safe_h = torch.nan_to_num(raw_h, nan=0.0, posinf=10.0, neginf=-10.0)
    is_too_low = safe_h < threshold_h
    
    return is_contact_force | is_too_low

def termination_physics_blowup(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    
    j_vel = asset.data.joint_vel
    b_lin_vel = asset.data.root_lin_vel_w
    b_ang_vel = asset.data.root_ang_vel_w

    is_nan = (torch.isnan(j_vel).any(dim=1) | 
              torch.isnan(b_lin_vel).any(dim=1) | 
              torch.isnan(b_ang_vel).any(dim=1))
              
    is_j_huge = torch.max(torch.abs(j_vel), dim=1)[0] > 50.0
    is_b_huge = torch.max(torch.abs(b_lin_vel), dim=1)[0] > 50.0
    is_a_huge = torch.max(torch.abs(b_ang_vel), dim=1)[0] > 50.0
    
    return is_nan | is_j_huge | is_b_huge | is_a_huge

def penalty_leg_crossing(env, asset_cfg: SceneEntityCfg):
    """Part_2とPart_2_01が一定距離以上に近づく（交差する）ことを罰する"""
    asset = env.scene[asset_cfg.name]
    
    idx_2 = asset_cfg.body_ids[0]
    idx_2_01 = asset_cfg.body_ids[1]
    
    pos_2 = asset.data.body_pos_w[:, idx_2, :]
    pos_2_01 = asset.data.body_pos_w[:, idx_2_01, :]
    
    # 2つのパーツの距離を計算
    dist = torch.norm(pos_2 - pos_2_01, dim=-1)
    
    # 距離が 0.08m (8cm) 以下になったら、その食い込んだ分だけペナルティ
    # ※ 0.08 の部分は実際のロボットの脚の隙間の幅に合わせて調整してください
    crossing_penalty = torch.clamp(0.04 - dist, min=0.0)
    
    return crossing_penalty

##
# 3. 環境設定クラス群
##

@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.1,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.5, 3.0),
        ),
    )

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["__07", "_"], 
        scale=0.1, 
        use_default_offset=True
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=safe_joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["__07", "_"])})
        joint_vel = ObsTerm(func=safe_joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["__07", "_"])})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale, 
        mode="reset", 
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    # 重みを強化し、ペナルティをスパルタ化（サボり対策）
# 以前のバランスに戻す
    posture_penalty = RewTerm(
        func=penalty_low_posture, 
        weight=-10.0, 
        params={"target_height": 0.1, "asset_cfg": SceneEntityCfg("robot", body_names=["Part_1"])}
    )
        
    track_spin_vel = RewTerm(func=reward_track_command_spin, weight=20.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=["Part_6"])})
    
    # 【新規】空中で足を伸ばすペナルティ
# 【修正】空中で両足を揃えて伸ばすペナルティ（統合版）
    leg_stretch_in_air = RewTerm(
        func=penalty_leg_stretch_in_air,
        weight=-5.0, 
        params={
            # Part_1を基準点として最初に書き、次に評価したい2つの足先を書く
            "asset_cfg": SceneEntityCfg("robot", body_names=["Part_1", "Part_3", "Part_3_01"]),
            "sensor_cfg": SceneEntityCfg("feet_contact")
        }
    )
        
    # 【新規】地面を蹴って飛び上がるジャンプ報酬
# 【修正】Part_1の上昇速度を評価する
    jump_push_off = RewTerm(
        func=reward_jump_push_off,
        weight=5.0, 
        params={
            # 速度を測りたい具体的な動くパーツを指定する
            "asset_cfg": SceneEntityCfg("robot", body_names=["Part_1"]) 
        }
    )
            
    # 【新規】脚の交差・干渉ペナルティ
    leg_crossing = RewTerm(
        func=penalty_leg_crossing,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Part_2", "Part_2_01"])
        }
    )
    
    feet_ground_time = RewTerm(func=penalty_feet_ground_time, weight=-2.0, params={"asset_cfg": SceneEntityCfg("feet_contact")})
    
    foot_slip = RewTerm(
        func=penalty_foot_slip, 
        weight=-1.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Part_3.*"]),
            "sensor_cfg": SceneEntityCfg("feet_contact")
        }
    )
    
    arm_horizontal = RewTerm(func=reward_arm_horizontal_maintaining, weight=1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=["Part_5"])})
    
    action_rate = RewTerm(func=safe_action_rate_l2, weight=-0.005)
    joint_vel_l2 = RewTerm(func=safe_joint_vel_l2, weight=-0.0005)
    joint_acc_l2 = RewTerm(func=safe_joint_acc_l2, weight=-0.0001)
    
    termination_penalty = RewTerm(func=reward_termination_penalty, weight=1.0)
    
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_tilt = DoneTerm(func=termination_bad_tilt_arm_advanced, params={"asset_cfg": SceneEntityCfg("robot", body_names=["Part_5"])})
    
    part1_contact = DoneTerm(
        func=termination_part1_contact, 
        params={
            "threshold_h": 0.01, 
            "robot_cfg": SceneEntityCfg("robot", body_names=["Part_1"]),
            "sensor_cfg": SceneEntityCfg("part1_contact")
        }
    )
    
    physics_blowup = DoneTerm(
        func=termination_physics_blowup, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class RllegorobotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/aibayuto/Documents/plate_07/Assembly_1_edit.usd",
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                solver_position_iteration_count=128,
                solver_velocity_iteration_count=64,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.41), 
            joint_pos={
                "__07": 0.5, "_": 0.5, 
                "__02": 0.0, "__04": 0.0,
                "__05": 0.0, "__06": 0.0, "__01": 0.0
            },
        ),
        actuators={
            "kicking_legs": ImplicitActuatorCfg(
                joint_names_expr=["__07", "_"], 
                stiffness=10.0, 
                damping=0.2, 
                effort_limit=0.325,
                velocity_limit=11.7,
                armature=0.01
            ),
            "passive_joints": ImplicitActuatorCfg(
                joint_names_expr=["__02", "__04", "__05", "__06", "__01"], 
                stiffness=0.0, 
                damping=0.5,
                effort_limit=0.0,
                velocity_limit=15.0,
                armature=0.05
            ),
        },
    )
    
    feet_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*/Part_3.*", history_length=3, track_air_time=False)
    part1_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*/Part_1.*", history_length=3, track_air_time=False)
    
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=500.0))

@configclass
class RllegorobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RllegorobotSceneCfg = RllegorobotSceneCfg(num_envs=128, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
