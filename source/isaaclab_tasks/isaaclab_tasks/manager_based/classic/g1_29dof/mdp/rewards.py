# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def track_base_height(
    env: ManagerBasedRLEnv, target_height: float = 0.75, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining a specific base height (forces straight legs)."""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    height_error = torch.square(base_height - target_height)
    return torch.exp(-height_error / 0.01)


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled
        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class energy_consumption(ManagerTermBase):
    """Penalty for the total mechanical power consumed by the joints."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.arm_indices = [
            i for i, name in enumerate(asset.joint_names)
            if "shoulder" in name.lower() or "elbow" in name.lower()
        ]
        self.joint_cost_weights = torch.ones(1, asset.num_joints, device=env.device)
        self.joint_cost_weights[:, self.arm_indices] = 3

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        tau = asset.data.applied_torque * self.joint_cost_weights
        vel = asset.data.joint_vel
        power = torch.clamp(tau * vel, min=0.0)
        return torch.sum(power, dim=-1)


class energy_consumption_arms(ManagerTermBase):
    """Penalty for the mechanical power consumed by arm and hand joints only.

    For the 29-DOF robot the arm group covers shoulder/elbow/wrist joints and the
    hand group covers index/middle/thumb finger joints — all counted here.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        # 29-DOF: shoulder, elbow (single), wrist, index, middle, thumb fingers
        self.arm_indices = [
            i for i, name in enumerate(asset.joint_names)
            if "shoulder" in name.lower()
            or "elbow" in name.lower()
            or "wrist" in name.lower()
            or "index" in name.lower()
            or "middle" in name.lower()
            or "thumb" in name.lower()
        ]

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        tau = asset.data.applied_torque[:, self.arm_indices]
        vel = asset.data.joint_vel[:, self.arm_indices]
        power = torch.clamp(tau * vel, min=0.0)
        return torch.sum(power, dim=-1)


class energy_consumption_torso(ManagerTermBase):
    """Penalty for the mechanical power consumed by the waist joints.

    The 29-DOF robot replaces the single torso_joint with a 3-DOF waist group
    (waist_yaw_joint, waist_roll_joint, waist_pitch_joint).
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        # Match waist_yaw_joint, waist_roll_joint, waist_pitch_joint
        self.torso_indices = [
            i for i, name in enumerate(asset.joint_names)
            if "waist" in name.lower()
        ]

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        tau = asset.data.applied_torque[:, self.torso_indices]
        vel = asset.data.joint_vel[:, self.torso_indices]
        power = torch.clamp(tau * vel, min=0.0)
        return torch.sum(power, dim=-1)


class energy_consumption_legs(ManagerTermBase):
    """Penalty for the mechanical power consumed by leg joints only."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.leg_indices = [
            i for i, name in enumerate(asset.joint_names)
            if "hip" in name.lower()
            or "knee" in name.lower()
            or "ankle" in name.lower()
        ]
        boost_joints = cfg.params.get("joints", {})
        self.joint_cost_weights = torch.ones(1, len(self.leg_indices), device=env.device)
        for keyword, multiplier in boost_joints.items():
            for j, global_idx in enumerate(self.leg_indices):
                if keyword.lower() in asset.joint_names[global_idx].lower():
                    self.joint_cost_weights[:, j] = multiplier

    def __call__(
        self, env: ManagerBasedRLEnv,
        joints: dict[str, float] = {},
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        tau = asset.data.applied_torque[:, self.leg_indices]
        vel = asset.data.joint_vel[:, self.leg_indices]
        power = torch.clamp(tau * vel, min=0.0) * self.joint_cost_weights
        return torch.sum(power, dim=-1)


class joule_heating_energy(ManagerTermBase):
    """Penalty for Joule heating: sum((torque / gear_ratio_scaled)^2)."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        tau = asset.data.applied_torque
        base_heating = (tau / self.gear_ratio_scaled) ** 2
        return torch.sum(base_heating, dim=-1)


def off_track(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty for lateral deviation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_vel_w[:, 1])


def forward_speed(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_speed: float = 1.0
) -> torch.Tensor:
    """Reward for moving forward at a specific speed."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_b[:, 0]
    return torch.clamp(vel, 0.0, target_speed)


def hybrid_forward_speed(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_speed: float = 2.0
) -> torch.Tensor:
    """Linear reward up to target_speed, exponential penalty if exceeding it."""
    asset = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_b[:, 0]
    linear_reward = torch.clamp(vel, min=0.0, max=target_speed)
    overshoot = torch.clamp(vel - target_speed, min=0.0)
    penalty_multiplier = torch.exp(-torch.square(overshoot) / 0.25)
    return linear_reward * penalty_multiplier


class joint_torque_limit_penalty_ratio(ManagerTermBase):
    """Penalty for joints operating close to their torque limit."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.tau_max = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["tau_max"], asset.joint_names
        )
        self.tau_max[:, index_list] = torch.tensor(value_list, device=env.device)
        self.tau_max = torch.clamp(self.tau_max, min=1e-6)

    def __call__(self,
                env: ManagerBasedRLEnv,
                exponent: int,
                tau_max: dict[str, float],
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        tau = asset.data.applied_torque
        rel_tau = torch.abs(tau) / self.tau_max
        return torch.sum(rel_tau ** float(exponent), dim=-1)


class joint_torque_fatigue_penalty_global(ManagerTermBase):
    """Global fatigue model based on relative applied joint torque."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.tau_max = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["tau_max"], asset.joint_names
        )
        self.tau_max[:, index_list] = torch.tensor(value_list, device=env.device)
        self.tau_max = torch.clamp(self.tau_max, min=1e-6)
        self.fatigue = torch.zeros(env.num_envs, device=env.device)
        self.dt = float(env.step_dt)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        exponent: float,
        tau_max: dict[str, float],
        buildup_rate: float,
        recovery_rate: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        tau = asset.data.applied_torque
        rel_tau = torch.abs(tau) / self.tau_max
        cost = torch.sum(rel_tau ** float(exponent), dim=-1)
        self.fatigue = torch.clamp(self.fatigue - float(recovery_rate) * self.dt, min=0.0)
        self.fatigue = self.fatigue + float(buildup_rate) * cost * self.dt
        if hasattr(env, "termination_manager"):
            dones = env.termination_manager.dones
            if dones is not None:
                self.fatigue = torch.where(dones, torch.zeros_like(self.fatigue), self.fatigue)
        return self.fatigue


class joint_torque_fatigue_penalty_per_joint_uniform(ManagerTermBase):
    """Per-joint fatigue model based on relative applied joint torque."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.tau_max = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["tau_max"], asset.joint_names
        )
        self.tau_max[:, index_list] = torch.tensor(value_list, device=env.device)
        self.tau_max = torch.clamp(self.tau_max, min=1e-6)
        self.fatigue = torch.zeros(env.num_envs, asset.num_joints, device=env.device)
        self.dt = float(env.step_dt)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        exponent: float,
        tau_max: dict[str, float],
        buildup_rate: float,
        recovery_rate: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        tau = asset.data.applied_torque
        rel_tau = torch.abs(tau) / self.tau_max
        cost_j = rel_tau ** float(exponent)
        self.fatigue = torch.clamp(self.fatigue - float(recovery_rate) * self.dt, min=0.0)
        self.fatigue = self.fatigue + float(buildup_rate) * cost_j * self.dt
        if hasattr(env, "termination_manager"):
            dones = env.termination_manager.dones
            if dones is not None:
                self.fatigue = torch.where(dones.unsqueeze(-1), torch.zeros_like(self.fatigue), self.fatigue)
        return torch.sum(self.fatigue, dim=-1)


def feet_air_time(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rewards foot swing time; requires single-stance phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    return reward


def feet_air_time_2(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rewards foot air time per foot independently (double stance allowed)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    reward_per_foot = torch.clamp(air_time, max=threshold) * (~in_contact).float()
    return torch.sum(reward_per_foot, dim=-1)


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding while in contact with the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_contact_limit(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_force: float) -> torch.Tensor:
    """Penalize excessive contact forces at the feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    excess_force = torch.clamp(contact_forces - max_force, min=0.0)
    return torch.sum(excess_force, dim=-1)


def roll_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize lateral roll of the base."""
    asset = env.scene[asset_cfg.name]
    roll = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[0]
    return torch.square(roll)


class cost_of_transport_gauss(ManagerTermBase):
    """2D Gaussian reward centred on (target_speed, target_power)."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.arm_indices = [
            i for i, name in enumerate(asset.joint_names)
            if "shoulder" in name.lower() or "elbow" in name.lower()
        ]
        self.joint_cost_weights = torch.ones(1, asset.num_joints, device=env.device)
        self.joint_cost_weights[:, self.arm_indices] = 3.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_speed: float = 2.0,
        target_power: float = 150.0,
        sigma_speed: float = 0.3,
        sigma_power: float = 20.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        speed = asset.data.root_lin_vel_b[:, 0]
        tau = asset.data.applied_torque * self.joint_cost_weights
        vel = asset.data.joint_vel
        power = torch.sum(torch.clamp(tau * vel, min=0.0), dim=-1)
        speed_gauss = torch.exp(-0.5 * ((speed - target_speed) / sigma_speed) ** 2)
        power_gauss = torch.exp(-0.5 * ((power - target_power) / sigma_power) ** 2)
        return speed_gauss * power_gauss


class body_collision_penalty(ManagerTermBase):
    """Penalizes proximity between body-link pairs based on their 3-D distance.

    Prints a warning (but does not crash) if a body name is not found in the robot.
    Verify body names by inspecting asset.body_names at startup if you see warnings.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.pair_indices = []
        self.min_distances = []
        body_names = asset.body_names
        for link_a, link_b, min_dist in cfg.params["collision_pairs"]:
            try:
                idx_a = body_names.index(link_a)
                idx_b = body_names.index(link_b)
                self.pair_indices.append((idx_a, idx_b))
                self.min_distances.append(min_dist)
            except ValueError as e:
                print(f"[WARNUNG] Body nicht gefunden: {e}. Verfügbar: {body_names}")
        self.min_distances = torch.tensor(self.min_distances, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        collision_pairs: list,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        body_pos = asset.data.body_pos_w
        penalty = torch.zeros(env.num_envs, device=env.device)
        for i, (idx_a, idx_b) in enumerate(self.pair_indices):
            pos_a = body_pos[:, idx_a, :]
            pos_b = body_pos[:, idx_b, :]
            dist = torch.norm(pos_a - pos_b, dim=-1)
            violation = torch.clamp(self.min_distances[i] - dist, min=0.0)
            penalty += violation ** 2
        return penalty
