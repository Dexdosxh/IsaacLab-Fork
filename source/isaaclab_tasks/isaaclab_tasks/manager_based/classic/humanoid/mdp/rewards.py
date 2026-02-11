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


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
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
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled, dim=-1)
        # return torch.sum(torch.clamp((env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), min=0.0), dim=-1)

class energy_consumption(ManagerTermBase):
    """Penalty for the mechanical power consumed by the joints.
    
    This is computed as the sum of absolute mechanical power: 
    P = sum(|applied_torque * joint_velocity|)
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # extract the used quantities (to enable type-hinting)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[asset_cfg.name]

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        # 1. Das Roboter-Asset aus der Szene holen
        asset: Articulation = env.scene[asset_cfg.name]
        
        # 2. Die echten physikalischen Werte holen
        # applied_torque: Das Drehmoment [Nm], das die Physik-Engine wirklich angewendet hat
        # joint_vel: Die aktuelle Geschwindigkeit [rad/s]
        tau = asset.data.applied_torque
        vel = asset.data.joint_vel
        
        # 3. Mechanische Leistung berechnen: P = |tau * vel|
        # Wir nehmen den Betrag (abs), da auch Bremsen (negative Arbeit) 
        # Energie kostet bzw. den Motor belastet.
        power = torch.clamp(tau * vel, min=0.0)  # Nur positive Leistung zählt als Energieverbrauch
        
        # 4. Summe über alle Gelenke bilden
        return torch.sum(power, dim=-1)


class joule_heating_energy(ManagerTermBase):
    """Penalty for the Joule heating in the motors, which is proportional to the square of the current.
    Assuming a simple motor model where current is proportional to torque, we can compute this as:
    Joule_heating = sum((applied_torque / gear_ratio_scaled) ** 2)
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum((asset.data.applied_torque / self.gear_ratio_scaled) ** 2, dim=-1)


def off_track(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """penalty for going off track."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_vel_w[:, 1])

def forward_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for going forward in specific speed"""
    asset: Articulation = env.scene[asset_cfg.name]
    speed = 1.5
    # vel = asset.data.root_vel_w[:,0]
    # result = torch.clamp(vel, 0.0, speed)
    vel = asset.data.root_lin_vel_b[:, 0]
    result = torch.clamp(vel, 0.0, speed)
    return result


class joint_torque_limit_penalty_ratio(ManagerTermBase):
    """ 
    Penalty for joints that have a torque too close to the max torque.
    torque_ratio_at_joint_j = torque_at_joint_j / torque_max_at_joint_j
    cost = sum(torque_ratios)
    """
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # one for every joint in every env: shape = (envs, joints)
        self.tau_max = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["tau_max"], asset.joint_names
        )
        self.tau_max[:, index_list] = torch.tensor(value_list, device=env.device)
        # protection against division with zero
        self.tau_max = torch.clamp(self.tau_max, min=1e-6)

    def __call__(self,
                env: ManagerBasedRLEnv,
                exponent: int,
                tau_max: dict[str, float],
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # get the torque
        tau = asset.data.applied_torque
        # compute the relative joint utilization
        rel_tau = torch.abs(tau) / self.tau_max
        # return cost with l(exponent)
        return torch.sum(rel_tau ** float(exponent), dim=-1)
    

class joint_torque_fatigue_penalty_global(ManagerTermBase):
    """
    Global fatigue model based on relative applied joint torque.

    Per step:
        r_j = |tau_applied_j| / tau_max_j
        s   = sum_j (r_j ** exponent)
        fatigue = max(fatigue - recovery_rate * dt, 0) + buildup_rate * s * dt

    Returns:
        fatigue  (shape: [num_envs])
    """
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # one for every joint in every env: shape = (envs, joints)
        self.tau_max = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["tau_max"], asset.joint_names
        )
        self.tau_max[:, index_list] = torch.tensor(value_list, device=env.device)
        # protection against division with zero
        self.tau_max = torch.clamp(self.tau_max, min=1e-6)
        # fatigue buffer per env
        self.fatigue = torch.zeros(env.num_envs, device=env.device)

        # timestep
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
        # get the torque
        tau = asset.data.applied_torque
        # compute the relative joint utilization
        rel_tau = torch.abs(tau) / self.tau_max
        # cost with l(exponent)
        cost = torch.sum(rel_tau ** float(exponent), dim=-1)
        # cast recovery rate and buildup rate as floats
        br = float(buildup_rate)
        rr = float(recovery_rate)

        self.fatigue = torch.clamp(self.fatigue - rr * self.dt, min=0.0)
        self.fatigue = self.fatigue + br * cost  * self.dt
        # reset on episode termination
        if hasattr(env, "termination_manager"):
            dones = env.termination_manager.dones
            if dones is not None:
                self.fatigue = torch.where(dones, torch.zeros_like(self.fatigue), self.fatigue)

        return self.fatigue
    

class joint_torque_fatigue_penalty_per_joint_uniform(ManagerTermBase):
    """
    Per-joint fatigue model based on relative applied joint torque (uniform parameters).

    For each joint j:
        r_j = |tau_applied_j| / tau_max_j
        s_j = r_j ** exponent
        F_j = max(F_j - recovery_rate * dt, 0) + buildup_rate * s_j * dt

    Returns:
        sum_j F_j  (shape: [num_envs])
    """
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # default robot asset
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]

        # tau_max per joint (shape: [num_envs, num_joints])
        self.tau_max = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["tau_max"], asset.joint_names
        )
        self.tau_max[:, index_list] = torch.tensor(value_list, device=env.device)
        self.tau_max = torch.clamp(self.tau_max, min=1e-6)

        # fatigue state per env per joint
        self.fatigue = torch.zeros(env.num_envs, asset.num_joints, device=env.device)

        # timestep
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
        # get the torque
        tau = asset.data.applied_torque
        # compute the relative joint utilization
        rel_tau = torch.abs(tau) / self.tau_max
        # instant cost of each joint
        cost_j = rel_tau ** float(exponent)
        # cast recovery rate and buildup rate as floats
        br = float(buildup_rate)
        rr = float(recovery_rate)
        # update per joint
        self.fatigue = torch.clamp(self.fatigue - rr * self.dt, min=0.0)
        self.fatigue = self.fatigue + br * cost_j * self.dt

        # reset on episode termination
        if hasattr(env, "termination_manager"):
            dones = env.termination_manager.dones
            if dones is not None:
                self.fatigue = torch.where(dones.unsqueeze(-1), torch.zeros_like(self.fatigue), self.fatigue)

        
        return torch.sum(self.fatigue, dim=-1)
    