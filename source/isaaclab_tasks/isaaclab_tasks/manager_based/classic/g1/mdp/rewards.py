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
    env: ManagerBasedRLEnv, target_height: float = 0.74, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining a specific base height (forces straight legs)."""
    asset = env.scene[asset_cfg.name]
    # z position of the base in world coordinates
    base_height = asset.data.root_pos_w[:, 2]
    
    height_error = torch.square(base_height - target_height)
    # Gauss
    result = torch.exp(-height_error / 0.01) 
    
    return result


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


class energy_consumption(ManagerTermBase):
    """Penalty for the mechanical power consumed by the joints.
    
    This is computed as the sum of absolute mechanical power: 
    P = sum(|applied_torque * joint_velocity|)
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # extract the used quantities (to enable type-hinting)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.arm_indices = [
            i for i, name in enumerate(asset.joint_names) 
            if "shoulder" in name.lower() or "elbow" in name.lower()
        ]
        self.joint_cost_weights = torch.ones(1, asset.num_joints, device=env.device)
        self.joint_cost_weights[:, self.arm_indices] = 3  # Arms cost x more energy

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

        # for ".*_upper_arm.*" and ".*_lower_arm" joints, we multiply the torque by a factor of 3 to account for
        # the higher energy consumption of arm joints compared to leg joints.
        tau = tau * self.joint_cost_weights
        
        # 3. Mechanische Leistung berechnen: P = |tau * vel|
        power = torch.clamp(tau * vel, min=0.0)  # Nur positive Leistung zählt als Energieverbrauch
        
        # 4. Summe über alle Gelenke bilden
        return torch.sum(power, dim=-1)
    

class energy_consumption_arms(ManagerTermBase):
    """Penalty for the mechanical power consumed by arm joints only."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.arm_indices = [
            i for i, name in enumerate(asset.joint_names)
            if "shoulder" in name.lower()
            or "elbow" in name.lower()
            or "five_joint" in name.lower()
            or "three_joint" in name.lower()
            or "six_joint" in name.lower()
            or "four_joint" in name.lower()
            or "zero_joint" in name.lower()
            or "one_joint" in name.lower()
            or "two_joint" in name.lower()
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
    """Penalty for the mechanical power consumed by the torso joint only."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.torso_indices = [
            i for i, name in enumerate(asset.joint_names)
            if "torso_joint" in name.lower()
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
        tau = asset.data.applied_torque
        base_heating = (tau / self.gear_ratio_scaled) ** 2

        return torch.sum(base_heating, dim=-1)


def off_track(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """penalty for going off track."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_vel_w[:, 1])

def forward_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_speed: float = 1.0
) -> torch.Tensor:
    """Reward for going forward in specific speed"""
    asset: Articulation = env.scene[asset_cfg.name]
    speed = target_speed
    vel = asset.data.root_lin_vel_b[:, 0]
    result = torch.clamp(vel, 0.0, speed)
    return result


def hybrid_forward_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_speed: float = 2.0) -> torch.Tensor:
    """Linear reward up to target_speed, exponential penalty if exceeding it."""
    asset = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_b[:, 0]
    
    # 1. Die Rampe: Linearer Anstieg bis zum Ziel (wie dein allererster Reward)
    # Stoppt exakt bei target_speed (2.0)
    linear_reward = torch.clamp(vel, min=0.0, max=target_speed)
    
    # 2. Der Fehler: Wie viel ist er zu schnell? (Alles unter 2.0 wird hier zu 0.0)
    overshoot = torch.clamp(vel - target_speed, min=0.0)
    
    # 3. Der Absturz: Exponentielle Strafe (Gauss) nur für die Überschreitung
    # Die 0.25 steuert, wie extrem der Absturz ist.
    penalty_multiplier = torch.exp(-torch.square(overshoot) / 0.25)
    
    # 4. Kombinieren: Rampe * Strafe
    result = linear_reward * penalty_multiplier
    
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
    

def feet_air_time(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rewards the agent for taking steps that are longer than a threshold, based on G1 contact sensor data."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Lese Flug- und Kontaktzeit aus dem Sensor
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    
    # Prüfe, ob genau EIN Fuß auf dem Boden ist (Single Stance)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    
    # Belohne die Zeit, aber kappe sie beim threshold
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)

    return reward


def feet_air_time_2(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rewards the agent for taking steps that are longer than a threshold.
    
    Belohnt Air Time pro Fuß unabhängig — Doppelstützphasen sind erlaubt.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    
    in_contact = contact_time > 0.0
    
    # Belohne die Air Time jedes Fußes einzeln, gekappt beim threshold
    # Nur Füße die gerade in der Luft sind bekommen Reward
    reward_per_foot = torch.clamp(air_time, max=threshold) * (~in_contact).float()
    
    # Summiere über beide Füße
    return torch.sum(reward_per_foot, dim=-1)


def feet_slide(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_contact_limit(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_force: float) -> torch.Tensor:
    """
    Penalizes using too much force on the contact points of the feet. 
    Incentivizes the agent to walk lightly and not slam its feet.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 1. Get contact forces for the specified bodies and compute their norm (total force magnitude)
    # Shape: (num_envs, num_feet)
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    # 2. ReLU / Threshold 
    # Shape: (num_envs, num_feet)
    excess_force = torch.clamp(contact_forces - max_force, min=0.0)
    # 3. Sum over both feet
    # Shape: (num_envs,)
    penalty = torch.sum(excess_force, dim=-1)
    
    return penalty


def roll_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize rolling (lateral sway) of the base."""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    # euler_xyz_from_quat gibt (roll, pitch, yaw) zurück
    roll = math_utils.euler_xyz_from_quat(quat)[0]
    return torch.square(roll)


class cost_of_transport_gauss(ManagerTermBase):
    """
    Reward that is exactly 1.0 when speed == target_speed AND power == target_power.
    Falls off as a 2D Gaussian in both dimensions.

    reward = exp(-0.5 * ((v - v*) / σ_v)²) * exp(-0.5 * ((P - P*) / σ_P)²)
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]

        # Arm-Indizes für gewichteten Energieverbrauch (wie in energy_consumption)
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

        # Geschwindigkeit
        speed = asset.data.root_lin_vel_b[:, 0]

        # Mechanische Leistung (identisch zu energy_consumption)
        tau = asset.data.applied_torque * self.joint_cost_weights
        vel = asset.data.joint_vel
        power = torch.sum(torch.clamp(tau * vel, min=0.0), dim=-1)

        # 2D Gaussglocke
        speed_gauss = torch.exp(-0.5 * ((speed - target_speed) / sigma_speed) ** 2)
        power_gauss = torch.exp(-0.5 * ((power - target_power) / sigma_power) ** 2)

        return speed_gauss * power_gauss


class body_collision_penalty(ManagerTermBase):
    """
    Bestraft Penetration zwischen Body-Paaren basierend auf dem Abstand ihrer Positionen.
    
    Verhindert:
    - Arme gehen durch den Torso
    - Beine kreuzen sich / gehen durcheinander
    - Ellbogen geht durch Hüfte
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]

        # Body-Paare und deren Mindestabstände auflösen
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
        # body_pos_w shape: (num_envs, num_bodies, 3)
        body_pos = asset.data.body_pos_w

        penalty = torch.zeros(env.num_envs, device=env.device)

        for i, (idx_a, idx_b) in enumerate(self.pair_indices):
            pos_a = body_pos[:, idx_a, :]  # (num_envs, 3)
            pos_b = body_pos[:, idx_b, :]  # (num_envs, 3)
            dist = torch.norm(pos_a - pos_b, dim=-1)  # (num_envs,)

            # Strafe: je näher, desto stärker (quadratisch)
            violation = torch.clamp(self.min_distances[i] - dist, min=0.0)
            penalty += violation ** 2

        return penalty