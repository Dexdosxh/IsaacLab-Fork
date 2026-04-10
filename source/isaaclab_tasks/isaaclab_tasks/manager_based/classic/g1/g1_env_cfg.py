# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.g1.mdp as mdp

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG, G1_CFG  # isort:skip


##
# Scene definition
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensor
    robot.spawn = robot.spawn.replace(activate_contact_sensors=True)
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link", history_length=3, track_air_time=True)

    # Lichter
    light = AssetBaseCfg(
        prim_path="/World/sun_light", 
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Position Action für den G1 PD-Controller
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True, 
        scale=0.5, 
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # --- BASIS ---
    progress = RewTerm(func=mdp.hybrid_forward_speed, weight=1.0, params={"target_speed": 1.0})
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.2, params={"threshold": 0.93})
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01) # -0.01
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0)
    # height = RewTerm(func=mdp.track_base_height, weight=0.5, params={"target_height": 0.74})

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio,
        weight=-0.25,
        params={
            "threshold": 0.98,
            "gear_ratio": {
                "torso_joint": 88.0,       
                ".*_hip_.*": 88.0,         
                ".*_knee_joint": 139.0,    
                ".*_ankle_.*": 40.0,       
                ".*_shoulder_.*": 21.0,  
                ".*_elbow_.*": 21.0,
            },
        },
    )

    # --- ENERGY ---
    # energy = RewTerm(func=mdp.energy_consumption, weight=-0.001)
    # --- ENERGY divided --- 
    energy_arms = RewTerm(func=mdp.energy_consumption_arms, weight=-0.001)
    energy_torso = RewTerm(func=mdp.energy_consumption_torso, weight=-0.001)
    energy_legs = RewTerm(func=mdp.energy_consumption_legs, weight=-0.001)

    joule_heating = RewTerm(
        func=mdp.joule_heating_energy,
        weight=-0.00001,
        params={
            "gear_ratio": {
                "torso_joint": 88.0,       
                ".*_hip_.*": 88.0,         
                ".*_knee_joint": 139.0,    
                ".*_ankle_.*": 40.0,       
                ".*_shoulder_.*": 21.0,  
                ".*_elbow_.*": 21.0,    
            }
        },
    )


    # --- TORQUE / FATIGUE ---
    per_joint_fatigue = RewTerm(
        func=mdp.joint_torque_fatigue_penalty_per_joint_uniform,
        weight=-0.01,
        params={
            "exponent": 2,
            "buildup_rate": 1.0,
            "recovery_rate": 0.5, 
            "tau_max": {
                "torso_joint": 88.0,       
                ".*_hip_.*": 88.0,         
                ".*_knee_joint": 139.0,    
                ".*_ankle_.*": 40.0,       
                ".*_shoulder_.*": 21.0,  
                ".*_elbow_.*": 21.0,       
            },
        },
    )

    cost_off_track = RewTerm(func=mdp.off_track, weight=-1.0)

    # --- FEET ---
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_mujoco,  
        weight=0.25,                
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "threshold": 0.25,       
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )

    # feet_impact_penalty = RewTerm(
    #     func=mdp.feet_contact_limit, 
    #     weight=-0.001, 
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]), 
    #         "max_force": 600.0,
    #     },
    # )

    # Joint penetration penalty
    body_collision = RewTerm(
        func=mdp.body_collision_penalty,
        weight=-25.0,     # Stark genug um Penetration zu verhindern
        params={
            "collision_pairs": [
                # --- ARME vs. TORSO/KÖRPER ---
                # Format: (link_a, link_b, min_distance_in_meters)
                
                # Ellbogen darf nicht in den Torso
                ("left_elbow_pitch_link", "torso_link", 0.1),
                ("right_elbow_pitch_link", "torso_link", 0.1),
                
                # Ellbogen darf nicht ans Pelvis
                ("left_elbow_pitch_link", "pelvis", 0.1),
                ("right_elbow_pitch_link", "pelvis", 0.1),
                
                # Unterarm/Hand darf nicht durch den Körper
                ("left_elbow_roll_link", "torso_link", 0.05),
                ("right_elbow_roll_link", "torso_link", 0.05),
                ("left_elbow_roll_link", "pelvis", 0.05),
                ("right_elbow_roll_link", "pelvis", 0.05),
                
                # Linker Arm darf nicht zum rechten Arm
                ("left_elbow_pitch_link", "right_elbow_pitch_link", 0.10),
                
                # --- BEINE ---
                # Knie dürfen nicht zusammenstoßen
                ("left_knee_link", "right_knee_link", 0.1),
                
                # Knöchel dürfen nicht kreuzen
                ("left_ankle_pitch_link", "right_ankle_pitch_link", 0.10),
                ("left_ankle_roll_link", "right_ankle_roll_link", 0.10),
                
                # Oberschenkel dürfen nicht ineinander
                ("left_hip_pitch_link", "right_hip_pitch_link", 0.1),

                # --- HÄNDE vs. OBERSCHENKEL / BECKEN ---
                # Unterarm darf nicht in den schwingenden Oberschenkel
                ("left_elbow_roll_link", "left_hip_pitch_link", 0.04),
                ("right_elbow_roll_link", "right_hip_pitch_link", 0.04),
                
                # Überkreuz (falls er die Arme vor dem Bauch kreuzt)
                ("left_elbow_roll_link", "right_hip_pitch_link", 0.04),
                ("right_elbow_roll_link", "left_hip_pitch_link", 0.04),

                # Die Hand selbst (repräsentiert durch 'one_link') darf nicht in Becken oder Oberschenkel
                ("left_one_link", "pelvis", 0.04),
                ("right_one_link", "pelvis", 0.04),
                ("left_one_link", "left_hip_pitch_link", 0.04),
                ("right_one_link", "right_hip_pitch_link", 0.04),
            ],
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45})


@configclass
class G1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the unitree G1 Humanoid walking environment."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 16.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
        if hasattr(self.scene, "contact_forces") and self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt