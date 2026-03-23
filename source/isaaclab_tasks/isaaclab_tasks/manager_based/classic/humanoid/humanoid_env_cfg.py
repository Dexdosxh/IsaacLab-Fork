# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
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

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from isaaclab_assets.robots.humanoid import HUMANOID_CFG  # isort:skip


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
    robot = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    #sensor
    robot.spawn = robot.spawn.replace(activate_contact_sensors=True)
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*_waist.*": 67.5,
            ".*_upper_arm.*": 67.5,
            "pelvis": 67.5,
            ".*_lower_arm": 45.0,
            ".*_thigh:0": 45.0,
            ".*_thigh:1": 135.0,
            ".*_thigh:2": 45.0,
            ".*_shin": 90.0,
            ".*_foot.*": 22.5,
        },
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
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
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

    # -----------------------------------------------------------
    # BASIS
    # -----------------------------------------------------------
    # (1) Reward for moving forward
    progress = RewTerm(func=mdp.forward_speed, weight=1.0, params={"target_speed": 2.0})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    # alive = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # (3) Reward for non-upright posture
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.2, params={"threshold": 0.93})
    # (4) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # (5) Penalty for reaching close to joint limits
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio,
        weight=-0.25,
        params={
            "threshold": 0.98,
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            },
        },
    )

    # -----------------------------------------------------------
    # ENERGY
    # -----------------------------------------------------------
    # (6) Penalty for mechanical energy consumption
    energy = RewTerm(func=mdp.energy_consumption, weight=-0.00005) # -0.00005
    upper_energy = RewTerm(func=mdp.energy_consumption_upper_body, weight=-0.0)
    lower_energy = RewTerm(func=mdp.energy_consumption_lower_body, weight=-0.0)

    # Joule Heating Energy Penalty
    joule_heating = RewTerm(
        func=mdp.joule_heating_energy,
        weight=-0.00001, # -0.00001
        params={
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            }
        },
    )

    joule_heating_lower_body = RewTerm(
        func=mdp.joule_heating_energy_lower_body,
        weight=-0.0, # -0.00001
        params={
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            }
        },
    )

    joule_heating_upper_body = RewTerm(
        func=mdp.joule_heating_energy_upper_body,
        weight=-0.0, # -0.00006
        params={
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            }
        },
    )


    # -----------------------------------------------------------
    # TORQUE
    # ----------------------------------------------------------
    # PER JOINT FATIGUE
    #(7) Per Joint Fatigue penalty for joint usage
    per_joint_fatigue = RewTerm(
        func=mdp.joint_torque_fatigue_penalty_per_joint_uniform,
        weight=-0.01,
        params={
            "exponent": 2,
            "buildup_rate": 1.0,
            "recovery_rate": 0.5, 
            "tau_max": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            },
        },
    )


    # (8) Penalty for moving in y direction
    cost_off_track = RewTerm(func=mdp.off_track, weight=-1.0)

    # (9) Feet forces rewards
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_mujoco,  
        weight=0.25,                
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot", "right_foot"]),
            "threshold": 0.7,       
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot", "right_foot"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"]),
        },
    )

    feet_impact_penalty = RewTerm(
        func=mdp.feet_contact_limit, 
        weight=-0.001, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot", "right_foot"]), 
            "max_force": 2500.0,
        },
    )

    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_upper_arm.*",
    #                 ".*_lower_arm",
    #             ],
    #         )
    #     },
    # )

    # joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2, # Relativ starke Strafe für das Spreizen
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             # Bestraft die Rotation im Becken und Oberschenkel, die nicht fürs Vorwärtsgehen (Pitch) da ist
    #             joint_names=["pelvis", ".*_thigh:1", ".*_thigh:2"], 
    #         )
    #     },
    # )

    # joint_deviation_ankles = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1, # Zwingt das Sprunggelenk in Richtung 0.0 (Fuß flach)
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot.*"])
    #     },
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.8})


@configclass
class HumanoidEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
        if hasattr(self.scene, "contact_forces") and self.scene.contact_forces is not None:
            # Der Sensor soll sich mit der Physik-Schrittweite (dt) aktualisieren
            self.scene.contact_forces.update_period = self.sim.dt
