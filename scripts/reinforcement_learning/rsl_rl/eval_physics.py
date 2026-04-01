# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# --- NEU FÜR TEACHER-STUDENT EVALUATION ---
parser.add_argument(
    "--teacher_path", type=str, default=None, help="Path to the trained teacher policy.pt to freeze legs."
)
# ------------------------------------------
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import isaaclab.utils.string as string_utils

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
import isaaclab.utils.string as string_utils
import isaaclab.utils.math as math_utils
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
# PLACEHOLDER: Extension template (do not remove this comment)
# --- NEU: DER WRAPPER FÜR EVALUATION ---
class ArmTrainingWrapper(gym.Wrapper):
    def __init__(self, env, teacher_policy_path):
        super().__init__(env)
        print(f"[INFO] Lade Teacher-Modell für Evaluation aus: {teacher_policy_path}")
        
        self.teacher_policy = torch.jit.load(teacher_policy_path).to(env.unwrapped.device)
        self.teacher_policy.eval()
        
        joint_names = env.unwrapped.scene["robot"].joint_names
        self.frozen_indices = [
            i for i, name in enumerate(joint_names) 
            if "arm" not in name.lower() and "shoulder" not in name.lower() and "elbow" not in name.lower()
        ]
        self.current_obs = None

    def step(self, action):
        with torch.no_grad():
            teacher_action = self.teacher_policy(self.current_obs)
            
        mixed_action = action.clone()
        mixed_action[:, self.frozen_indices] = teacher_action[:, self.frozen_indices]
        
        obs, rewards, dones, truncated, extras = self.env.step(mixed_action)
        self.current_obs = obs["policy"]
        return obs, rewards, dones, truncated, extras

    def reset(self, **kwargs):
        obs, extras = self.env.reset(**kwargs)
        self.current_obs = obs["policy"]
        return obs, extras
# ---------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # --- ÄNDERUNG: Episodenlänge drastisch erhöhen ---
    env_cfg.episode_length_s = 100.0

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir
    # ---------------------------------------------------------
    # --- LICHT-BOOST (STUDIO SETUP) ---
    # ---------------------------------------------------------
    env_cfg.scene.dome_light = AssetBaseCfg(
        prim_path="/World/sky",
        spawn=sim_utils.DomeLightCfg(
            # Standard Omniverse HDRI für einen klaren/leicht bewölkten Himmel
            texture_file="C:/Users/Leo/IsaacLab/pictures/sky.hdr",
            intensity=1500.0,
        ),
    )
    # ---------------------------------------------------------
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # --- NEU: WRAPPER AKTIVIEREN ---
    if args_cli.teacher_path is not None:
        if not os.path.exists(args_cli.teacher_path):
            raise FileNotFoundError(f"Teacher Modell nicht gefunden: {args_cli.teacher_path}")
        env = ArmTrainingWrapper(env, args_cli.teacher_path)
    # -------------------------------

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    # ---------------------------------------------------------
    # --- EVALUATION SETUP ---
    # ---------------------------------------------------------
    robot_entity = env.unwrapped.scene["robot"]
    joint_names = robot_entity.joint_names
    num_joints = len(joint_names)

    # Listen für schnelles Append
    history_torque = []
    history_vel = []
    history_power = []
    history_total_power = []
    
    # Listen für Joule Heating (Optional)
    history_joule = []
    history_total_joule = []
    history_forces = []
    history_speed = []
    eval_step_counter = 0
    EVAL_DURATION_STEPS = 800 
    
    # Zeitschritt holen
    dt = env.unwrapped.step_dt
    
    # -----------------------------------------------------------------
    # AUTOMATISCHE ROBOTER-ERKENNUNG FÜR GEAR RATIOS (JOULE HEATING)
    # -----------------------------------------------------------------
    # Wir prüfen anhand des ersten Gelenknamens, welcher Roboter aktiv ist
    calculate_joule_heating = True
    limit_config = {}

    if any("waist" in name for name in joint_names):
        print("[INFO] MuJoCo Humanoid erkannt. Lade entsprechende Gear Ratios.")
        limit_config = {
            "waist": 67.5, "upper_arm": 67.5, "pelvis": 67.5,
            "lower_arm": 45.0, "thigh:0": 45.0, "thigh:1": 135.0,
            "thigh:2": 45.0, "shin": 90.0, "foot": 22.5
        }
    elif any("torso_joint" in name for name in joint_names):
        print("[INFO] Unitree G1 erkannt. Lade entsprechende Gear Ratios.")
        limit_config = {
            "hip": 88.0, "knee": 139.0, "ankle": 40.0, "torso": 88.0,
            "shoulder": 21.0, "elbow": 21.0,
            # (Setze hier die echten Gear Ratios für den G1 ein, falls du sie hast, ansonsten 1.0)
        }
    else:
        print("[WARNUNG] Unbekannter Roboter. Joule Heating wird NICHT berechnet.")
        calculate_joule_heating = False

    # Gear Ratios vorbereiten (nur wenn nötig)
    if calculate_joule_heating:
        gear_ratios_list = []
        for name in joint_names:
            found_val = 1.0 # Fallback
            for key, val in limit_config.items():
                if key in name: 
                    found_val = val
                    break
            gear_ratios_list.append(found_val)

        gear_ratio_tensor = torch.tensor(gear_ratios_list, device=env.device)
        gear_ratio_scaled = gear_ratio_tensor / torch.max(gear_ratio_tensor)

    print(f"\n[INFO] Starte Physics-Evaluation für {num_joints} Gelenke ({EVAL_DURATION_STEPS} Steps)...")
    print(f"[INFO] Daten werden gesammelt und am Ende gespeichert.\n")
    # ---------------------------------------------------------
    # --- EVALUATION SETUP (END) ---
    # ---------------------------------------------------------

    # reset environment
    obs = env.get_observations()
    timestep = 0
    
    # --- NEU: TIMING & RTF SETUP ---
    rtf_start_time = None
    rtf_last_window_time = None
    print(f"[INFO] Real-Time Factor (RTF) Tracking aktiv.\n")
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()    
        
        # Startzeit exakt beim ersten Schritt festhalten
        if rtf_start_time is None:
            rtf_start_time = time.time()
            rtf_last_window_time = time.time()
            
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            
            # ---------------------------------------------------------
            # --- EVALUATION LOOP ---
            # ---------------------------------------------------------
            # --- DATEN SAMMELN ---
            if eval_step_counter < EVAL_DURATION_STEPS:
                current_torques = robot_entity.data.applied_torque 
                current_vels = robot_entity.data.joint_vel
                
                # NaNs entfernen
                if torch.isnan(current_torques).any(): current_torques = torch.nan_to_num(current_torques, nan=0.0)
                if torch.isnan(current_vels).any(): current_vels = torch.nan_to_num(current_vels, nan=0.0)

                # 1. MECHANISCHE ENERGIE (Power) - Immer berechnen
                current_power = torch.clamp(current_torques * current_vels, min=0.0)
                total_power_step = torch.sum(current_power, dim=-1)
                
                # 2. JOULE HEATING (Optional)
                if calculate_joule_heating:
                    current_joule_per_joint = (current_torques / gear_ratio_tensor) ** 2
                    total_joule_step = torch.sum(current_joule_per_joint, dim=-1)
                    
                    history_joule.append(torch.mean(current_joule_per_joint, dim=0).cpu().numpy())
                    history_total_joule.append(torch.mean(total_joule_step).cpu().numpy())

                # 3. GESCHWINDIGKEIT
                lin_vel = robot_entity.data.root_lin_vel_w
                current_speed = torch.norm(lin_vel[:, :2], dim=-1).mean().item()
                history_speed.append(current_speed)

                # 4. Speichern
                history_torque.append(torch.mean(current_torques, dim=0).cpu().numpy())
                history_vel.append(torch.mean(current_vels, dim=0).cpu().numpy())
                history_power.append(torch.mean(current_power, dim=0).cpu().numpy())
                history_total_power.append(torch.mean(total_power_step).cpu().numpy())

                contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
                # Hole die Kraft in Z-Richtung (oder die Norm)
                forces = contact_sensor.data.net_forces_w_history[:, :, :, :].norm(dim=-1).max(dim=1)[0]
                current_max_force = forces.max().item()
                history_forces.append(current_max_force)

                eval_step_counter += 1  
                
                # --- OUTPUT ALLE 200 SCHRITTE (INKL. RTF) ---
                if eval_step_counter % 200 == 0:
                    # Timing berechnen
                    current_time = time.time()
                    elapsed_real_time = current_time - rtf_last_window_time
                    elapsed_sim_time = 200 * dt
                    
                    rtf = elapsed_sim_time / elapsed_real_time if elapsed_real_time > 0 else 0
                    video_speedup = 1.0 / rtf if rtf > 0 else 1.0
                    fps = 200 / elapsed_real_time if elapsed_real_time > 0 else 0
                    
                    rtf_last_window_time = current_time

                    # Nimm die letzten 200 Einträge aus der Liste
                    last_200_forces = history_forces[-200:]
                    last_200_speeds = history_speed[-200:]
                    
                    # Finde den größten Wert
                    max_force_in_window = max(last_200_forces)
                    max_speed_in_window = max(last_200_speeds)

                    print(f"\n[Eval] {eval_step_counter}/{EVAL_DURATION_STEPS} Steps...")
                    print(f" -> Härtester Aufprall in den letzten 200 Steps: {max_force_in_window:.2f} N")
                    print(f" -> Höchste Geschwindigkeit in den letzten 200 Steps: {max_speed_in_window:.2f} m/s")
                    print(f" -> Performance (RTF):    {rtf:.2f}x ({fps:.1f} Steps/s)")
                    print(f" -> Video-Schnitt:        Um das {video_speedup:.2f}-fache beschleunigen")

            # --- SPEICHERN & BEENDEN ---
            if eval_step_counter == EVAL_DURATION_STEPS:
                total_real_time = time.time() - rtf_start_time
                total_sim_time = EVAL_DURATION_STEPS * dt
                overall_rtf = total_sim_time / total_real_time if total_real_time > 0 else 0
                overall_speedup = 1.0 / overall_rtf if overall_rtf > 0 else 1.0

                print("\n" + "="*60)
                print(f"EVALUATION FERTIG. SPEICHERE DATEN...")
                print("="*60)
                print(f"Gesamte echte Zeit: {total_real_time:.1f} Sekunden")
                print(f"Durchschnittlicher RTF: {overall_rtf:.2f}x")
                print(f"--> Finale Video-Beschleunigung: {overall_speedup:.2f}x")
                
                arr_torque = np.array(history_torque)
                arr_vel = np.array(history_vel)
                arr_power = np.array(history_power)
                arr_total_power = np.array(history_total_power)
                arr_forces = np.array(history_forces)
                arr_speed = np.array(history_speed)
                
                # Dictionary für np.savez vorbereiten
                save_data = {
                    "torque": arr_torque, 
                    "velocity": arr_vel, 
                    "power": arr_power, 
                    "total_power": arr_total_power,
                    "joint_names": np.array(joint_names),
                    "forces": arr_forces,
                    "speed": arr_speed
                }

                # Joule-Daten nur speichern, wenn berechnet
                if calculate_joule_heating:
                    save_data["joule"] = np.array(history_joule)
                    save_data["total_joule"] = np.array(history_total_joule)

                save_filename = os.path.join(log_dir, "evaluation_data.npz")
                eval_step_counter += 1  # Verhindert mehrfaches Speichern
                
                # Daten speichern (Entpackt das Dictionary)
                np.savez(save_filename, **save_data)
                
                print(f"\nDaten gespeichert unter:\n-> {save_filename}")
                print(f"\nFormat der Datei:")
                print(f" - torque:      {arr_torque.shape} (Steps x Joints)")
                print(f" - power:       {arr_power.shape}")
                if calculate_joule_heating:
                    print(f" - joule:       {save_data['joule'].shape}")
                print("="*60 + "\n")
                
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
