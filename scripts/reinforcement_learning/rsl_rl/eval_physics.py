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

# PLACEHOLDER: Extension template (do not remove this comment)


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
    # Standard ist oft 20s (1000 Steps). Wir setzen es auf 100s (5000 Steps).
    # So wird der Fatigue-Buffer nicht durch einen Time-Out-Reset gelöscht.
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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
    # --- EVALUATION SETUP (START) ---
    # ---------------------------------------------------------
    
    # 1. Config Parameter definieren
    EVAL_EXPONENT = 2.0
    EVAL_BUILDUP_RATE = 1.0
    EVAL_RECOVERY_RATE = 0.5
    
    EVAL_TAU_MAX_PARAMS = {
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

    # 2. Tensor für Tau Max erstellen (Mapping auf die Gelenke)
    # WICHTIG: Hier .unwrapped nutzen, um durch den RSL-RL Wrapper zu kommen!
    robot = env.unwrapped.scene["robot"] 
    
    # Initialisiere mit Default 1.0
    tau_max_tensor = torch.ones(env.num_envs, robot.num_joints, device=env.device)
    # Die IsaacLab Funktion nutzen, um die Namen den Indizes zuzuordnen
    index_list, _, value_list = string_utils.resolve_matching_names_values(
        EVAL_TAU_MAX_PARAMS, robot.joint_names
    )
    tau_max_tensor[:, index_list] = torch.tensor(value_list, device=env.device)
    # Sicherstellen, dass nichts 0 ist
    tau_max_tensor = torch.clamp(tau_max_tensor, min=1e-6)

    # 3. Fatigue Buffer initialisieren (startet bei 0)
    fatigue_buffer = torch.zeros(env.num_envs, robot.num_joints, device=env.device)

    # 4. Speicher für Ergebnisse
    eval_metrics = {
        "torque_utilization": [], 
        "fatigue_mean": [],       
        "fatigue_peak": 0.0       
    }
    
    eval_step_counter = 0
    EVAL_DURATION_STEPS = 1600 
    previous_pos = None
    print(f"\n[INFO] Starting Shadow Evaluation for {EVAL_DURATION_STEPS} steps...\n")
    # ---------------------------------------------------------
    # --- EVALUATION SETUP (END) ---
    # ---------------------------------------------------------
    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            # ---------------------------------------------------------
            # --- EVALUATION LOOP (EXTENDED) ---
            # ---------------------------------------------------------
            current_pos = robot.data.root_pos_w[:, :2]
            if previous_pos is not None:
                # 2. Distanz berechnen (Euklidischer Abstand: Wurzel aus (dx^2 + dy^2))
                # Wir nehmen den Durchschnitt über alle Envs, falls du mehrere hast
                distance_covered = torch.norm(current_pos - previous_pos, dim=-1)
                
                # 3. Geschwindigkeit berechnen: Weg / Zeit
                measured_speed = distance_covered / dt
                
                # 4. Ausgabe (Mittelwert über alle Roboter)
                avg_speed = measured_speed.mean().item()
                print(f"Gemessene Speed (Pos-Diff): {avg_speed:.2f} m/s")

            # Update für den nächsten Durchlauf
            previous_pos = current_pos.clone()
            if eval_step_counter < EVAL_DURATION_STEPS:
                # --- 0. DATEN HOLEN ---
                # A) Physik Daten
                applied_torque = robot.data.applied_torque
                dof_vel = robot.data.joint_vel
                root_lin_vel_w = robot.data.root_lin_vel_w
                root_quat_w = robot.data.root_quat_w
                
                # 2. Wir rotieren sie selbst in den Roboter-Frame
                # (Das macht genau das gleiche wie .root_lin_vel_b im Hintergrund)
                vel_b = math_utils.quat_apply_inverse(root_quat_w, root_lin_vel_w)
                
                # 3. Wir nehmen die X-Achse (Vorwärts)
                current_speed = vel_b[:, 0]

                # B) Konstanten (Bitte anpassen!)
                ROBOT_MASS = 1
                GRAVITY = 9.81
                
                # --- SICHERHEIT: NaNs abfangen ---
                if torch.isnan(applied_torque).any():
                    applied_torque = torch.nan_to_num(applied_torque, nan=0.0)

                # --- METRIK 1: TORQUE UTILIZATION ---
                rel_tau = torch.abs(applied_torque) / tau_max_tensor
                rel_tau_clamped = torch.clamp(rel_tau, max=10.0) 
                eval_metrics["torque_utilization"].append(rel_tau_clamped.mean().item())

                # --- METRIK 2: FATIGUE (Dein alter Code) ---
                fatigue_buffer = torch.clamp(fatigue_buffer - EVAL_RECOVERY_RATE * dt, min=0.0)
                instant_cost = torch.clamp(rel_tau ** EVAL_EXPONENT, max=1000.0)
                fatigue_buffer += EVAL_BUILDUP_RATE * instant_cost * dt
                fatigue_buffer = torch.clamp(fatigue_buffer, max=10000.0)
                
                # Reset Logic (Safe)
                reset_mask = dones.unsqueeze(-1).float()
                fatigue_buffer = fatigue_buffer * (1.0 - reset_mask)
                
                eval_metrics["fatigue_mean"].append(fatigue_buffer.mean().item())
                if fatigue_buffer.max().item() > eval_metrics["fatigue_peak"]:
                    eval_metrics["fatigue_peak"] = fatigue_buffer.max().item()

                # --- NEUE METRIK 3: COST OF TRANSPORT (CoT) ---
                # Mechanical Power P = sum(|tau * q_dot|)
                mech_power = torch.sum(torch.abs(applied_torque * dof_vel), dim=-1)
                
                # CoT = Power / (mgv)
                # Wir clampen velocity min auf 0.1, damit wir nicht durch 0 teilen
                safe_speed = torch.clamp(current_speed, min=0.1)
                cot = mech_power / (ROBOT_MASS * GRAVITY * safe_speed)
                
                # Speichern (wir nehmen den Mean über alle Envs)
                # Falls "cot" noch nicht im Dict ist, initialisieren wir es dynamisch
                if "cot" not in eval_metrics: eval_metrics["cot"] = []
                eval_metrics["cot"].append(cot.mean().item())

                # --- NEUE METRIK 4: ACTION SMOOTHNESS ---
                # Wir brauchen die "letzte" Action. Wir speichern sie im eval_metrics dict
                if "last_actions" in eval_metrics:
                    # Berechne Differenz zum letzten Schritt: ||a_t - a_{t-1}||^2
                    diff = (actions - eval_metrics["last_actions"]) ** 2
                    smoothness = diff.mean().item()
                    if "smoothness" not in eval_metrics: eval_metrics["smoothness"] = []
                    eval_metrics["smoothness"].append(smoothness)
                
                # Aktuelle Action für nächsten Schritt speichern
                eval_metrics["last_actions"] = actions.clone()


                # --- OUTPUT ---
                eval_step_counter += 1
                if eval_step_counter % 200 == 0:
                     print(f"[Eval] {eval_step_counter}/{EVAL_DURATION_STEPS} - Fatigue: {fatigue_buffer.mean().item():.1f} | CoT: {cot.mean().item():.2f}")

                if eval_step_counter == EVAL_DURATION_STEPS:
                    import numpy as np
                    print("\n" + "="*60)
                    print(f"OFFLINE EVALUATION RESULTS (over {EVAL_DURATION_STEPS} steps)")
                    print("="*60)
                    print(f"1. Efficiency (lower is better):")
                    print(f"   -> Cost of Transport (CoT): {np.mean(eval_metrics['cot']):.4f}")
                    print(f"   -> Avg Torque Utilization:  {np.mean(eval_metrics['torque_utilization'])*100:.2f} %")
                    print("-" * 30)
                    print(f"2. Sustainability:")
                    print(f"   -> Avg Fatigue Level:       {np.mean(eval_metrics['fatigue_mean']):.2f}")
                    print(f"   -> Peak Fatigue Level:      {eval_metrics['fatigue_peak']:.2f}")
                    print("-" * 30)
                    print(f"3. Control Quality (lower is better):")
                    if "smoothness" in eval_metrics:
                        print(f"   -> Action Jitter (Smoothness): {np.mean(eval_metrics['smoothness']):.6f}")
                    else:
                        print("   -> Smoothness: (N/A for first step)")
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
