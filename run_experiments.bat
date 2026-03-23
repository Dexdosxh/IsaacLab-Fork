@echo off
echo 
echo ===================================================
:: Run3: verschiedene thresholds feet air time rewards für energy target [-0.01, -0.8] ausprobieren
:: G1 - progress reward 2.0, progress target 3.0, energy target [-0.01, -0.8], action_l2 penalty included, threshold 0.45 (Run3_1)
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run3_1 env.rewards.action_l2.weight=-0.01 env.rewards.feet_air_time.params.threshold=0.45 agent.rewards_expect.energy=[-0.01,-0.08] agent.rewards_expect.progress=3.0
:: G1 - progress reward 2.0, progress target 3.0, energy target [-0.01, -0.8], action_l2 penalty included, threshold 0.5 (Run3_2)
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run3_2 env.rewards.action_l2.weight=-0.01 env.rewards.feet_air_time.params.threshold=0.5 agent.rewards_expect.energy=[-0.01,-0.08] agent.rewards_expect.progress=3.0
:: G1 - progress reward 2.0, progress target 3.0, energy target [-0.01, -0.8], action_l2 penalty included, threshold 0.6 (Run3_3)
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run3_3 env.rewards.action_l2.weight=-0.01 env.rewards.feet_air_time.params.threshold=0.6 agent.rewards_expect.energy=[-0.01,-0.08] agent.rewards_expect.progress=3.0

:: Run4: Normal + action_rate
:: G1 - progress reward 2.0, progress target 2.0, energy target [-0.01, -0.8], action_l2 penalty included (Run4_1)
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run4_1 env.rewards.action_l2.weight=-0.01 env.rewards.action_rate_l2.weight=-0.005 agent.rewards_expect.energy=[-0.01,-0.08]
:: G1 - progress reward 2.0, progress target 3.0, energy target [-0.01, -0.8], action_l2 penalty included (Run4_2)
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run4_2 env.rewards.action_l2.weight=-0.01 env.rewards.action_rate_l2.weight=-0.005 agent.rewards_expect.energy=[-0.01,-0.08] agent.rewards_expect.progress=3.0

:: Run5: joule heating target ausprobieren
:: G1 - progress reward 2.0 progress target 2.0 energy target [-0.01, -0.8], joule heating target -0.15, action_l2 penalty included (Run5_1)
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run5_1 env.rewards.action_l2.weight=-0.01 agent.rewards_expect.energy=[-0.01,-0.08] agent.rewards_expect.joule_heating=-0.15
:: G1 - progress reward 2.0 progress target 2.0 energy target [-0.01, -0.8], joule heating target [-0.15, -0.1] action_l2 penalty included (Run5_2)
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run5_2 env.rewards.action_l2.weight=-0.01 agent.rewards_expect.energy=[-0.01,-0.08] agent.rewards_expect.joule_heating=[-0.15,-0.1]
:: G1 - progress reward 2.0 progress target 3.0 energy target [-0.01, -0.8], joule heating target -0.15, action_l2 penalty included (Run5_3)
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run5_3 env.rewards.action_l2.weight=-0.01 agent.rewards_expect.energy=[-0.01,-0.08] agent.rewards_expect.joule_heating=-0.15 agent.rewards_expect.progress=3.0
:: G1 - progress reward 2.0 progress target 3.0 energy target [-0.01, -0.8], joule heating target [-0.15, -0.1] action_l2 penalty included (Run5_4)
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run5_4 env.rewards.action_l2.weight=-0.01 agent.rewards_expect.energy=[-0.01,-0.08] agent.rewards_expect.joule_heating=[-0.15,-0.1] agent.rewards_expect.progress=3.0

echo ALLE EXPERIMENTE ABGESCHLOSSEN!
echo ===================================================
pause