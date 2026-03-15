@echo off
echo 
echo ===================================================
:: 4B Seed 1
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Humanoid-v0 --headless --seed 1 --run_name Exp4BSeed2 env.rewards.energy.weight=-0.00005 env.rewards.joule_heating.weight=-0.00001 agent.rewards_expect.energy=-0.035 agent.rewards_expect.progress=3.0 

:: 4B Seed 2
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Humanoid-v0 --headless --seed 2 --run_name Exp4BSeed3 env.rewards.energy.weight=-0.00005 env.rewards.joule_heating.weight=-0.00001 agent.rewards_expect.energy=-0.035 agent.rewards_expect.progress=3.0

:: 4B Seed 3
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Humanoid-v0 --headless --seed 3 --run_name Exp4BSeed5 env.rewards.energy.weight=-0.00005 env.rewards.joule_heating.weight=-0.00001 agent.rewards_expect.energy=-0.035 agent.rewards_expect.progress=3.0

:: 4B Seed 4
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Humanoid-v0 --headless --seed 4 --run_name Exp4BSeed5 env.rewards.energy.weight=-0.00005 env.rewards.joule_heating.weight=-0.00001 agent.rewards_expect.energy=-0.035 agent.rewards_expect.progress=3.0 

echo ALLE EXPERIMENTE ABGESCHLOSSEN!
echo ===================================================
pause