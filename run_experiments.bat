@echo off
echo 
echo ===================================================
:: energy 3fach arme, joule heating normal, energy target auf -0.035 kein joule heating target, progress auf 2.0
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Humanoid-v0 --headless --run_name Exp1 env.rewards.energy.weight=-0.00005 env.rewards.joule_heating.weight=-0.00001 agent.rewards_expect.energy=-0.035

:: energy 3fach arme, joule heating normal, energy target auf -0.035, joule heating target auf -0.15, progress auf 2.0
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Humanoid-v0 --headless --run_name Exp2 env.rewards.energy.weight=-0.00005 env.rewards.joule_heating.weight=-0.00001 agent.rewards_expect.energy=-0.035 agent.rewards_expect.joule_heating=-0.15

:: energy 3fach arme, joule heating normal, energy target auf -0.035, joule heating target auf -0.24, progress auf 2.0
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Humanoid-v0 --headless --run_name Exp3 env.rewards.energy.weight=-0.00005 env.rewards.joule_heating.weight=-0.00001 agent.rewards_expect.energy=-0.035 agent.rewards_expect.joule_heating=-0.24

:: energy 3fach arme, joule heating normal, energy target auf -0.035 kein joule heating target, progress auf 3.0
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Humanoid-v0 --headless --run_name Exp4 env.rewards.energy.weight=-0.00005 env.rewards.joule_heating.weight=-0.00001 agent.rewards_expect.energy=-0.035 agent.rewards_expect.progress=3.0

:: energy 3fach arme, joule heating normal, energy target auf -0.035, joule heating target auf -0.15, progress auf 3.0
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Humanoid-v0 --headless --run_name Exp5 env.rewards.energy.weight=-0.00005 env.rewards.joule_heating.weight=-0.00001 agent.rewards_expect.energy=-0.035 agent.rewards_expect.joule_heating=-0.15 agent.rewards_expect.progress=3.0

:: energy 3fach arme, joule heating normal, energy target auf -0.035, joule heating target auf -0.24, progress auf 3.0
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Humanoid-v0 --headless --run_name Exp6 env.rewards.energy.weight=-0.00005 env.rewards.joule_heating.weight=-0.00001 agent.rewards_expect.energy=-0.035 agent.rewards_expect.joule_heating=-0.24 agent.rewards_expect.progress=3.0
echo ALLE EXPERIMENTE ABGESCHLOSSEN!
echo ===================================================
pause