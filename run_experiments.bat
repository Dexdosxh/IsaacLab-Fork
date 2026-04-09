@echo off
echo 
echo ===================================================
:: Run 1 energy weight -0.0001
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run1 env.rewards.energy.weight=-0.0001
:: Run 2 energy weight -0.0005
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run2 env.rewards.energy.weight=-0.0005
:: Run 3 energy weight -0.001
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run3 env.rewards.energy.weight=-0.001





echo ALLE EXPERIMENTE ABGESCHLOSSEN!
echo ===================================================
pause