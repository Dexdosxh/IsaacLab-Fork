@echo off
echo 
echo ===================================================
:: Run3: thresholds 0.2, 0.25, 0.3, 0.35, action_l2 included, energy target [-0.01, -0.08], progress target 3.0
:: threshold: 0.2
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run1 env.rewards.height.params.target_height=0.7
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run2
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run3 env.rewards.height.params.target_height=0.8
:: :: threshold: 0.25




echo ALLE EXPERIMENTE ABGESCHLOSSEN!
echo ===================================================
pause