@echo off
echo 
echo ===================================================
:: Run 1 upright threshold 0.97
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run1 env.rewards.upright.params.threshold=0.97
:: Run 2 upright threshold 0.98
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run2 env.rewards.upright.params.threshold=0.98
:: Run 3 upright threshold 0.99
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run3 env.rewards.upright.params.threshold=0.99
:: Run 4 upright threshold 0.98 roll penalty -0.5
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Run4 env.rewards.roll_penalty.weight=-0.5





echo ALLE EXPERIMENTE ABGESCHLOSSEN!
echo ===================================================
pause