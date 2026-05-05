@echo off
echo 
echo ===================================================
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Knee32Seed2 --seed 1

python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Knee32Seed3 --seed 2

python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Knee32Seed4 --seed 3

python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-G1-v0 --headless --run_name Knee32Seed5 --seed 4


echo ALLE EXPERIMENTE ABGESCHLOSSEN!
echo ===================================================
pause