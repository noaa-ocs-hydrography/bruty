set pythonpath=D:\git_repos\bruty;D:\git_repos\national_bathymetric_source;D:\git_repos\vyperdatum
call D:\envs\nbs_20250425\scripts\activate NBS_20250425
rem change the directory and drive to local to the batch which is local to the python script
cd /D %~dp0
rem to see stdout and stderr from the console pipe it to a file 
rem call python run_bruty_operations.py //OCS-S-HyperV17/NBS_Store1/configs/bruty/windows/service.config
call python run_bruty_operations.py //OCS-S-HyperV17/NBS_Store1/configs/bruty/windows/service.config > D:\git_repos\sched.txt 2>&1
