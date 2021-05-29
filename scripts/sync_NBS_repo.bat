set root=D:\national-bathymetric-source
set logfile=D:\NBS_Scripts\nbs_repo_pull.log
cd /d %root%
echo %date% > %logfile%
echo %time% >> %logfile%
git branch --show-current >> %logfile% 2>&1
git pull --recurse-submodules=no >> %logfile% 2>&1
git ls-remote ssh://glen.rice@vlab.noaa.gov:29418/national_bathymetric_source > remote_commit_ids.txt
