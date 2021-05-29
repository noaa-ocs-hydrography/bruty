::weekly automated enc process and combine script
::Bill Shi Oct 2022

set root=D:\national-bathymetric-source
cd /d %root%
set pythonpath=D:\national-bathymetric-source;D:\test_environments\Barry\bruty
call D:\Languages\Miniconda3\Scripts\activate.bat NBS
if "%1" == "-combine" (
  goto:combine
)
::enc_processing script
call:setLogFile D:\automated_scrapers_repo\data_management\enc_scrape\weekly_enc_processing
python -m fuse_dev.scripts.workflow.enc_processing "" "D:\\national-bathymetric-source\\fuse_dev\\scripts\\workflow\\fuse_config_path.config" > %logfile% 2>&1
call:checkExitCode
::enc_supersession script
call:setLogFile D:\automated_scrapers_repo\data_management\enc_scrape\weekly_enc_supersession
python -m fuse_dev.scripts.workflow.enc_supersession "" "D:\\national-bathymetric-source\\fuse_dev\\scripts\\workflow\\fuse_config_path.config" > %logfile% 2>&1
call:checkExitCode
::update script
call:setLogFile D:\automated_scrapers_repo\data_management\enc_scrape\weekly_update
python -m fuse_dev.scripts.workflow.update "" "D:\\national-bathymetric-source\\fuse_dev\\scripts\\workflow\\fuse_config_path.config" > %logfile% 2>&1
call:checkExitCode
::score script
call:setLogFile D:\automated_scrapers_repo\data_management\enc_scrape\weekly_score
python -m fuse_dev.scripts.workflow.score "" "D:\\national-bathymetric-source\\fuse_dev\\scripts\\workflow\\fuse_config_path.config" > %logfile% 2>&1
call:checkExitCode
::validate script
call:setLogFile D:\automated_scrapers_repo\data_management\enc_scrape\weekly_validate_entries
python -m fuse_dev.scripts.workflow.validate_entries "" "D:\\national-bathymetric-source\\fuse_dev\\scripts\\workflow\\fuse_config_path.config" > %logfile% 2>&1
:combine
::combine script
call:setLogFile D:\automated_scrapers_repo\data_management\enc_scrape\weekly_enc_combine
python D:\test_environments\Barry\bruty\nbs\scripts\combine_tiles.py "D:\test_environments\Barry\bruty\nbs\scripts\OCS.SVC.NBS\enc.config" > %logfile% 2>&1
goto:eof

:setLogFile
set CUR_YYYY=%date:~10,4%
set CUR_MM=%date:~4,2%
set CUR_DD=%date:~7,2%
set CUR_HH=%time:~0,2%
set CUR_NN=%time:~3,2%
set CUR_SS=%time:~6,2%
if %CUR_HH% lss 10 (set CUR_HH=0%time:~1,1%)
set logfile=%~1_%CUR_YYYY%%CUR_MM%%CUR_DD%_%CUR_HH%%CUR_NN%%CUR_SS%.log
goto:eof

:checkExitCode
if NOT %ERRORLEVEL% == 0 exit
goto:eof