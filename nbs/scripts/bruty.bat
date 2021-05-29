echo off
echo bruty.bat <configname> [num_processes=1]
echo This script starts a lock server, runs the cleanup code on the config given then runs the num_processes copies of the combine script for that config given
echo MAX processes is 5
SETLOCAL EnableDelayedExpansion

if "%~1"=="" (exit /B 0)
if "%~2"=="" (set /A np = 1) else (set /A np = %2)
if %np% gtr 5 (set /A np = 5)

rem use the NBS environment so it can access the CARIS35 environment
rem call D:\Pydro21_Dev\Scripts\activate Pydro38
call D:\languages\miniconda3\Scripts\activate NBS

echo Starting a lock server
cd "%~dp0"
rem read the port number from the config
call get_config_val %1 lock_server_port

cd ..\bruty
rem first argument for start is the title
start "lock server %lock_server_port%" cmd /c "cd "%cd%" & D:\languages\miniconda3\Scripts\activate NBS & python lock_server.py %lock_server_port%"

echo Giving the lock server time to start up - don't press a key unless you know the server is started already
timeout 20

echo Run cleanup which looks for data without scoring parameters and removes them as needed
cd "%~dp0"
python cleanup.py %1
rem python validate.py %1
rem if %ERRORLEVEL% gtr 0 (exit /B !ERRORLEVEL!)

echo launching %np% combine processes
rem first argument for start is the title
for /L %%a in (1,1,!np!) Do (
    start "combine %1" cmd /k "cd "%~dp0" & D:\languages\miniconda3\Scripts\activate NBS & python combine_surveys.py %1"
)


rem if "%~2"=="" (set /A port = 5000) else (set /A port = %2)
