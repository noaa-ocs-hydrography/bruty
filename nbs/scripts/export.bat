echo off
SETLOCAL EnableDelayedExpansion
echo export.bat [regex=""] [num_processes=1] [delay=0]
echo pbg14 for regex would only process pbg14 while "(PBG14)|(PBG16)" would do 14 and 16 (note it's case insensitive)
echo "Tile((15)|(17)|(18)|(19)|(21)|(22)|(28)|(29))_PBG15" is an example of getting certain tiles from a specific zone
echo the delay argument will wait that many seconds before trying to start the processes.  Useful if combine is not quite finished.
echo MAX processes is 5

if "%~1"=="" (set regex=) else (set regex=%1)
if "%~2"=="" (set /A np = 1) else (set /A np = %2)
if %np% gtr 5 (set /A np = 5)

if "%~3"=="" (set /A dsec = 0) else (set /A dsec = %3)
timeout %dsec%

echo Starting a lock server
cd "%~dp0"
cd ..\bruty
rem using Pydro38 environment for gdal 3.2
rem first argument for start is the title
start "lock server 5000" cmd /c "cd "%cd%" & set pythonpath= & D:\Pydro21_Dev\Scripts\activate Pydro38 & python lock_server.py 5000"

echo Giving the lock server time to start up - don't press a key unless you know the server is started already
timeout 20

cd "%~dp0"

set dt=%date% %time%
echo launching %np% combine processes
rem first argument for start is the title
for /L %%a in (1,1,!np!) Do (
    start "export tiles" cmd /k "cd "%~dp0..\bruty" & set pythonpath=C:\git_repos\national_bathymetric_source;C:\git_repos\bruty & D:\Pydro21_Dev\Scripts\activate Pydro38 & python tile_export.py !regex! !dt!"
)
