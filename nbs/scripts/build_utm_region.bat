echo off

echo build_utm_region [config_prefix] [delay=0]
echo This script should launch 5 processes for the main public data and 1 each for the prereview, sensitive and 3 permutations with not_for_navigation
echo pass in the prefix of the config files group:  pbc18 or pbg14 etc
SETLOCAL EnableDelayedExpansion

if "%~1"=="" (exit /B 0)
if "%~2"=="" (set /A dsec = 0) else (set /A dsec = %2)
timeout !dsec!

start "Bruty public" cmd /k "cd "%~dp0" & bruty.bat %1_public.config 5"
start "Bruty sensitive" cmd /k "cd "%~dp0" & bruty.bat %1_sensitive.config 1"
start "Bruty enc" cmd /k "cd "%~dp0" & bruty.bat %1_enc.config 1"
start "Bruty prereview" cmd /k "cd "%~dp0" & bruty.bat %1_prereview.config 1"
start "Bruty public NFN" cmd /k "cd "%~dp0" & bruty.bat %1_public_not_for_navigation.config 1"
start "Bruty sensitive NFN" cmd /k "cd "%~dp0" & bruty.bat %1_sensitive_not_for_navigation.config 1"
start "Bruty prereview NFN" cmd /k "cd "%~dp0" & bruty.bat %1_prereview_not_for_navigation.config 1"
