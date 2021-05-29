@echo off
rem Passing in a python config file path and desired variable name will create an environment variable of the same name
rem this is accomplished by calling a python script which creates a temporary batch file that is run then deleted.
rem Extensions are enabled by default in Windows, using setlocal causes environment variables to only be visible in this batch - which defeats the purpose
rem setlocal EnableExtensions

rem get unique file name
:uniqLoop
  set "uniqueFileName=%tmp%\bat~%RANDOM%.bat"
  if exist "%uniqueFileName%" goto :uniqLoop

rem echo Writing to "%uniqueFileName%"
python config_to_bat.py -o "%uniqueFileName%" -i %1 -v %2
call "%uniqueFileName%"
del "%uniqueFileName%"

