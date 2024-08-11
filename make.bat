@echo off
setlocal

REM Modular bat file for easy addition of new commands

REM Check command-line arguments
echo Command-line argument: %1

if "%1"=="clean" goto clean
if "%1"=="package" goto package
if "%1"=="compile" goto compile
if "%1"=="html" goto html
if "%1"=="profile" goto profile

REM Default case: no valid argument provided
echo No valid argument provided.
echo Usage: make.bat [clean|package|compile|html|profile [profile_file]]
goto end

:clean
echo Deleting all .c and .pyd files from the Sloth directory and its subdirectories...
@REM del /s /q "Sloth\*.c"
@REM del /s /q "Sloth\*.pyd"

echo Removing the build directory...
rmdir /s /q "build"

echo Removing the CMakeFiles directory...
rmdir /s /q "CMakeFiles"

if exist "docs\build" (
    echo Removing the docs\build directory...
    rmdir /s /q "docs\build"
)

echo Removing the test\profile directory...
if exist "test\profile" (
    rmdir /s /q "test\profile"
)

echo Cleanup completed.
goto end


:package
echo Packaging the project...
pip install .
goto end

:compile
echo Compiling Cython files...
pip install --no-build-isolation -ve .
goto end

:html
@REM echo Building HTML documentation...
@REM sphinx-build -M html ./docs/source ./docs/build/ -E
echo Command not yet implemented.
goto end

:profile
@REM REM Create the test\profile directory if it doesn't exist
@REM if not exist "tests\profile" (
@REM     mkdir "tests\profile"
@REM )

@REM set PROF_FILE=%2
@REM if "%PROF_FILE%"=="" (
@REM     set PROF_FILE=Sloth.prof
@REM )

@REM set PROFILE_PATH=tests\profile\%PROF_FILE%

@REM echo Profiling the project and saving to %PROFILE_PATH%...

@REM REM Ensure the profile_example.py script is configured correctly to profile your code.
@REM python -m cProfile -o %PROFILE_PATH% "tests\profiler.py"

@REM if exist %PROFILE_PATH% (
@REM     gprof2dot -f pstats %PROFILE_PATH% | dot -Tpng -o %PROFILE_PATH%.png
@REM     echo Call graph generated as %PROFILE_PATH%.png
@REM )

@REM echo Profiling completed.

echo Command not yet implemented.

goto end

:end
endlocal