@echo off
setlocal enabledelayedexpansion

:: Check number of arguments
if "%~3"=="" (
    echo Usage: %~nx0 ^<video_path^> ^<title^> ^<description^> [use-half]
    exit /b 1
)

set "VIDEO_PATH=%~1"
set "TITLE=%~2"
set "DESCRIPTION=%~3"
set "USE_HALF_FLAG=%~4"

set "MODEL_CONFIG=ThinkSound\configs\model_configs\thinksound.json"

:: Generate unique ID
for /f %%i in ('powershell -Command "[guid]::NewGuid().ToString().Substring(0,8)"') do set "UNIQUE_ID=%%i"

:: Create necessary directories
if not exist videos mkdir videos
if not exist cot_coarse mkdir cot_coarse
if not exist results mkdir results

:: Extract file info
for %%f in ("%VIDEO_PATH%") do (
    set "VIDEO_FILE=%%~nxf"
    set "VIDEO_ID=%%~nf"
    set "VIDEO_EXT=%%~xf"
)

:: Normalize extension
set "VIDEO_EXT=!VIDEO_EXT:.=!"
set "TEMP_VIDEO_PATH=videos\demo.mp4"

:: Convert to mp4 if needed
echo VIDEO_EXT is: !VIDEO_EXT!

if /i not "!VIDEO_EXT!"=="mp4" (
    echo Converting to mp4...
    ffmpeg -y -i "%VIDEO_PATH%" -c:v libx264 -preset fast -c:a aac "%TEMP_VIDEO_PATH%" >nul 2>&1
    if errorlevel 1 (
        echo Video conversion failed.
        exit /b 2
    )
) else (
    echo Copying "%VIDEO_PATH%" to "%TEMP_VIDEO_PATH%"
    copy "%VIDEO_PATH%" "%TEMP_VIDEO_PATH%"
)

:: Get duration (in seconds)
for /f %%i in ('ffprobe -v error -show_entries format^=duration -of default^=noprint_wrappers^=1:nokey^=1 "%TEMP_VIDEO_PATH%"') do set "DURATION=%%i"
for /f "tokens=1 delims=." %%a in ("%DURATION%") do set "DURATION_SEC=%%a"
echo Duration is: %DURATION_SEC%

:: Create cot.csv
set "CSV_PATH=cot_coarse\cot.csv"
echo id,caption,caption_cot> "%CSV_PATH%"
echo demo,"%TITLE%","%DESCRIPTION:"='%" >> "%CSV_PATH%"

:: Run feature extraction
echo Extracting features...
set "CMD=python extract_latents.py --duration_sec %DURATION_SEC%"
if "%USE_HALF_FLAG%"=="use-half" (
    set "CMD=%CMD% --use_half"
)
call %CMD%
if errorlevel 1 (
    echo Feature extraction failed.
    del /f "%TEMP_VIDEO_PATH%"
    exit /b 3
)

:: Run inference
echo Running inference...
python predict.py --model-config "%MODEL_CONFIG%" --duration-sec %DURATION_SEC% --results-dir "results"
if errorlevel 1 (
    echo Inference failed.
    del /f "%TEMP_VIDEO_PATH%"
    exit /b 4
)

:: Locate audio output
for /f %%i in ('powershell -Command "(Get-Date).ToString('MMdd')"') do set "CURRENT_DATE=%%i"
set "AUDIO_PATH=results\%CURRENT_DATE%_batch_size1\demo.wav"

if not exist "%AUDIO_PATH%" (
    echo Audio file not found.
    del /f "%TEMP_VIDEO_PATH%"
    exit /b 5
)

del /f "%TEMP_VIDEO_PATH%"
echo Audio successfully generated: %AUDIO_PATH%
exit /b 0
