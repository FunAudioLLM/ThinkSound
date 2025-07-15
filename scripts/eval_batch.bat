@echo off
setlocal enabledelayedexpansion

set ARG_COUNT=0
if not "%~1"=="" set /a ARG_COUNT+=1
if not "%~2"=="" set /a ARG_COUNT+=1
if not "%~3"=="" set /a ARG_COUNT+=1
if not "%~4"=="" set /a ARG_COUNT+=1

if !ARG_COUNT! LSS 2 (
    echo Usage: %~nx0 ^<video_folder_path^> ^<csv_path^> [save_path] [use-half]
    exit /b 1
)

set "VIDEO_PATH=%~1"
set "CSV_PATH=%~2"
set "SAVE_PATH=%~3"
set "USE_HALF_FLAG=%~4"

if "!SAVE_PATH!"=="" (
    set "SAVE_PATH=results\features"
)

set "DATASET_CONFIG=ThinkSound\configs\multimodal_dataset_demo.json"
set "MODEL_CONFIG=ThinkSound\configs\model_configs\thinksound.json"

if not exist results mkdir results
if not exist results\features mkdir results\features
if not exist "!SAVE_PATH!" mkdir "!SAVE_PATH!"

set "FIRST_VIDEO="
for %%f in ("!VIDEO_PATH!\*.mp4") do (
    if not defined FIRST_VIDEO set "FIRST_VIDEO=%%~ff"
)

if not defined FIRST_VIDEO (
    echo ❌ No .mp4 video file found in folder "!VIDEO_PATH!"
    exit /b 1
)

echo First video found: !FIRST_VIDEO!

for /f %%i in ('ffprobe -v error -show_entries format^=duration -of default^=noprint_wrappers^=1:nokey^=1 "!FIRST_VIDEO!"') do set "DURATION=%%i"
for /f "tokens=1 delims=." %%a in ("!DURATION!") do set "DURATION_SEC=%%a"
echo Video duration: !DURATION_SEC! seconds

echo ⏳ Extracting features...
set "CMD=python extract_latents.py --root !VIDEO_PATH! --tsv_path !CSV_PATH! --save-dir results\features --duration_sec !DURATION_SEC!"
if /i "!USE_HALF_FLAG!"=="use-half" (
    set "CMD=!CMD! --use_half"
)
echo Running: !CMD!
call !CMD!
if errorlevel 1 (
    echo ❌ Feature extraction failed.
    exit /b 3
)

echo ⏳ Running model inference...
set "CMD=python eval_batch.py --dataset-config !DATASET_CONFIG! --model-config !MODEL_CONFIG! --duration-sec !DURATION_SEC! --results-dir results\features --save-dir !SAVE_PATH!"
echo Running: !CMD!
call !CMD!
if errorlevel 1 (
    echo ❌ Inference failed.
    exit /b 4
)

for /f %%i in ('powershell -Command "Get-Date -Format MMdd"') do set "CURRENT_DATE=%%i"
set "AUDIO_PATH=!SAVE_PATH!\!CURRENT_DATE!_batch_size1"

echo ✅ Audio files saved in: !AUDIO_PATH!

endlocal
exit /b 0
