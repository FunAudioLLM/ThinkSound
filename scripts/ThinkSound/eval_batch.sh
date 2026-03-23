#!/bin/bash

# Check number of arguments
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <video_path> <csv_path> <save_path (optional)> [use-half]"
    exit 1
fi

VIDEO_PATH="$1"
CSV_PATH="$2"
SAVE_PATH="$3"
USE_HALF_FLAG="$4"

dataset_config="ThinkSound/configs/multimodal_dataset_demo.json"
model_config="ThinkSound/configs/model_configs/thinksound.json"

# Create necessary directories
mkdir -p results results/features

SAVE_PATH=${SAVE_PATH:-"results/features"}


FIRST_VIDEO=$(find "$VIDEO_PATH" -type f \( -iname "*.mp4" \) | head -n 1)

if [ -z "$FIRST_VIDEO" ]; then
    echo "❌ No .mp4 video file found in $VIDEO_PATH"
    exit 1
fi

DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$FIRST_VIDEO")
DURATION_SEC=${DURATION%.*}


# Run feature extraction
echo "⏳ Extracting features..."
EXTRACT_CMD=("python" "extract_latents.py" "--root" "$VIDEO_PATH" "--tsv_path" "$CSV_PATH" "--save-dir" "results/features" "--duration_sec" "$DURATION_SEC")
if [ "$USE_HALF_FLAG" = "use-half" ]; then
    EXTRACT_CMD+=("--use_half")
fi

"${EXTRACT_CMD[@]}" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Feature extraction failed"
    exit 3
fi

# Run inference
echo "⏳ Running model inference..."
python eval_batch.py --dataset-config "$dataset_config" \
    --model-config "$model_config" \
    --duration-sec "$DURATION_SEC" \
    --results-dir "results/features"\
    --save-dir "$SAVE_PATH" 2>&1 \

if [ $? -ne 0 ]; then
    echo "❌ Inference failed"
    exit 4
fi

# Get generated audio file
CURRENT_DATE=$(date +"%m%d")
AUDIO_PATH=$SAVE_PATH"/${CURRENT_DATE}_batch_size1"


echo "Audio files path: $AUDIO_PATH"