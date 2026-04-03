
PYTHON=$(which python)
TORCHRUN=$(which torchrun)
export CUDA_LAUNCH_BLOCKING=1 
export TF_GPU_ALLOCATOR=cuda_malloc_async
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export TF_CPP_MIN_LOG_LEVEL=2

# Check number of arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <video_path> <description>"
    exit 1
fi

VIDEO_PATH="$1"
DESCRIPTION="$2"


model_config="PrismAudio/configs/model_configs/prismaudio.json"
ckpt_dir="ckpts/prismaudio.ckpt"
# Generate unique ID
UNIQUE_ID=$(uuidgen | cut -c 1-8)

# Create necessary directories
mkdir -p videos cot_coarse results

# Get video filename and extension
VIDEO_FILE=$(basename "$VIDEO_PATH")
VIDEO_EXT="${VIDEO_FILE##*.}"
VIDEO_ID="${VIDEO_FILE%.*}"
TEMP_VIDEO_PATH="videos/demo.mp4"

# Convert video to MP4 format if needed
if [ "${VIDEO_EXT,,}" != "mp4" ]; then
    echo "⏳ Converting video to MP4 format..."
    ffmpeg -y -i "$VIDEO_PATH" -c:v libx264 -preset fast -c:a aac -strict experimental "$TEMP_VIDEO_PATH" >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Video conversion failed"
        exit 2
    fi
else
    cp "$VIDEO_PATH" "$TEMP_VIDEO_PATH"
fi

# Calculate video duration
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$TEMP_VIDEO_PATH")
DURATION_SEC=${DURATION%.*}
echo "Duration is: $DURATION_SEC"

# Create cot.csv file
CAPTION_COT=$(echo "$DESCRIPTION" | tr '"' "'")
CSV_PATH="cot_coarse/cot.csv"
echo "id,caption_cot" > "$CSV_PATH"
echo "demo,\"$CAPTION_COT\"" >> "$CSV_PATH"

# Run feature extraction
echo "⏳ Extracting features..."
$TORCHRUN --nproc_per_node=1 data_utils/prismaudio_data_process.py --inference_mode True

if [ $? -ne 0 ]; then
    echo "❌ Feature extraction failed"
    rm -f "$TEMP_VIDEO_PATH"
    exit 3
fi

# Run inference
echo "⏳ Running model inference..."
$PYTHON predict.py \
    --model-config "$model_config" \
    --duration-sec "$DURATION" \
    --ckpt-dir "$ckpt_dir" \
    --results-dir "results"\

if [ $? -ne 0 ]; then
    echo "❌ Inference failed"
    rm -f "$TEMP_VIDEO_PATH"
    exit 4
fi

# Get generated audio file
CURRENT_DATE=$(date +"%m%d")
AUDIO_PATH="results/${CURRENT_DATE}_batch_size1/demo.wav"

# Check if audio file exists
if [ ! -f "$AUDIO_PATH" ]; then
    echo "❌ Generated audio file not found"
    rm -f "$TEMP_VIDEO_PATH"
    exit 5
fi

# Clean up temporary video file
rm -f "$TEMP_VIDEO_PATH"


echo "✅ Audio generated successfully!"
echo "Audio file path: $AUDIO_PATH"
rm -f "results/demo.npz"
