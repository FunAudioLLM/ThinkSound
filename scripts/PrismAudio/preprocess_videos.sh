

NPROC_PER_NODE=4          # Number of GPUs per node
MASTER_PORT=29605         # Communication port for distributed training

ROOT="path_to_your_videos"      # Root directory of input videos
SAVE_DIR="path_to_your_dataset"    # Output directory for processed results
TSV_PATH="demo_test.csv"  # Path to video metadata CSV file (must contain video id and caption_cot)

# ===== Run =====
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master-port=$MASTER_PORT \
    data_utils/prismaudio_data_process.py \
    --root "$ROOT" \
    --save-dir "$SAVE_DIR" \
    --tsv_path "$TSV_PATH"
