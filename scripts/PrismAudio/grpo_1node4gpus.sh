

export NCCL_IB_DISABLE=1
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=3
export HF_ENDPOINT=https://hf-mirror.com

MASTER_PORT=29501

# Launch command (parameters automatically read from accelerate_multi_node.yaml)
PYTHONPATH=. accelerate launch \
        --config_file grpo/scripts/accelerate_configs/multi_gpu.yaml \
        --num_machines 1 --num_processes 4 \
        --main_process_port ${MASTER_PORT} \
        grpo/scripts/train_audio_fast.py \
        --config grpo/config/grpo.py:general_thinksound_4gpus
