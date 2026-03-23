#!/bin/bash

# 变量定义
ckpt_dir="ckpts/thinksound.ckpt"
test_batch_size=1
dataset_config="ThinkSound/configs/multimodal_dataset_demo.json"
model_config="ThinkSound/configs/model_configs/thinksound.json"
pretransform_ckpt_path="ckpts/vae.ckpt"
# 默认值
debug_mode="true"
node_rank=0

result_path="results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration-sec)
      if [[ -n "$2" && "$2" != --* ]]; then
        duration_sec="$2"
        shift 2
      else
        echo "❌ Argument --duration-sec requires a value"
        exit 1
      fi
      ;;
    --result-path)
      if [[ -n "$2" && "$2" != --* ]]; then
        result_path="$2"
        shift 2
      else
        echo "❌ Argument --result-path requires a path"
        exit 1
      fi
      ;;
    *)
      echo "❌ Unknown argument: $1"
      exit 1
      ;;
  esac
done

export NODE_RANK=$node_rank
export RANK=$node_rank

num_gpus=1
num_nodes=1

export WORLD_SIZE=$((num_gpus * num_nodes))
# 打印配置信息
echo "Training Configuration:"
echo "Checkpoint Directory: $ckpt_dir"
echo "Dataset Config: $dataset_config"
echo "Model Config: $model_config"
echo "Pretransform Checkpoint Path: $pretransform_ckpt_path"
echo "Num GPUs: $num_gpus"
echo "Num Nodes: $num_nodes"
echo "Test Batch Size: $test_batch_size"
echo "Num Workers: 20"
echo "Node Rank: $node_rank"
echo "WORLD SIZE: $WORLD_SIZE"


python predict.py \
        --dataset-config "$dataset_config" \
        --model-config "$model_config" \
        --ckpt-dir "$ckpt_dir" \
        --pretransform-ckpt-path "$pretransform_ckpt_path" \
        --checkpoint-every 2000 \
        --num-gpus "$num_gpus" \
        --num-nodes "$num_nodes" \
        --batch-size 1 \
        --test-batch-size $test_batch_size \
        --num-workers 32 \
        --duration-sec $duration_sec \
        --results-dir $result_path \

