#!/bin/bash


ckpt_dir="your_project_name"
log_dir="logs/$ckpt_dir"
dataset_config="ThinkSound/configs/multimodal_dataset_demo.json"
model_config="ThinkSound/configs/model_configs/thinksound.json"
pretransform_ckpt_path="ckpts/vae.ckpt"
#export MASTER_ADDR="10.32.3.240"
export MASTER_PORT="9511"
# pip install git+https://github.com/patrick-kidger/torchcubicspline.git

debug_mode="false"
node_rank=0


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) debug_mode="true"; shift ;;
        --node-rank) node_rank="$2"; shift; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

export NODE_RANK=$node_rank
export WORLD_SIZE=8

mkdir demos

if [ "$debug_mode" != "true" ]; then
    mkdir -p "$log_dir"

    cp "$dataset_config" "$log_dir/"
    cp "$model_config" "$log_dir/"
    cp "$0" "$log_dir/"
fi



if [ "$debug_mode" == "true" ]; then
    num_gpus=1
    num_nodes=1
else
    num_gpus=8
    num_nodes=1
fi


echo "Training Configuration:"
echo "Checkpoint Directory: $ckpt_dir"
echo "Log Directory: $log_dir"
echo "Dataset Config: $dataset_config"
echo "Model Config: $model_config"
echo "Pretransform Checkpoint Path: $pretransform_ckpt_path"
echo "Num GPUs: $num_gpus"
echo "Num Nodes: $num_nodes"
echo "Batch Size: 32"
echo "Num Workers: 24"
echo "Node Rank: $node_rank"


if [ "$debug_mode" == "true" ]; then
    nohup python train.py \
        --dataset-config "$dataset_config" \
        --model-config "$model_config" \
        --name "$ckpt_dir" \
        --save-dir "logs/" \
        --pretransform-ckpt-path "$pretransform_ckpt_path" \
        --checkpoint-every 2000 \
        --num-gpus "$num_gpus" \
        --num-nodes "$num_nodes" \
        --batch-size 32 \
        --num-workers 24
else
    nohup python train.py \
        --dataset-config "$dataset_config" \
        --model-config "$model_config" \
        --name "$ckpt_dir" \
        --save-dir "logs/" \
        --pretransform-ckpt-path "$pretransform_ckpt_path" \
        --checkpoint-every 4000 \
        --num-gpus "$num_gpus" \
        --num-nodes "$num_nodes" \
        --batch-size 32 \
        --num-workers 24 \
        > "$log_dir/train.log" 2>&1 &
    
    echo "Training started. Logs can be found in $log_dir/train.log"
fi
