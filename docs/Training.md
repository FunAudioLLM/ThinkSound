# Training Guide

This guide will walk you through the process of preparing data, configuring your training setup, and launching training for the ThinkSound model. For best results, we recommend reading through all steps before starting.

---

## Step 1: Prepare the Dataset

Before training, you must preprocess the dataset following the instructions in [`Dataset.md`](../docs/Dataset.md). This includes:

1. Converting raw data (e.g., video/audio/text) into structured feature files.
2. Constructing a valid dataset metadata JSON that points to all precomputed features.

Make sure your extracted dataset includes all required modalities and is organized correctly.

---

## Step 2: Configure Training Script

Open [`scripts/train.sh`](../scripts/train.sh) and customize the following items:

* Update the paths to your dataset, model config, and checkpoint directory:

  * `dataset_config`
  * `model_config`
  * `pretransform_ckpt_path`

* Modify distributed training settings as needed:

  * `num_gpus`, `num_nodes`, `node_rank`, `MASTER_PORT`, etc.
* (Optional) Enable debug mode by adding the `--debug` flag when running the script.

### 🔍 Tip

If you're using a multi-GPU setup, ensure the `WORLD_SIZE`, `NODE_RANK`, and `MASTER_PORT` are correctly set for your environment. These are critical for DistributedDataParallel (DDP) training.

---

## Step 3: Configure Demo Monitoring

To monitor training quality visually, modify the `training-demo-demo_cond` section in the [`model config`](../ThinkSound/configs/model_configs/thinksound.json) and make sure it contains exactly 10 test samples.
Set this section to include several representative **video features**. These will be periodically passed through the generator during training, allowing you to visually assess output quality over time.

---

## Step 4: Launch Training

Make the script executable (if not already) and start training:

```bash
chmod +x scripts/train.sh
./scripts/train.sh
```

Logs will be written to the specified log directory (`log_dir`). 

---

## Step 5: Customize Model and Training Parameters

To modify model architecture or training strategy, open the [`model config`](../ThinkSound/configs/model_configs/thinksound.json).
You can adjust a wide range of parameters, such as:

* Number of model parameters
* Optimizer type
* Learning rate
* Latent dimension

Be sure to keep a backup of your config for reproducibility.

---

## Optional: Debug Mode

Add the `--debug` flag when running the training script to run on a single GPU (single-node)

This is useful for quick sanity checks or development iterations.

---

## Fine-Tuning & Checkpoints

Checkpoints (including model weights, optimizer state, and EMA versions) are saved periodically in the configured log directory.
You can resume training or fine-tune on the original model by modifying the training script accordingly (add --ckpt-path). If you plan to fine-tune using our pretrained model, please use thinksound.ckpt instead of thinksound_light.ckpt.

---



Happy training! 🚀
If you run into any issues, consider opening an issue or checking the documentation for detailed help.

