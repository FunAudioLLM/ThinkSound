# PrismAudio Training Guide

This guide walks you through data preparation, configuration, and launching GRPO training for PrismAudio. We recommend reading through all steps before starting.

---

## Step 1: Prepare the Dataset

Before training, you must preprocess the dataset following the instructions in [Dataset.md](./Dataset.md). This includes:

1. Converting raw videos and CoT annotations into structured feature `.npz` files.
2. Constructing a valid dataset configuration JSON that points to all precomputed features.

Make sure your extracted dataset includes all required modalities and is organized correctly.

---

## Step 2: Configure Training Script

Open `grpo/config/grpo.py` and select or customize the appropriate config function:

- `general_thinksound_8gpus()` — for training **PrismAudio**


The key parameters to modify are:

| Parameter | Description |
|-----------|-------------|
| `model_config` | Path to model architecture config, e.g. `ThinkSound/configs/model_configs/prismaudio.json` |
| `pretransform_ckpt_path` | Path to VAE checkpoint, e.g. `ckpts/vae.ckpt` |
| `ref_model` | Path to reference model checkpoint, e.g. `ckpts/prismaudio.ckpt` |
| `ckpt_dir` | Path to the checkpoint to fine-tune from |
| `dataset_config` | Path to your dataset configuration JSON prepared in Step 1 |

Reward function weights can be adjusted under `reward_fn`:

| Reward | Key | Default Weight | Description |
|--------|-----|---------------|-------------|
| **Temporal** | `synch_reward` | `0.6` | Audio-visual synchronization via Synchformer |
| **Semantic** | `ms_clap` | `1.0` | Audio-text alignment via MS-CLAP |
| **Spatial** | `itd_reward` | `0.4` | Inter-track distance reward using reference audio |
| **Aesthetic** | `meta_reward` | `0.1` | Aesthetic quality via Meta Audiobox Aesthetics |

GRPO-specific parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.beta` | `0.04` | KL penalty weight to mitigate reward hacking |
| `sample.num_audio_per_prompt` | `16` | Number of audio candidates per prompt for group-relative advantage computation |
| `sample.num_steps` | `24` | Number of denoising steps during training sampling |
| `sample.train_num_steps` | `2` | Number of SDE steps in Fast-GRPO hybrid sampling |
| `train.learning_rate` | `1e-5` | Learning rate for GRPO optimization |
| `train.ema` | `True` | Whether to use exponential moving average |




## Step 5: Launch Training

Make the script executable (if not already) and start training:

```bash
chmod +x scripts/PrismAudio/grpo_1node8gpus.sh
./scripts/PrismAudio/grpo_1node8gpus.sh
```


---

Happy training! 🚀  
If you run into any issues, consider opening an issue or checking the documentation for detailed help.
