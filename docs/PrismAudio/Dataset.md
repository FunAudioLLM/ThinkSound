# Dataset Preparation Guide

This guide provides step-by-step instructions for preparing datasets to train models in this repository.

## 0. Pre-requisites

Ensure the following checkpoint files exist in the `ckpts/` directory before continuing:

* `ckpts/vae.ckpt`
* `ckpts/synchformer_state_dict.pth`

## 1. Preparing Video-Text Datasets

To convert raw videos and CoT annotations into training features, use the following command:

```bash
torchrun --nproc_per_node=8 data_utils/extract_training_video.py \
    --root <video_path> \
    --tsv_path <csv_path> \
    --save-dir <feature_output_dir> \
    --add_video_path <reference_video_path> \
    --add_audio_path <reference_audio_path>
```

### Arguments

* `--root <video_path>`: Path to the root directory containing all `.mp4` videos to be processed.
* `--tsv_path <csv_path>`: Path to the TSV/CSV file containing `id` and `caption_cot` columns.
* `--save-dir <feature_output_dir>`: Directory where extracted feature `.npz` files will be saved.
* `--add_video_path <reference_video_path>` *(optional)*: Reference video directory required for enabling the **Synchformer reward** in GRPO. Typically the same as `--root`.
* `--add_audio_path <reference_audio_path>` *(optional)*: Reference audio directory required for enabling the **ITD reward** in GRPO.

> **Note:** `--add_video_path` and `--add_audio_path` are optional. Only provide them if you intend to use the corresponding reward functions during GRPO training.

---

## 2. Organizing Feature Files

After extraction, create a `.txt` file listing all generated feature file names (one per line), for example:

```
item1.npz
item2.npz
item3.npz
...
```

This file acts as the dataset split index and will be referenced in the dataset configuration.

---

## 3. Creating the Dataset Configuration JSON

Create a JSON file following the structure below (adapted from `ThinkSound/configs/multimodal_dataset_demo_prismaudio.json`):

```json
{
    "dataset_type": "video_dataset",
    "datasets": [
        {
            "id": "your_dataset_id",
            "path": "path_to_feature_dir",
            "split_path": "path_to_train_split_txt"
        }
    ],
    "val_datasets": [
        {
            "id": "your_val_dataset_id",
            "path": "path_to_val_feature_dir",
            "split_path": "path_to_val_split_txt"
        }
    ],
    "test_datasets": [
        {
            "id": "your_test_dataset_id",
            "path": "path_to_test_feature_dir",
            "split_path": "path_to_test_split_txt"
        }
    ],
    "random_crop": false,
    "input_type": "video",
    "fps": 8
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `dataset_type` | Fixed as `"video_dataset"` |
| `datasets` | List of training feature directories with their split `.txt` files |
| `val_datasets` | Validation set, same structure as `datasets` |
| `test_datasets` | Test set, same structure as `datasets` |
| `random_crop` | Whether to apply random cropping, typically `false` |
| `input_type` | Fixed as `"video"` |


You can include multiple datasets under `datasets` by appending additional dictionary blocks to the list.

---

## 4. Proceed to Training

Refer to [`Training.md`](./Training.md) for detailed training instructions once the dataset configuration is complete.