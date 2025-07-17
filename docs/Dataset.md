# Dataset Preparation Guide

This guide provides step-by-step instructions for preparing datasets to train models in this repository.

## 0. Pre-requisites

Ensure the following checkpoint files exist in the `ckpts/` directory before continuing:

* `ckpts/vae.ckpt`
* `ckpts/synchformer_state_dict.pth`

## 1. Preparing Video-Text Datasets

To convert raw videos and CoT into training features, use the following command:

```bash
torchrun --nproc_per_node=8 data_utils/extract_training_video.py \
    --root <video_path> \
    --tsv_path <csv_path> \
    --save-dir <feature_output_dir> \
    --duration_sec <uniform_video_duration_in_seconds> \
    --audio_samples duration_sec*44100
```

* `<video_path>`: Path to the root directory containing all .mp4 videos to be processed (all videos must be of equal duration).
* `<csv_path>`: Path to the TSV/CSV file that lists video-text pairs.(see `demo_test.csv` for format).
* `<feature_output_dir>`: Directory where extracted video features will be saved.
* `<uniform_video_duration_in_seconds>`: Duration to which all videos will be uniformly trimmed or padded.

## 2. Preparing Audio-Text Datasets

You can also include audio-text pairs for training. Use the following command to extract features:

```bash
torchrun --nproc_per_node=8 data_utils/extract_training_audio.py \
    --root <audio_path> \
    --tsv_path <csv_path> \
    --save-dir <feature_output_dir> \
    --duration_sec <uniform_audio_duration_in_seconds> \
    --audio_samples duration_sec*44100
```

* `<audio_path>`: Path to the raw audio files.
* `<csv_path>`: Path to the TSV/CSV file that lists audio-text pairs.
* `<feature_output_dir>`: Directory where extracted audio features will be saved.
* `<uniform_audio_duration_in_seconds>`: Duration to which all audios will be uniformly trimmed or padded.
* Note that the audio input for feature extraction must be trimmed to match the duration of the video-text datasets.
 
## 3. Organizing Feature Files

For each dataset (video or audio), create a `.txt` file listing all feature file names (one per line), for example:

```
item1.pth
item2.pth
item3.pth
...
```

This file acts as the training split and will be referenced in the dataset config.

## 4. Creating the Dataset Configuration JSON

Create a JSON file following the structure below (adapted from `ThinkSound/configs/multimodal_dataset_demo.json`):

```json
{
    "dataset_type": "multimodal_dir",
    "video_datasets": [
        {
            "id": "video_dataset_id",
            "path": "path_to_video_feature_dir",
            "split_path": "path_to_video_split_txt"
        }
    ],
    "audio_datasets": [
        {
            "id": "audio_dataset_id",
            "path": "path_to_audio_feature_dir",
            "split_path": "path_to_audio_split_txt"
        }
    ],
    "val_datasets": [
        {
            "id": "val_dataset_id",
            "path": "path_to_val_feature_dir",
            "split_path": "path_to_val_split_txt"
        }
    ],
    "random_crop": true,
    "input_type": "prompt"
}
```

You can include multiple datasets under `video_datasets` and `audio_datasets` by appending additional dictionary blocks to each list. The `val_datasets` is encouraged and must be a video-text dataset.

## 5. Proceed to Training

Refer to [`docs/Training.md`](./Training.md) for detailed training instructions once the dataset configuration is complete.

