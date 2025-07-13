<h1 align="center">ThinkSound</h1>

<p align="center">
  🌐
  <a href="https://openaitx.github.io/view.html?user=FunAudioLLM&project=ThinkSound&lang=en">English</a> |
  <a href="https://openaitx.github.io/view.html?user=FunAudioLLM&project=ThinkSound&lang=zh-CN">简体中文</a> |
  <a href="https://openaitx.github.io/view.html?user=FunAudioLLM&project=ThinkSound&lang=zh-TW">繁體中文</a> |
  <a href="https://openaitx.github.io/view.html?user=FunAudioLLM&project=ThinkSound&lang=es">Español</a> |
  <a href="https://openaitx.github.io/view.html?user=FunAudioLLM&project=ThinkSound&lang=fr">Français</a> |
  <a href="https://openaitx.github.io/view.html?user=FunAudioLLM&project=ThinkSound&lang=ja">日本語</a>
  
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2506.21448">
    <img src="https://img.shields.io/badge/arXiv-2506.21448-b31b1b.svg" alt="arXiv"/>
  </a>
  &nbsp;
  <a href="https://thinksound-project.github.io/">
    <img src="https://img.shields.io/badge/Online%20Demo-🌐-blue" alt="Online Demo"/>
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/FunAudioLLM/ThinkSound">
    <img src="https://img.shields.io/badge/HuggingFace-Spaces-orange?logo=huggingface" alt="Hugging Face"/>
  </a>
  &nbsp;
  <a href="https://modelscope.cn/studios/iic/ThinkSound">
    <img src="https://img.shields.io/badge/ModelScope-在线体验-green" alt="ModelScope"/>
  </a>
</p>

<p align="center">
  If you find this project useful,<br>
  a star ⭐ on GitHub would be greatly appreciated!
</p>

---

**ThinkSound** is a unified Any2Audio generation framework with flow matching guided by Chain-of-Thought (CoT) reasoning.

PyTorch implementation for multimodal audio generation and editing: generate or edit audio from video, text, and audio, powered by step-by-step reasoning from Multimodal Large Language Models (MLLMs).

![Teaser](assets/figs/fig1_teaser.png)
---

## 📰 News
- **2025.07** &nbsp;  🔧 Major update: model lightweighted and optimized memory and GPU usage, now supports high-throughput audio generation at scale!
- **2025.07** &nbsp; 🔥Online demo on [Hugging Face Spaces](https://huggingface.co/spaces/FunAudioLLM/ThinkSound) and [ModelScope](https://modelscope.cn/studios/iic/ThinkSound) for interactive experience!
- **2025.07** &nbsp; 🔥Released inference scripts and web interface; 
- **2025.06** &nbsp; 🔥[ThinkSound paper](https://arxiv.org/pdf/2506.21448) released on arXiv!
- **2025.06** &nbsp; 🔥[Online Demo](http://thinksound-project.github.io/) is live - try it now!

---


## 🚀 Features

- **Any2Audio**: Generate audio from arbitrary modalities — video, text, audio, or their combinations.
- **Video-to-Audio SOTA**: Achieves state-of-the-art results on multiple V2A benchmarks.
- **CoT-Driven Reasoning**: Chain-of-Thought reasoning for compositional and controllable audio generation via MLLMs.
- **Interactive Object-centric Editing**: Refine or edit specific sound events by clicking on visual objects or using text instructions.
- **Unified Framework**: One foundation model supports generation, editing, and interactive workflow.

---

## ✨ Method Overview

ThinkSound decomposes audio generation and editing into three interactive stages, all guided by MLLM-based Chain-of-Thought (CoT) reasoning:

1. **Foley Generation:** Generate foundational, semantically and temporally aligned soundscapes from video.
2. **Object-Centric Refinement:** Refine or add sounds for user-specified objects via clicks or regions in the video.
3. **Targeted Audio Editing:** Modify generated audio using high-level natural language instructions.

![ThinkSound Overview](assets/figs/fig3_model.png)
<!-- A large-scale CoT-annotated dataset (**AudioCoT**) is used to train both the reasoning module and the unified audio foundation model.
![AudioCoT Pipeline](assets/figs/fig2_dataset.png) -->

---

## ⚡ Quick Start

**Environment Preparation:**
```bash
git clone https://github.com/liuhuadai/ThinkSound.git
cd ThinkSound
pip install -r requirements.txt
conda install -y -c conda-forge 'ffmpeg<7'
# Download pretrained weights https://huggingface.co/liuhuadai/ThinkSound to Directory ckpts/
# model weights can be also downloaded from https://www.modelscope.cn/models/iic/ThinkSound
git lfs install
git clone https://huggingface.co/liuhuadai/ThinkSound ckpts
```

**Make it executable**
```bash
chmod +x scripts/demo.sh
```

**Run the script**
```bash
./scripts/demo.sh <video_path> <title> <CoT description> [use-half]
```
Add use-half at the end to enable half precision inference, which reduces GPU memory usage.

Use the `eval_batch.sh` script to extract features from a batch of videos and run inference to generate audio outputs.

```bash
chmod +x scripts/eval_batch.sh
./scripts/eval_batch.sh <video_path> <csv_path> <save_path (optional)> [use-half]
```

`<video_path>`:Path to the root directory containing video files.
  * **Requirement**: All videos should be in `.mp4` format.
  * **Assumption**: All videos have **equal duration**.

`<csv_path>`:Path to the CSV file containing text descriptions (e.g., captions, CoT prompts) for each video.
  * Format should be similar to `demo_test.csv`, where each row corresponds to a video and includes at least the filename (without extension) and associated text.

`<save_path>` (optional):
  Directory where the generated audios will be saved.
  * Defaults to `results/features` if not provided.

`[use-half]` (optional):


### Web Interface Usage

For an interactive experience, launch the Gradio web interface:

```bash
python app.py
```

---

## 📝 TODO
* - [ ] Release training scripts for ThinkSound models
* - [ ] Open-source AudioCoT dataset and automated pipeline
* - [ ] Provide detailed documentation and API reference
* - [ ] Add support for additional modalities and downstream tasks
* - [ ] Release models at different scales
* - [ ] Provide a ready-to-use environment image
* - [ ] A beginner-friendly Windows quick-start README
---

## 📄 License

This project is released under the [Apache 2.0 License](LICENSE).

> **Note:**  
> The code, models, and dataset are **for research and educational purposes only**.  
> **Commercial use is NOT permitted.**
>
> For commercial licensing, please contact the authors.

---

## 📖 Citation

If you find ThinkSound useful in your research or work, please cite our paper:

```bibtex
@misc{liu2025thinksoundchainofthoughtreasoningmultimodal,
      title={ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing}, 
      author={Huadai Liu and Jialei Wang and Kaicheng Luo and Wen Wang and Qian Chen and Zhou Zhao and Wei Xue},
      year={2025},
      eprint={2506.21448},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2506.21448}, 
}
```

---

## 📬 Contact

✨ Feel free to [open an issue](https://github.com/liuhuadai/ThinkSound/issues) or contact us via email ([liuhuadai@zju.edu.cn](mailto:liuhuadai@zju.edu.cn)) if you have any questions or suggestions!
