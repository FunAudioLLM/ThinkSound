import os
# ⭐ Must be set before importing gradio

os.environ["JAX_PLATFORMS"] = "cpu" 
import gradio as gr
import logging
import sys
import json
import torch
import torchaudio
import numpy as np
import tempfile
import shutil
import subprocess
from pathlib import Path
import torch.nn.functional as F
import mediapy
from torio.io import StreamingMediaDecoder
from torchvision.transforms import v2
import time
import random
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

# ==================== Logging ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

# ==================== Constants ====================
_CLIP_FPS = 4
_CLIP_SIZE = 288
_SYNC_FPS = 25
_SYNC_SIZE = 224
SAMPLE_RATE = 44100

# ==================== Model Path Configuration ====================
from huggingface_hub import snapshot_download
snapshot_download(repo_id="FunAudioLLM/PrismAudio", local_dir="./ckpts")

MODEL_CONFIG_PATH     = "PrismAudio/configs/model_configs/prismaudio.json"
CKPT_PATH             = "ckpts/prismaudio.ckpt"
VAE_CKPT_PATH         = "ckpts/vae.ckpt"
VAE_CONFIG_PATH       = "PrismAudio/configs/model_configs/stable_audio_2_0_vae.json"
SYNCHFORMER_CKPT_PATH = "ckpts/synchformer_state_dict.pth"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ==================== Global Model Registry ====================
_MODELS = {
    "feature_extractor": None,
    "diffusion":         None,
    "model_config":      None,
    "sync_transform":    None,
}


def load_all_models():
    """Load all models once at application startup."""
    global _MODELS

    log.info("=" * 50)
    log.info("Loading all models...")

    # ---- 1. Sync video transform ----
    _MODELS["sync_transform"] = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    log.info("✅ sync_transform ready")

    # ---- 2. FeaturesUtils ----
    from data_utils.v2a_utils.feature_utils_288 import FeaturesUtils

    feature_extractor = FeaturesUtils(
        vae_ckpt=None,
        vae_config=VAE_CONFIG_PATH,
        enable_conditions=True,
        synchformer_ckpt=SYNCHFORMER_CKPT_PATH,
    )
    feature_extractor = feature_extractor.eval().to(DEVICE)
    _MODELS["feature_extractor"] = feature_extractor
    log.info("✅ FeaturesUtils loaded")

    # ---- 3. Diffusion model ----
    from PrismAudio.models import create_model_from_config
    from PrismAudio.models.utils import load_ckpt_state_dict

    with open(MODEL_CONFIG_PATH) as f:
        model_config = json.load(f)
    _MODELS["model_config"] = model_config

    diffusion = create_model_from_config(model_config)
    diffusion.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))

    vae_state = load_ckpt_state_dict(VAE_CKPT_PATH, prefix='autoencoder.')
    diffusion.pretransform.load_state_dict(vae_state)

    diffusion = diffusion.eval().to(DEVICE)
    _MODELS["diffusion"] = diffusion
    log.info("✅ Diffusion model loaded")

    log.info("=" * 50)
    log.info("All models ready. Waiting for inference requests.")


# ==================== Video Utilities ====================

def get_video_duration(video_path: str) -> float:
    video = VideoFileClip(str(video_path))
    duration = video.duration
    video.close()
    return duration


def convert_to_mp4(src: str, dst: str) -> tuple[bool, str]:
    """Re-encode any video format to h264/aac mp4 via ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", src,
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-strict", "experimental",
            dst,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr


def combine_audio_video(video_path: str, audio_path: str, output_path: str) -> tuple[bool, str]:
    """Mux generated audio into the original silent video via ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac", "-strict", "experimental",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr


def pad_to_square(video_tensor: torch.Tensor) -> torch.Tensor:
    """(L, C, H, W) -> (L, C, _CLIP_SIZE, _CLIP_SIZE)"""
    if len(video_tensor.shape) != 4:
        raise ValueError("Input tensor must have shape (L, C, H, W)")
    l, c, h, w = video_tensor.shape
    max_side = max(h, w)
    pad_h = max_side - h
    pad_w = max_side - w
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    video_padded = F.pad(video_tensor, pad=padding, mode='constant', value=0)
    return F.interpolate(
        video_padded, size=(_CLIP_SIZE, _CLIP_SIZE),
        mode='bilinear', align_corners=False,
    )


def extract_video_frames(video_path: str):
    """
    Decode clip_chunk and sync_chunk from video entirely in memory.

    Returns:
        clip_chunk : (L, H, W, C) float32 [0, 1]
        sync_chunk : (L, C, H, W) float32 normalized
        duration   : float (seconds)
    """
    sync_transform = _MODELS["sync_transform"]
    assert sync_transform is not None, "Call load_all_models() first."

    duration_sec = get_video_duration(video_path)

    reader = StreamingMediaDecoder(video_path)
    reader.add_basic_video_stream(
        frames_per_chunk=int(_CLIP_FPS * duration_sec),
        frame_rate=_CLIP_FPS,
        format='rgb24',
    )
    reader.add_basic_video_stream(
        frames_per_chunk=int(_SYNC_FPS * duration_sec),
        frame_rate=_SYNC_FPS,
        format='rgb24',
    )
    reader.fill_buffer()
    data_chunk = reader.pop_chunks()

    clip_chunk = data_chunk[0]
    sync_chunk = data_chunk[1]

    if clip_chunk is None:
        raise RuntimeError("CLIP video stream returned None")
    if sync_chunk is None:
        raise RuntimeError("Sync video stream returned None")

    # ---- clip_chunk ----
    clip_expected = int(_CLIP_FPS * duration_sec)
    clip_chunk = clip_chunk[:clip_expected]
    if clip_chunk.shape[0] < clip_expected:
        pad_n = clip_expected - clip_chunk.shape[0]
        clip_chunk = torch.cat(
            [clip_chunk, clip_chunk[-1:].repeat(pad_n, 1, 1, 1)], dim=0
        )
    clip_chunk = pad_to_square(clip_chunk)
    clip_chunk = clip_chunk.permute(0, 2, 3, 1)
    clip_chunk = mediapy.to_float01(clip_chunk)

    # ---- sync_chunk ----
    sync_expected = int(_SYNC_FPS * duration_sec)
    sync_chunk = sync_chunk[:sync_expected]
    if sync_chunk.shape[0] < sync_expected:
        pad_n = sync_expected - sync_chunk.shape[0]
        sync_chunk = torch.cat(
            [sync_chunk, sync_chunk[-1:].repeat(pad_n, 1, 1, 1)], dim=0
        )
    sync_chunk = sync_transform(sync_chunk)

    log.info(f"clip_chunk: {clip_chunk.shape}, sync_chunk: {sync_chunk.shape}")
    return clip_chunk, sync_chunk, duration_sec


# ==================== Feature Extraction ====================

def extract_features(clip_chunk: torch.Tensor, sync_chunk: torch.Tensor, caption: str) -> dict:
    """Reuses globally loaded FeaturesUtils — no reload per call."""
    model = _MODELS["feature_extractor"]
    assert model is not None, "FeaturesUtils not initialized."

    info = {}
    with torch.no_grad():
        text_features = model.encode_t5_text([caption])
        info['text_features'] = text_features[0].cpu()

        clip_input = torch.from_numpy(clip_chunk).unsqueeze(0)
        video_feat, frame_embed, _, text_feat = \
            model.encode_video_and_text_with_videoprism(clip_input, [caption])

        info['global_video_features'] = torch.tensor(np.array(video_feat)).squeeze(0).cpu()
        info['video_features']        = torch.tensor(np.array(frame_embed)).squeeze(0).cpu()
        info['global_text_features']  = torch.tensor(np.array(text_feat)).squeeze(0).cpu()

        sync_input = sync_chunk.unsqueeze(0).to(DEVICE)
        info['sync_features'] = model.encode_video_with_sync(sync_input)[0].cpu()

    return info


# ==================== Build Meta ====================

def build_meta(info: dict, duration: float, caption: str):
    latent_length = round(SAMPLE_RATE * duration / 2048)
    audio_latent  = torch.zeros((1, 64, latent_length), dtype=torch.float32)

    meta = dict(info)
    meta['id']          = 'demo'
    meta['relpath']     = 'demo.npz'
    meta['path']        = 'demo.npz'
    meta['caption_cot'] = caption
    meta['video_exist'] = torch.tensor(True)

    return audio_latent, meta


# ==================== Diffusion Sampling ====================

def run_diffusion(audio_latent: torch.Tensor, meta: dict, duration: float) -> torch.Tensor:
    """Reuses globally loaded diffusion model — no reload per call."""
    from PrismAudio.inference.sampling import sample, sample_discrete_euler
    import time

    diffusion    = _MODELS["diffusion"]
    model_config = _MODELS["model_config"]
    assert diffusion is not None, "Diffusion model not initialized."

    diffusion_objective = model_config["model"]["diffusion"]["diffusion_objective"]
    latent_length       = round(SAMPLE_RATE * duration / 2048)

    meta_on_device = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in meta.items()
    }
    metadata = (meta_on_device,)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            conditioning = diffusion.conditioner(metadata, DEVICE)

        video_exist = torch.stack([item['video_exist'] for item in metadata], dim=0)
        if 'metaclip_features' in conditioning:
            conditioning['metaclip_features'][~video_exist] = \
                diffusion.model.model.empty_clip_feat
        if 'sync_features' in conditioning:
            conditioning['sync_features'][~video_exist] = \
                diffusion.model.model.empty_sync_feat

        cond_inputs = diffusion.get_conditioning_inputs(conditioning)
        noise       = torch.randn([1, diffusion.io_channels, latent_length]).to(DEVICE)

        with torch.amp.autocast('cuda'):
            if diffusion_objective == "v":
                fakes = sample(
                    diffusion.model, noise, 24, 0,
                    **cond_inputs, cfg_scale=5, batch_cfg=True,
                )
            elif diffusion_objective == "rectified_flow":
                t0    = time.time()
                fakes = sample_discrete_euler(
                    diffusion.model, noise, 24,
                    **cond_inputs, cfg_scale=5, batch_cfg=True,
                )
                log.info(f"Sampling time: {time.time() - t0:.2f}s")

            if diffusion.pretransform is not None:
                fakes = diffusion.pretransform.decode(fakes)

    return (
        fakes.to(torch.float32)
             .div(torch.max(torch.abs(fakes)))
             .clamp(-1, 1)
             .mul(32767)
             .to(torch.int16)
             .cpu()
    )


# ==================== Full Inference Pipeline ====================

def generate_audio(video_file, caption: str):
    start_time =time.time()

    """
    Gradio generator function (yields status + result progressively).

    Yields:
        (status_str, combined_video_path_or_None)
    """
    # ---- Basic validation ----
    if video_file is None:
        yield "❌ Please upload a video file first.", None
        return
    if not caption or caption.strip() == "":
        caption=""

    caption = caption.strip()
    logs    = []

    def log_step(msg: str):
        log.info(msg)
        logs.append(msg)
        return "\n".join(logs)

    # ---- Working directory (auto-cleaned on exit) ----
    work_dir = tempfile.mkdtemp(dir=os.environ["GRADIO_TEMP_DIR"], prefix="PrismAudio_")

    try:
        # ---- Step 1: Convert / copy to mp4 ----
        status = log_step("📹 Step 1: Preparing video...")

        yield status, None

        src_ext  = os.path.splitext(video_file)[1].lower()
        mp4_path = os.path.join(work_dir, "input.mp4")

        if src_ext != ".mp4":
            log_step("   Converting to mp4...")
            ok, err = convert_to_mp4(video_file, mp4_path)
            if not ok:
                yield log_step(f"❌ Video conversion failed:\n{err}"), None
                return
        else:
            shutil.copy(video_file, mp4_path)
        log_step("   Video ready.")

        # ---- Step 2: Validate duration ----
        status = log_step("📹 Step 2: Checking video duration...")
        yield status, None

        duration = get_video_duration(mp4_path)
        log_step(f"   Duration: {duration:.2f}s")

        # ---- Step 3: Extract video frames ----
        status = log_step("🎞️  Step 3: Extracting video frames (clip & sync)...")
        yield status, None

        clip_chunk, sync_chunk, duration = extract_video_frames(mp4_path)
        log_step(f"   clip_chunk : {tuple(clip_chunk.shape)}")
        log_step(f"   sync_chunk : {tuple(sync_chunk.shape)}")

        # ---- Step 4: Extract model features ----
        status = log_step("🧠 Step 4: Extracting text / video / sync features...")
        yield status, None

        info = extract_features(clip_chunk, sync_chunk, caption)
        log_step(f"   text_features         : {tuple(info['text_features'].shape)}")
        log_step(f"   global_video_features : {tuple(info['global_video_features'].shape)}")
        log_step(f"   video_features        : {tuple(info['video_features'].shape)}")
        log_step(f"   global_text_features  : {tuple(info['global_text_features'].shape)}")
        log_step(f"   sync_features         : {tuple(info['sync_features'].shape)}")

        # ---- Step 5: Build inference batch ----
        status = log_step("📦 Step 5: Building inference batch...")
        yield status, None

        audio_latent, meta = build_meta(info, duration, caption)
        log_step(f"   audio_latent : {tuple(audio_latent.shape)}")

        # ---- Step 6: Diffusion sampling ----
        status = log_step("🎵 Step 6: Running diffusion sampling...")
        yield status, None

        generated_audio = run_diffusion(audio_latent, meta, duration)
        log_step(f"   Generated audio shape : {tuple(generated_audio.shape)}")

        # ---- Step 7: Save generated audio (temp) ----
        status = log_step("💾 Step 7: Saving generated audio...")
        yield status, None

        audio_path = os.path.join(work_dir, "generated_audio.wav")
        torchaudio.save(
            audio_path,
            generated_audio[0],  # (1, T)
            SAMPLE_RATE,
        )
        log_step(f"   Audio saved: {audio_path}")

        # ---- Step 8: Mux audio into original video ----
        status = log_step("🎬 Step 8: Merging audio into video...")
        yield status, None

        combined_path = os.path.join(work_dir, "output_with_audio.mp4")
        ok, err = combine_audio_video(mp4_path, audio_path, combined_path)
        if not ok:
            yield log_step(f"❌ Failed to combine audio and video:\n{err}"), None
            return

        log_step("✅ Done! Audio and video merged successfully.")
        yield "\n".join(logs), combined_path

    except Exception as e:
        log_step(f"❌ Unexpected error: {str(e)}")
        log.exception(e)
        yield "\n".join(logs), None
    
    end_time =time.time()
    print("cost: ",end_time-start_time)

    # Note: work_dir is NOT deleted here so Gradio can serve the output file.
    # Gradio manages its own GRADIO_TEMP_DIR cleanup on restart.


# ==================== Gradio UI ====================

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="PrismAudio - Video to Audio Generation",
        theme=gr.themes.Soft(),
        css="""
        .title { text-align:center; font-size:2em; font-weight:bold; margin-bottom:.2em; }
        .sub   { text-align:center; color:#666; margin-bottom:1.5em; }
        .mono  { font-family:monospace; font-size:.85em; }
        """,
    ) as demo:

        gr.HTML('<div class="title">🎵 PrismAudio</div>')
        gr.HTML(
            '<div class="sub">'
            'Upload a video and a text prompt — '
            'the generated audio will be merged back into your video.'
            '</div>'
        )

        # ======================================================
        # Row 1 — Inputs
        # ======================================================
        with gr.Row():

            # ---------- Left: inputs ----------
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")

                video_input = gr.Video(
                    label="Upload Video",
                    sources=["upload"],
                    height=300,
                )
                caption_input = gr.Textbox(
                    label="Caption / Prompt",
                    placeholder=(
                        "Describe the audio you want to generate, e.g.:\n"
                        "A dog barking in the park with wind blowing"
                    ),
                    lines=4,
                    max_lines=8,
                )
                with gr.Row():
                    clear_btn  = gr.Button("🗑️ Clear",         variant="secondary", scale=1)
                    submit_btn = gr.Button("🚀 Generate Audio", variant="primary",   scale=2)

            # ---------- Right: live log ----------
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Run Log")
                log_output = gr.Textbox(
                    label="",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                    elem_classes=["mono"],
                )
                gr.Markdown("### 📤 Output")
                video_output = gr.Video(
                    label="Video + Generated Audio",
                    interactive=False,
                    height=300,
                )


        # ======================================================
        # Instructions
        # ======================================================
        with gr.Accordion("📖 Instructions", open=False):
            gr.Markdown(f"""
**Steps**
1. Upload a video file (mp4 / avi / mov / etc.).
2. Enter a text prompt describing the desired audio content.
3. Click **🚀 Generate Audio** and watch the log on the right for progress.
4. The output video (original visuals + generated audio) appears below when done.
            """)

        # ======================================================
        # Event bindings
        # ======================================================
        submit_btn.click(
            fn=generate_audio,
            inputs=[video_input, caption_input],
            outputs=[log_output, video_output],
            show_progress=True,
        )

        def clear_all():
            return None, "", "", None

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[video_input, caption_input, log_output, video_output],
        )

    return demo


# ==================== Entry Point ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PrismAudio Gradio App")
    parser.add_argument("--server_name", type=str, default="0.0.0.0",
                        help="Gradio server host")
    parser.add_argument("--server_port", type=int, default=7860,
                        help="Gradio server port")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    # ---- Check model files ----
    missing = []
    for name, path in [
        ("Model Config",    MODEL_CONFIG_PATH),
        ("Checkpoint",      CKPT_PATH),
        ("VAE Checkpoint",  VAE_CKPT_PATH),
        ("Synchformer",     SYNCHFORMER_CKPT_PATH),
    ]:
        if not os.path.exists(path):
            missing.append(f"  ⚠️  {name}: {path}")

    if missing:
        log.warning("The following model files were not found — please check your paths:")
        for m in missing:
            log.warning(m)
    else:
        log.info("✅ All model files found.")

    # ⭐ Load all models once at startup
    load_all_models()

    demo = build_ui()
    demo.queue(max_size=3)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )
