import pytorch_lightning as pl
import sys, gc
import random
import torch
import torchaudio
import typing as tp
import wandb

from aeiou.viz import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_file
from torch import optim
from torch.nn import functional as F
from ..inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ..models.diffusion import DiffusionModelWrapper, ConditionedDiffusionModelWrapper

from ..models.lm import AudioLMContinuousModelWrapper
from .utils import create_optimizer_from_config, create_scheduler_from_config

class AudioLMContinuousModelTrainingWrapper(pl.LightningModule):
    def __init__(
            self, 
            model: AudioLanguageModelWrapper, 
            lr = 1e-4, 
            diffusion_objective: tp.Literal["rectified_flow", "v"] = "v",
            timestep_sampler: tp.Literal["uniform", "logit_normal"] = "uniform",
            use_ema=False, 
            ema_copy=None,
            optimizer_configs: dict = None,
            diffusion_batch_mul=4,
            pre_encoded=False
        ):
        super().__init__()

        self.model = model
        self.diffusion = diffusion
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.model.pretransform.requires_grad_(False)

        self.timestep_sampler = timestep_sampler

        self.diffusion_objective = model.diffusion_objective

        loss_modules = [
            MSELoss("v",
                     "targets",
                     weight=1.0,
                     name="mse_loss"
                )
        ]

        self.losses = MultiLoss(loss_modules)

        self.model_ema = None
        if use_ema:
            self.model_ema = EMA(self.model, ema_model=ema_copy, beta=0.99, update_every=10)

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "lm": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (0.9, 0.95),
                            "weight_decay": 0.1
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        
        self.optimizer_configs = optimizer_configs

        self.diffusion_batch_mul = diffusion_batch_mul

        self.pre_encoded = pre_encoded

    def configure_optimizers(self):
        lm_opt_config = self.optimizer_configs['lm']
        opt_lm = create_optimizer_from_config(lm_opt_config['optimizer'], self.model.parameters())

        if "scheduler" in lm_opt_config:
            sched_lm = create_scheduler_from_config(lm_opt_config['scheduler'], opt_lm)
            sched_lm_config = {
                "scheduler": sched_lm,
                "interval": "step"
            }
            return [opt_lm], [sched_lm_config]

        return [opt_lm]
        
    # Copied and modified from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/solvers/musicgen.py under MIT license
    # License can be found in LICENSES/LICENSE_META.txt


    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        diffusion_input = reals

        loss_info = {}

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        if self.diffusion.pretransform is not None:
            if not self.pre_encoded:
                with torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
            else:            
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        loss_info["reals"] = diffusion_input

        padding_masks = []
        for md in metadata:
            if md["padding_mask"].ndim == 1:
                padding_masks.append(md["padding_mask"])
            else:
                padding_masks.append(md["padding_mask"][0])
            
        padding_masks = torch.stack(padding_masks, dim=0).to(self.device) # Shape (batch_size, sequence_length)

        condition_tensors = None

        # If the model is conditioned, get the conditioning tensors
        if self.model.conditioner is not None:
            with torch.cuda.amp.autocast():
                condition_tensors = self.model.conditioner(metadata, self.device)

        z = self.model.compute_logits(diffusion_input, condition_tensors=condition_tensors, cfg_dropout_prob=0.1)
        bsz, seq_len, _ = z.shape
        gt_inputs = diffusion_input.clone().detach()
        gt_inputs = gt_inputs.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(z.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(z.shape[0], device=self.device))

        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1-t, t

        # Combine the ground truth data and the noise
        alphas = alphas[:, None]
        sigmas = sigmas[:, None]
        
        noise = torch.randn_like(gt_inputs)
        noised_inputs = gt_inputs * alphas + noise * sigmas
        if self.diffusion_objective == "v":
            targets = noise * alphas - gt_inputs * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - gt_inputs
        cond = {}
        cond['z'] = z
        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_inputs, t, cond=cond)

            loss_info.update({
                "v": v,
                "targets": targets
            })

            loss, losses = self.losses()

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }


        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.model_ema is not None:
            self.model_ema.update()

    def export_model(self, path, use_safetensors=False):
        
        model = self.model_ema.ema_model if self.model_ema is not None else self.model

        if use_safetensors:
            save_file(model.state_dict(), path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)
        

class AudioLanguageModelDemoCallback(pl.Callback):loss_info
    def __init__(self, 
                 demo_every=2000,
                 num_demos=8,
                 sample_size=65536,
                 sample_rate=48000,
                 demo_conditioning: tp.Optional[tp.Dict[str, tp.Any]] = None,
                 demo_cfg_scales: tp.Optional[tp.List[int]] = [3, 5, 7],
                 **kwargs
    ):
        super().__init__()

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_samples = sample_size
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.demo_conditioning = demo_conditioning
        self.demo_cfg_scales = demo_cfg_scales

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: AudioLanguageModelTrainingWrapper, outputs, batch, batch_idx):        

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        module.eval()

        print(f"Generating demo")
        self.last_demo_step = trainer.global_step

        demo_length_tokens = self.demo_samples // module.model.pretransform.downsampling_ratio

        #demo_reals = batch[0][:self.num_demos]

        # if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
        #     demo_reals = demo_reals[0]

        #demo_reals_tokens = module.model.pretransform.tokenize(demo_reals)

        ##Limit to first 50 tokens
        #demo_reals_tokens = demo_reals_tokens[:, :, :50]

        try:
            print("Getting conditioning")

            for cfg_scale in self.demo_cfg_scales:

                model = module.model # module.model_ema.ema_model if module.model_ema is not None else module.model

                print(f"Generating demo for cfg scale {cfg_scale}")
                fakes = model.generate_audio(
                    batch_size=self.num_demos,
                    max_gen_len=demo_length_tokens, 
                    conditioning=self.demo_conditioning, 
                    #init_data = demo_reals_tokens,
                    cfg_scale=cfg_scale,
                    temp=1.0,
                    top_p=0.95
                )

                # Put the demos together
                fakes = rearrange(fakes, 'b d n -> d (b n)')

                log_dict = {}
                
                filename = f'demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                fakes = fakes / fakes.abs().max()
                fakes = fakes.type(torch.float32).clamp(-1, 1).mul(32767).type(torch.int16).cpu()
                torchaudio.save(filename, fakes, self.sample_rate)

                log_dict[f'demo_cfg_{cfg_scale}'] = wandb.Audio(filename,
                                                    sample_rate=self.sample_rate,
                                                    caption=f'Reconstructed')
            
                log_dict[f'demo_melspec_left_cfg_{cfg_scale}'] = wandb.Image(audio_spectrogram_image(fakes))

                trainer.logger.experiment.log(log_dict)

        except Exception as e:
            raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()