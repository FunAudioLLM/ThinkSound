# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL, AutoencoderDC
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from diffusers.schedulers import (
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import PreTrainedModel
import PIL
from tqdm import tqdm

from .model import MLLMInContextConfig, MLLMInContext
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)


class MetaQueryConfig(MLLMInContextConfig):
    model_type = "metaquery"

    def __init__(
        self,
        vae_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        input_size: int = 16,
        in_channels: int = 32,
        vae_downsample_f: int = 32,
        noise_scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        _gradient_checkpointing: bool = True,
        loss_type: str = "flow",
        num_metaqueries: int = 64,
        modules_to_freeze: tuple[str] = (),
        modules_to_unfreeze: tuple[str] = (),
        **kwargs,
    ):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.vae_id = vae_id
        self.input_size = input_size
        self.in_channels = in_channels
        self.vae_downsample_f = vae_downsample_f
        self.noise_scheduler_id = noise_scheduler_id
        self.scheduler_id = scheduler_id
        self._gradient_checkpointing = _gradient_checkpointing
        self.loss_type = loss_type
        self.num_metaqueries = num_metaqueries
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_unfreeze = modules_to_unfreeze


class MetaQuery(PreTrainedModel):
    config_class = MetaQueryConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

        self.model = MLLMInContext(MLLMInContextConfig(**config.to_dict()))
        self.loss_type = config.loss_type

        if "Sana" in config.vae_id:
            self.vae = AutoencoderDC.from_pretrained(config.vae_id, subfolder="vae")
        else:
            try:
                self.vae = AutoencoderKL.from_pretrained(config.vae_id)
            except:
                self.vae = AutoencoderKL.from_pretrained(config.vae_id, subfolder="vae")

        if self.loss_type == "flow":
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                config.noise_scheduler_id, subfolder="scheduler"
            )
        elif self.loss_type == "diff":
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                config.noise_scheduler_id, subfolder="scheduler"
            )
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")

        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            config.scheduler_id, subfolder="scheduler"
        )

        for module_name in config.modules_to_freeze:
            if "." in module_name:
                module = self
                for sub_module_name in module_name.split("."):
                    module = getattr(module, sub_module_name, None)
                    if module is None:
                        break
                else:
                    module.requires_grad_(False)
            else:
                module = getattr(self, module_name, None)
                if module is not None:
                    module.requires_grad_(False)

        for module_name in config.modules_to_unfreeze:
            if "." in module_name:
                module = self
                for sub_module_name in module_name.split("."):
                    module = getattr(module, sub_module_name, None)
                    if module is None:
                        break
                else:
                    module.requires_grad_(True)
            else:
                module = getattr(self, module_name, None)
                if module is not None:
                    module.requires_grad_(True)

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_tokenizer(self):
        return self.model.get_tokenizer()

    def get_tokenize_fn(self):
        return self.model.get_tokenize_fn()

    def forward(
        self, target, pixel_values=None, input_ids=None, attention_mask=None, **kwargs
    ):
        if self.vae is not None:
            if isinstance(self.vae, AutoencoderKL):
                latents = self.vae.encode(target).latent_dist.sample()
            elif isinstance(self.vae, AutoencoderDC):
                latents = self.vae.encode(target).latent
            else:
                raise ValueError(f"Unknown vae type {type(self.vae)}")
            if (
                "shift_factor" in self.vae.config
                and self.vae.config.shift_factor is not None
            ):
                latents = latents - self.vae.config.shift_factor
            latents = latents * self.vae.config.scaling_factor
        else:
            latents = target

        bsz = latents.shape[0]

        if (
            pixel_values is not None
            and hasattr(self.model, "mllm_type")
            and self.model.mllm_type == "qwenvl"
        ):
            pixel_values = pixel_values.squeeze(0)

        noise = torch.randn_like(latents, device=latents.device)

        if self.loss_type == "flow":
            weighting_scheme = "uniform"
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(
                device=latents.device
            )

            sigmas = self.get_sigmas(
                timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype
            )
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            prompt_embeds, attention_mask = self.model.encode_condition(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=kwargs.get("image_sizes", None),
            )

            model_pred = self.model(
                x=noisy_latents,
                timestep=timesteps,
                prompt_embeds=prompt_embeds,
                attention_mask=attention_mask,
            )

            target = noise - latents
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=weighting_scheme, sigmas=sigmas
            )
            loss = torch.mean(
                (
                    weighting.float() * (model_pred.float() - target.float()) ** 2
                ).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()

        elif self.loss_type == "diff":
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            prompt_embeds, attention_mask = self.model.encode_condition(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=kwargs.get("image_sizes", None),
            )

            noise_pred = self.model(
                x=noisy_latents,
                timestep=timesteps,
                prompt_embeds=prompt_embeds,
                attention_mask=attention_mask,
            )
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        return {"loss": loss}

    @torch.no_grad()
    def decode_latents(self, latents, normalize=True, return_tensor=False):
        if self.vae is not None:
            latents = latents / self.vae.config.scaling_factor
            if (
                "shift_factor" in self.vae.config
                and self.vae.config.shift_factor is not None
            ):
                latents = latents + self.vae.config.shift_factor
            samples = self.vae.decode(latents).sample
        else:
            samples = latents
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples

    def sample_images(
        self,
        caption="",
        input_images=None,
        guidance_scale: float = 3.0,
        image_guidance_scale: float = 1.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        negative_prompt="",
        enable_progress_bar=False,
        **kwargs,
    ):
        device = next(self.parameters()).device

        if not isinstance(caption, list):
            caption = [caption]
        if input_images is not None:
            if isinstance(input_images, list) and not isinstance(input_images[0], list):
                input_images = [[img] for img in input_images]
            elif isinstance(input_images, PIL.Image.Image):
                input_images = [[input_images]]
            assert isinstance(input_images, list) and all(
                isinstance(sublist, list) for sublist in input_images
            ), "input_images needs to be a nested list"

        bsz = len(caption)
        do_image_classifier_free_guidance = image_guidance_scale > 1.0

        tokenize_func = self.get_tokenize_fn()
        tokenizer = self.get_tokenizer()

        if input_images is not None:
            if do_image_classifier_free_guidance:
                caption = [negative_prompt] * bsz * 2 + caption
                input_images_null = [
                    (
                        [
                            PIL.Image.new("RGB", (img.size[0], img.size[1]))
                            for img in images
                        ]
                        if images
                        else None
                    )
                    for images in input_images
                ]
                input_images = input_images_null + input_images * 2
            else:
                caption = [negative_prompt] * bsz + caption
                input_images = input_images * 2
            input_ids, attention_mask, pixel_values, image_sizes = tokenize_func(
                tokenizer, caption, input_images
            )
        else:
            do_image_classifier_free_guidance = False
            caption = [negative_prompt] * bsz + caption
            input_ids, attention_mask = tokenize_func(tokenizer, caption)
            pixel_values = None
            image_sizes = None

        latent_size = self.config.input_size
        latent_channels = self.config.in_channels

        latents = randn_tensor(
            shape=(
                bsz * num_images_per_prompt,
                latent_channels,
                latent_size,
                latent_size,
            ),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        # set step values
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        else:
            self.scheduler.set_timesteps(num_inference_steps)

        # Repeat pixel_values and conditions for each image per prompt
        input_ids = input_ids.to(device=device).repeat_interleave(
            num_images_per_prompt, dim=0
        )
        attention_mask = attention_mask.to(device=device).repeat_interleave(
            num_images_per_prompt, dim=0
        )
        pixel_values = (
            pixel_values.to(device=device)
            .reshape(bsz, -1, *pixel_values.shape[1:])
            .repeat_interleave(num_images_per_prompt, dim=0)
            .flatten(0, 1)
            if pixel_values is not None
            else None
        )
        image_sizes = (
            image_sizes.to(device=device).repeat_interleave(
                num_images_per_prompt, dim=0
            )
            if image_sizes is not None
            else None
        )

        prompt_embeds, attention_mask = self.model.encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )
        # Convert to float32 before saving
        for t in tqdm(
            self.scheduler.timesteps,
            desc="Sampling images",
            disable=not enable_progress_bar,
        ):
            latent_model_input = torch.cat([latents] * (len(input_ids) // len(latents)))
            latent_model_input = latent_model_input.to(prompt_embeds.dtype)
            if hasattr(self.scheduler, "scale_model_input"):
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

            # predict noise model_output
            noise_pred = self.model(
                x=latent_model_input,
                timestep=t.unsqueeze(0)
                .expand(latent_model_input.shape[0])
                .to(latents.device),
                prompt_embeds=prompt_embeds,
                attention_mask=attention_mask,
            )

            # perform guidance
            if do_image_classifier_free_guidance:
                noise_pred_uncond, noise_pred_uncond_text, noise_pred = (
                    noise_pred.chunk(3)
                )
                noise_pred = (
                    noise_pred_uncond
                    + image_guidance_scale
                    * (noise_pred_uncond_text - noise_pred_uncond)
                    + guidance_scale * (noise_pred - noise_pred_uncond_text)
                )
            else:
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred - noise_pred_uncond
                )

            # compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        samples = self.decode_latents(
            latents.to(self.vae.dtype) if self.vae is not None else latents,
            return_tensor=return_tensor,
        )
        return samples
