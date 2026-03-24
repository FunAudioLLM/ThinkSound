# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
from typing import Any, Dict, List, Optional, Union
import torch
import random
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .flow_with_logprob import sde_step_with_logprob






def repeat_dict(d, repeat, dim=0):

    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_d[k] = repeat_dict(v, repeat, dim=dim)
        elif isinstance(v, torch.Tensor):
            new_d[k] = v.repeat([repeat if i==dim else 1 for i in range(v.ndim)])
        elif isinstance(v, List):
            new_d[k] = [x.repeat([repeat if i==dim else 1 for i in range(x.ndim)]) for x in v]
        else:
            new_d[k] = v
    return new_d





@torch.no_grad()
def thinksound_with_logprob(
    diffusion,
    conditioning,
    reals,
    config,
    extra_args,
    num_inference_steps: int = 24,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    scheduler = None,
    noise_level: float = 0.7,
    device = 0,
    process_index =0 ,
    mini_num_image_per_prompt: int = 1,
    train_num_steps: int = 1,
    random_timestep: Optional[int] = None,
    sample_type = "isotropic"
):

    #print("using sample type: ",sample_type)
    batch_size, length = reals.shape[0], reals.shape[2]


    if batch_size > 1:
        noise_list = []
        for _ in range(batch_size):
            noise_1 = torch.randn(
                [1, diffusion.io_channels, length],
                device=device,
                generator=generator
            )
            noise_list.append(noise_1)
        latents = torch.cat(noise_list, dim=0)
    else:
        latents = torch.randn(
            [batch_size, diffusion.io_channels, length],
            device=device,
            generator=generator
        )




    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )

    random.seed(process_index)
    if random_timestep is None:
        random_timestep = random.randint(0, num_inference_steps/2)

    # 6. Prepare image embeddings
    all_latents = []
    all_log_probs = []
    all_timesteps = []

    # 7. Denoising loop
    #if config.multi_prompts and mini_num_image_per_prompt > 1:
    #    original_text = []
    #    for i in range(len(conditioning["text_features"][0])):
    #        using_text = random.randint(0, config.multi_prompts)
    ##        original_text.append(conditioning["text_features"][0][i].clone())
     #       if using_text==0:
    #            continue
    #        conditioning["text_features"][0][i] = conditioning["text_features_"+str(using_text)][0][i]


    for i, t in enumerate(timesteps):
        if i < random_timestep:
            cur_noise_level = 0
        elif i == random_timestep:

            #if config.multi_prompts and mini_num_image_per_prompt > 1:
            #    for j in range(len(conditioning["text_features"][0])):
            #        conditioning["text_features"][0][j] = original_text[j]

            cur_noise_level= noise_level
          
            latents = latents.repeat(mini_num_image_per_prompt, 1, 1)
            conditioning = repeat_dict(conditioning, mini_num_image_per_prompt, dim=0)
            #prompt_embeds = prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
            #pooled_prompt_embeds = pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
            #negative_prompt_embeds = negative_prompt_embeds.repeat(mini_num_image_per_prompt, 1, 1)
            #negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(mini_num_image_per_prompt, 1)
            #print("repeated latents",latents[0],flush=True)
            #print("repeated latents",latents[batch_size],flush=True)
            #print("window",random_timestep,train_num_steps,flush=True)
            all_latents.append(latents)
        elif i > random_timestep and i < random_timestep + train_num_steps:
            cur_noise_level = noise_level
        else:
            cur_noise_level= 0

        # expand the latents if we are doing classifier free guidance
        latent_model_input = latents
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latent_model_input.shape[0])/ scheduler.config.num_train_timesteps


        #print("step",i, conditioning['sync_features'][0].shape)
        #print(latent_model_input.shape)
        noise_pred = diffusion(
            latent_model_input,
            timestep,
            cond=conditioning,
            cfg_dropout_prob = config.cfg_dropout_prob,
            **extra_args
        )

        latents_dtype = latents.dtype

        latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            scheduler, 
            noise_pred.float(), 
            t.unsqueeze(0), 
            latents.float(),
            noise_level=cur_noise_level,
            sample_type = sample_type 
        )
        #print("i",i,"random_timestep + train_num_steps",random_timestep + train_num_steps)
        if i >= random_timestep and i < random_timestep + train_num_steps:
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_timesteps.append(t.repeat(len(latents)))
            
 
        

    if diffusion.pretransform is not None:
        fakes = diffusion.pretransform.decode(latents)

    audios = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).clamp(-1, 1).mul(32767).to(torch.int32)


    return audios, all_latents, all_log_probs ,all_timesteps