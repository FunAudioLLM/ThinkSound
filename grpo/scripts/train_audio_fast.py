from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import grpo.fast_grpo.rewards
from grpo.fast_grpo.stat_tracking import PerPromptStatTracker
from grpo.fast_grpo.flow_with_logprob_fast import thinksound_with_logprob
from grpo.fast_grpo.flow_with_logprob import sde_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from grpo.fast_grpo.ema import EMAModuleWrapper
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
torch.autograd.set_detect_anomaly(True)
import copy

def int32_to_int16(audio_int32: torch.Tensor) -> torch.Tensor:
    if audio_int32.dtype != torch.int32:
        raise ValueError("Input type must be int32")

    return audio_int32.clamp(-32768, 32767).to(torch.int16)



tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

MAX_STEPS=1000

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def string_to_int_tensor(s, dim=8, max_int=255, dtype=torch.int64):
    h = hashlib.md5(s.encode('utf-8')).digest()
    ints = [b % max_int for b in h[:dim]]
    tensor = torch.tensor(ints, dtype=dtype)
    return tensor

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def compute_input_embeddings(metadata, diffusion, device):

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            conditioning = diffusion.conditioner(metadata, device)


    return conditioning

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(diffusion, MMdit, scheduler, sample, j, conditioning, extra_args, config):

    #latent_j = sample["latents"][:, j].clone().detach().requires_grad_(True)

    cond_inputs = diffusion.get_conditioning_inputs(conditioning)
    

    #noise_pred = MMdit(latent_j, sample["timesteps"][:, j], **cond_inputs_safe,
    #                cfg_dropout_prob=config.cfg_dropout_prob, **extra_args)
    
    
    noise_pred = MMdit(sample["latents"][:, j], sample["timesteps"][:, j]/MAX_STEPS, **cond_inputs, cfg_dropout_prob = config.cfg_dropout_prob, **extra_args)
    #noise_pred = diffusion(sample["latents"][:, j], sample["timesteps"][:, j], cond=conditioning, cfg_dropout_prob = config.cfg_dropout_prob, **extra_args)
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(test_dataloader, diffusion, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters,scheduler):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    #neg_prompt_embed, neg_pooled_prompt_embed = compute_input_embeddings([""],device=accelerator.device)

    #sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    #sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        reals, metadata = test_batch

        conditioning = compute_input_embeddings(
            metadata, 
            diffusion,
            device=accelerator.device
        )
        # The last batch may not be full batch_size

        with autocast():
            with torch.no_grad():
                audio, _, _, _ = thinksound_with_logprob(
                    diffusion,
                    conditioning,
                    reals,
                    config,
                    {"cfg_scale": 5},
                    noise_level=0,
                    mini_num_image_per_prompt=1,
                    process_index=accelerator.process_index,
                    device=accelerator.device,
                    num_inference_steps = config.sample.eval_num_steps,
                    scheduler=scheduler,
                    
                )
        #print(audio.shape)
        #print(metadata)
        rewards = executor.submit(reward_fn, int32_to_int16(audio) ,[meta['caption_cot'] for meta in metadata], metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()
        
        print(rewards,flush=True)

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_audio_gather = accelerator.gather(audio).cpu()

    


    last_batch_prompts_gather = [item['id'] for item in metadata]

    
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_audio_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                audio = last_batch_audio_gather[index]
                #print(audio.shape)
                #print(config.audio_sample_rate)
                wav_path = os.path.join(tmpdir, f"{idx}.wav")
                import torchaudio
                
                #print("audio",int32_to_int16(audio))
                torchaudio.save(wav_path, int32_to_int16(audio), sample_rate=config.audio_sample_rate)
                #torchaudio.save("demo.wav", int32_to_int16(audio), sample_rate=config.audio_sample_rate)


            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]

            for key, value in all_rewards.items():
                print(key, value.shape)

            wandb.log(
                {
                    "eval_audios": [
                        wandb.Audio(
                            os.path.join(tmpdir, f"{idx}.wav"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


def export_model(diffusion, save_dir, global_step, accelerator, ema=None, trainable_params=None, epoch=None):
    os.makedirs(save_dir, exist_ok=True)
    if epoch is not None:
        save_path = os.path.join(save_dir, f"checkpoints_epoch_{epoch}_step_{global_step}.ckpt")
    else:
        save_path = os.path.join(save_dir, f"checkpoints_{global_step}.ckpt")
    if accelerator.is_main_process:
        if ema is not None:
            ema.copy_ema_to(trainable_params, store_temp=True)

        model_to_save = accelerator.unwrap_model(diffusion.model)
        model_to_save = model_to_save._orig_mod if is_compiled_module(model_to_save) else model_to_save



        state_dict = model_to_save.state_dict()

        prefixed_state_dict = {f"model.{k}": v for k, v in state_dict.items()}

        for k, v in diffusion.state_dict().items():
            if not k.startswith("model."):
                if k.startswith("conditioner.module."):
                    k = k.replace("conditioner.module.", "conditioner.")
                prefixed_state_dict[k] = v

        torch.save(prefixed_state_dict, save_path)

        if ema is not None:
            ema.copy_temp_to(trainable_params)



def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model



def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

from typing import List
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


import torch

def recursive_cat(samples, dim=0, path=None):
    if path is None:
        path = []

    first = samples[0]

    if isinstance(first, torch.Tensor):
        try:
            return torch.cat(samples, dim=dim)
        except RuntimeError as e:
            raise RuntimeError(f"Tensor cat failed at path {'/'.join(map(str, path))}: {e}") from e

    elif isinstance(first, dict):
        return {
            k: recursive_cat([s[k] for s in samples], dim=dim, path=path + [k])
            for k in first
        }

    elif isinstance(first, list):
        return [
            recursive_cat([s[i] for s in samples], dim=dim, path=path + [i])
            for i in range(len(first))
        ]

    else:
        raise TypeError(f"Unsupported type {type(first)} at path {'/'.join(map(str, path))}")


def is_empty(x):
    if isinstance(x, torch.Tensor):
        return x.numel() == 0
    elif isinstance(x, dict):
        return all(is_empty(v) for v in x.values())
    elif isinstance(x, (list, tuple)):
        return all(is_empty(v) for v in x)
    else:
        raise TypeError(f"Unsupported type {type(x)} in is_empty")

def batch_reshape(x, batch_dim):
    """
    x: Tensor / dict / list / tuple
    batch_dim: int, 每个 batch 的大小，用于 reshape
    """
    if isinstance(x, torch.Tensor):
        return x.reshape(-1, batch_dim, *x.shape[1:])
    elif isinstance(x, dict):
        return {k: batch_reshape(v, batch_dim) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [batch_reshape(v, batch_dim) for v in x]
    else:
        raise TypeError(f"Unsupported type {type(x)} in batch_reshape")


def index_nested(x, i):
    if isinstance(x, torch.Tensor):
        return x[i]
    elif isinstance(x, dict):
        return {k: index_nested(v, i) for k, v in x.items()}
    elif isinstance(x, list):
        return [index_nested(v, i) for v in x]
    else:
        raise TypeError(f"Unsupported type {type(x)} in index_nested")


def get_length(x):
    if isinstance(x, torch.Tensor):
        return x.shape[0]
    elif isinstance(x, dict):
        return get_length(next(iter(x.values())))
    elif isinstance(x, list):
        return get_length(x[0])
    else:
        raise TypeError(f"Unsupported type {type(x)} in get_length")

from typing import Union

def recursive_slice(obj: Union[dict, list, torch.Tensor], start: int, end: int):
    if isinstance(obj, torch.Tensor):
        return obj[start:end]
    elif isinstance(obj, list):
        return [recursive_slice(x, start, end) for x in obj]
    elif isinstance(obj, dict):
        return {k: recursive_slice(v, start, end) for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported type {type(obj)}")

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config
    
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    
    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )

    if accelerator.is_main_process:
        wandb.init(project="grpo_fast",)
        #wandb.init(project="test", mode="offline")
        # accelerator.init_trackers(
        #     project_name="grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    with open(config.model_config) as f:
        model_config = json.load(f)

    # load scheduler, tokenizer and models.
    from PrismAudio.models import create_model_from_config
    from PrismAudio.models.utils import load_ckpt_state_dict
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=MAX_STEPS,
        use_dynamic_shifting=False
    )

    diffusion = create_model_from_config(model_config)

    if config.ckpt_dir is not None:
        diffusion.load_state_dict(torch.load(config.ckpt_dir))


    if config.pretransform_ckpt_path is not None:
        load_vae_state = load_ckpt_state_dict(config.pretransform_ckpt_path, prefix='autoencoder.') 
        diffusion.pretransform.load_state_dict(load_vae_state)

    # freeze parameters of models to save more memory


    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype

    diffusion.pretransform.requires_grad_(False)

    diffusion.model.requires_grad_(not config.use_lora)

    diffusion.pretransform.to(accelerator.device, dtype=torch.float32)

    
    diffusion.model.to(accelerator.device)



    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "joint_blocks.0.latent_block.linear1.weight",
        ]

        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            diffusion.model = PeftModel.from_pretrained(diffusion.model, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            diffusion.model.set_adapter("default")
        else:
            diffusion.model = get_peft_model(diffusion.model, transformer_lora_config)
    else:
        ref_diffusion = create_model_from_config(model_config)
        if config.ckpt_dir is not None:
            ref_diffusion.load_state_dict(torch.load(config.ref_model))
        if config.pretransform_ckpt_path is not None:
            load_vae_state = load_ckpt_state_dict(config.pretransform_ckpt_path, prefix='autoencoder.') 
            ref_diffusion.pretransform.load_state_dict(load_vae_state)
        ref_diffusion.pretransform.requires_grad_(False)
        ref_diffusion.model.requires_grad_(False)
        ref_diffusion.pretransform.to(accelerator.device, dtype=torch.float32)
        





    MMdit = diffusion.model
    MMdit.model.empty_sync_feat.requires_grad_(False)
    MMdit.model.empty_clip_feat.requires_grad_(False)
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, MMdit.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn

    reward_fn = getattr(grpo.fast_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    #eval_reward_fn = getattr(fast_grpo.fast_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = reward_fn

    from PrismAudio.data.datamodule import DataModule
    from PrismAudio.data.dataset import collation_fn
    dataset_config=config.dataset_config


    with open(dataset_config) as f:
        dataset_config = json.load(f)

    dm = DataModule(
        dataset_config, 
        batch_size=config.sample.train_batch_size,
        test_batch_size=config.sample.test_batch_size,
        num_workers=config.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    dm.setup('fit')
    train_dataset=dm.train_set
    test_dataset=dm.val_set
    # Create an infinite-loop DataLoader
    train_sampler = DistributedKRepeatSampler( 
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_audio_per_prompt//config.sample.mini_num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42
    )
    
    # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=12,
        collate_fn=collation_fn,
        # persistent_workers=True
    )

    # Create a regular DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=collation_fn,
        shuffle=False,
        num_workers=8,
    )
    

    if config.sample.num_audio_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    # initialize stat tracker
 

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    if config.use_lora:
        MMdit, optimizer, train_dataloader, test_dataloader = accelerator.prepare(MMdit, optimizer, train_dataloader, test_dataloader)
    else:
        diffusion.conditioner = accelerator.prepare(diffusion.conditioner)
        ref_diffusion_dit, MMdit, optimizer, train_dataloader, test_dataloader = accelerator.prepare(ref_diffusion.model ,MMdit, optimizer, train_dataloader, test_dataloader)
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0
    if hasattr(config, "resume_epoch"):
        epoch = config.resume_epoch
    else:
        epoch = 0
    if hasattr(config, "global_step"):
        global_step = config.global_step
    else:
        global_step = 0

    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        diffusion.model.eval()
        if epoch % config.eval_freq == 0:
            eval(test_dataloader, diffusion, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters,scheduler)
        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            if config.use_lora:
                save_ckpt(config.save_dir, MMdit, global_step, accelerator, ema, transformer_trainable_parameters, config)
            else:
                export_model(diffusion, config.save_dir , global_step, accelerator, ema=ema, trainable_params = transformer_trainable_parameters, epoch = epoch)

        #################### SAMPLING ####################
        diffusion.model.eval()
        samples = []

        reals_batch = []
        metadata_batch = []
        for i in range(config.sample.num_batches_per_epoch):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            reals, metadata = next(train_iter)
            
            reals_batch.append(reals)
            metadata_batch.append(metadata)

        gathered_batch = [x for sublist in metadata_batch for x in sublist]

        

        conditioning_batch = compute_input_embeddings(
            gathered_batch, 
            diffusion,
            device=accelerator.device
        )

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            #train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            reals = reals_batch[i]
            
            metadata = metadata_batch[i]

            
            batch_size = len(metadata)

            conditioning = recursive_slice(conditioning_batch,i*batch_size,i*batch_size + batch_size)


            
            ids = [string_to_int_tensor(meta["id"]).to(accelerator.device) for meta in metadata]
            
            captions = [meta["id"] for meta in metadata] * config.sample.mini_num_image_per_prompt 
            # sample
            if config.sample.same_latent:
                generator = create_generator(ids, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    audios, latents, log_probs, timesteps = thinksound_with_logprob(
                        diffusion,
                        conditioning,
                        reals,
                        config,
                        {"cfg_scale": 5},
                        num_inference_steps = config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        noise_level=config.sample.noise_level,
                        mini_num_image_per_prompt=config.sample.mini_num_image_per_prompt,
                        train_num_steps=config.sample.train_num_steps,
                        generator=generator,
                        process_index=accelerator.process_index,
                        scheduler=scheduler,
                        device=accelerator.device
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
            timesteps = torch.stack(timesteps, dim=1)

            # compute rewards asynchronously
            metadata_repeated = metadata * config.sample.mini_num_image_per_prompt
            caption_cot_repeated = [meta['caption_cot'] for meta in metadata_repeated]
            rewards = executor.submit(reward_fn, int32_to_int16(audios),caption_cot_repeated, metadata_repeated, only_strict=False)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": torch.stack(ids, dim=0).repeat(config.sample.mini_num_image_per_prompt,1),
                    "conditioning": repeat_dict(conditioning, config.sample.mini_num_image_per_prompt),
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()

            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
   

        samples = recursive_cat(samples, dim=0)

        

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(5, len(audios))
                sample_indices = random.sample(range(len(audios)), num_samples)

                for idx, i in enumerate(sample_indices):
                    audio = audios[i]
                    audio= audio.cpu()
                    wav_path = os.path.join(tmpdir, f"{idx}.wav")
                    import torchaudio
                    torchaudio.save(wav_path, int32_to_int16(audio), sample_rate=config.audio_sample_rate)

                sampled_ids = [captions[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "audio": [
                            wandb.Audio(
                                os.path.join(tmpdir, f"{idx}.wav"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_ids, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, config.sample.train_num_steps)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()


            prompt_list = [np.array2string(row) for row in prompt_ids]
            advantages = stat_tracker.update(prompt_list, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompt_list))
                print("len unique prompts", len(set(prompt_list)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompt_list, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        #total_batch_size, num_timesteps = samples["timesteps"].shape

        #mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        '''
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        '''

            

        

 
        #samples = {k: apply_mask(v, mask) for k, v in samples.items()}
        #samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # assert (
        #     total_batch_size
        #     == config.sample.train_batch_size * config.sample.num_batches_per_epoch
        # )


        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            #perm = torch.randperm(total_batch_size, device=accelerator.device)
            #samples = {k: apply_index(v,perm) for k, v in samples.items()}
            all_empty = is_empty(samples)
            if all_empty:
                continue

            samples_batched = batch_reshape(samples, total_batch_size // config.sample.num_batches_per_epoch)

            length = get_length(next(iter(samples_batched.values())))

            samples_batched = [
                {k: index_nested(v, i) for k, v in samples_batched.items()}
                for i in range(length)
            ]


            # train
            diffusion.model.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):

                conditioning = sample["conditioning"]



                for j in tqdm(
                    range(config.sample.train_num_steps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(MMdit):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(diffusion,MMdit, scheduler, sample, j, conditioning,{"cfg_scale": 5}, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    if config.use_lora:
                                        with MMdit.module.disable_adapter():
                                            _, _, prev_sample_mean_ref, _ = compute_log_prob(diffusion, MMdit, scheduler, sample, j, conditioning,{"cfg_scale": 5}, config)
                                    else:
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(ref_diffusion,ref_diffusion_dit, scheduler, sample, j, conditioning,{"cfg_scale": 5}, config)

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                MMdit.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)

