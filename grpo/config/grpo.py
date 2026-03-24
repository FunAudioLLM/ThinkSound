import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    config.per_prompt_stat_tracking = True
    return config

def get_config(name):
    return globals()[name]()




def general_thinksound_4gpus():
    gpu_number = 4
    config = compressibility()
    config.train.cfg = False
    config.use_lora = False
    config.num_workers=24
    config.audio_sample_rate = 44100
    config.cfg_dropout_prob = 0.1

    config.train.learning_rate = 1e-5
    config.train_num_step = None
    config.sample.num_steps = 24
    config.sample.eval_num_steps = 24
    config.sample.train_num_steps = 2
    config.sample.mini_num_image_per_prompt = 8
    config.train.batch_size = config.sample.mini_num_image_per_prompt

    config.resolution = 512
    config.sample.train_batch_size = 1
    config.sample.num_audio_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_audio_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    config.pretransform_ckpt_path="ckpts/vae.ckpt"
    config.dataset_config="PrismAudio/configs/multimodal_dataset_demo_prismaudio.json"
    config.model_config="PrismAudio/configs/model_configs/prismaudio.json"
    config.ref_model = "ckpts/prismaudio.ckpt"
    config.ckpt_dir = 'ckpts/prismaudio.ckpt'
    config.save_dir = 'logs/results'
    config.reward_fn = {
        "ms_clap":1.0
    }
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 20 # epoch
    config.eval_freq = 10

    
    config.per_prompt_stat_tracking = True
    return config



def general_thinksound_8gpus():
    gpu_number = 8
    config = compressibility()
    config.train.cfg = False
    config.use_lora = False
    
    config.num_workers=24
    config.audio_sample_rate = 44100
    config.cfg_dropout_prob = 0.1

    config.train.learning_rate = 1e-5
    config.train_num_step = None
    config.sample.num_steps = 24
    config.sample.eval_num_steps = 24
    
    config.sample.mini_num_image_per_prompt = 8
    config.train.batch_size = config.sample.mini_num_image_per_prompt

    config.resolution = 512
    config.sample.train_batch_size = 1
    config.sample.num_audio_per_prompt = 16
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_audio_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.multi_prompts = None
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 10
    config.sample.train_num_steps = 2
    config.pretransform_ckpt_path="ckpts/vae.ckpt"
    config.dataset_config="PrismAudio/configs/multimodal_dataset_demo_prismaudio.json"
    config.model_config="PrismAudio/configs/model_configs/prismaudio.json"
    config.ref_model = "ckpts/prismaudio.ckpt"
    config.ckpt_dir = 'ckpts/prismaudio.ckpt'
    config.save_dir = 'logs/results'
    config.reward_fn = {
        "synch_reward":0.6 ,
        "ms_clap":1.0,
        "itd_reward":0.4,
        "meta_reward":0.1,
    }
    config.global_step = 0
    config.resume_epoch = 0


    #config.train.lora_path = "logs/clap/checkpoints/checkpoint-140/lora"
    config.per_prompt_stat_tracking = True
    return config

