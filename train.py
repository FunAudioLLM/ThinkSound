from prefigure.prefigure import get_all_args, push_wandb_config
import json
import os
import torch
import torchaudio
# import pytorch_lightning as pl
import lightning as L
from lightning.pytorch.callbacks import Timer, ModelCheckpoint, BasePredictionWriter
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.tuner import Tuner
from lightning.pytorch import seed_everything
import random
from datetime import datetime
# from ThinkSound.data.dataset import create_dataloader_from_config
from ThinkSound.data.datamodule import DataModule
from ThinkSound.models import create_model_from_config
from ThinkSound.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from ThinkSound.training import create_training_wrapper_from_config, create_demo_callback_from_config
from ThinkSound.training.utils import copy_state_dict

class ExceptionCallback(Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval='batch'):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module, predictions, batch_indices, batch, batch_idx, dataloader_idx):

        audios = predictions
        ids = [item['id'] for item in batch[1]]
        # 获取当前日期
        current_date = datetime.now()

        # 格式化日期为 'MMDD' 形式
        formatted_date = current_date.strftime('%m%d')
        if trainer.ckpt_path is None:
            global_step = pl_module.global_step // 1000
        else:
            global_step = int(trainer.ckpt_path.split("-step=")[-1].split(".")[0]) // 1000
        os.makedirs(os.path.join(self.output_dir, f'{formatted_date}_step{global_step}k'),exist_ok=True)
        for audio, id in zip(audios, ids):
            save_path = os.path.join(self.output_dir, f'{formatted_date}_step{global_step}k', f'{id}.wav')
            torchaudio.save(save_path, audio, 44100)

def main():

    args = get_all_args()

    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    # random.seed(seed)
    # torch.manual_seed(seed)
    seed_everything(seed, workers=True)
    print('########################')
    print(f'precision is {args.precision}')
    print('########################')
    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    # train_dl = create_dataloader_from_config(
    #     dataset_config, 
    #     batch_size=args.batch_size, 
    #     num_workers=args.num_workers,
    #     sample_rate=model_config["sample_rate"],
    #     sample_size=model_config["sample_size"],
    #     audio_channels=model_config.get("audio_channels", 2),
    # )
    dm = DataModule(
        dataset_config, 
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        repeat_num=args.repeat_num
    )

    model = create_model_from_config(model_config)

    ## speed by torch.compile
    if args.compile:
        model = torch.compile(model)

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path,prefix='diffusion.')) # autoencoder.  diffusion.
    
    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)
    # import ipdb
    # ipdb.set_trace()
    if args.pretransform_ckpt_path:
        load_vae_state = load_ckpt_state_dict(args.pretransform_ckpt_path, prefix='autoencoder.') 
        # new_state_dict = {k.replace("autoencoder.", ""): v for k, v in load_vae_state.items() if k.startswith("autoencoder.")}
        model.pretransform.load_state_dict(load_vae_state)
    
    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    wandb_logger = L.pytorch.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(training_wrapper)

    exc_callback = ExceptionCallback()

    if args.save_dir and isinstance(wandb_logger.experiment.id, str):
        checkpoint_dir = os.path.join(args.save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id, "checkpoints")
    else:
        checkpoint_dir = None

    # ckpt_callback = ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, monitor='val_loss', mode='min', save_top_k=14)
    ckpt_callback = ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, monitor='epoch', mode='max', save_top_k=14)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
    # audio_dir = os.path.join(args.save_dir, args.name, "audios")
    # pred_writer = CustomWriter(output_dir=audio_dir, write_interval="batch")
    timer = Timer(duration="00:16:00:00")
    demo_callback = create_demo_callback_from_config(model_config, demo_dl=dm)

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    push_wandb_config(wandb_logger, args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2, 
                                        contiguous_gradients=True, 
                                        overlap_comm=True, 
                                        reduce_scatter=True, 
                                        reduce_bucket_size=5e8, 
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True
                                        )
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto" 

    trainer = L.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback, timer],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=90,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0,
        check_val_every_n_epoch=2,
    )

    # query training/validation/test time (in seconds)
    # timer.time_elapsed("train")
    # timer.start_time("validate")
    # tuner = Tuner(trainer)
    # Auto-scale batch size by growing it exponentially (default)
    # tuner.scale_batch_size(training_wrapper, mode="power")
    # tuner.lr_find(training_wrapper)
    # trainer.tune(training_wrapper, train_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)
    # trainer.validate(training_wrapper, dm)
    trainer.fit(training_wrapper, dm, ckpt_path=args.ckpt_path if args.ckpt_path else None)
    # trainer.predict(training_wrapper, dm, return_predictions=False)

if __name__ == '__main__':
    main()