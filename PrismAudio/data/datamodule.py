import lightning as L
from .dataset import LatentDataset, SampleDataset, VideoDataset, AudioDataset, MultiModalDataset, LocalDatasetConfig, collation_fn
import importlib
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,IterableDataset
import torch
from itertools import cycle

class AlternatingLoader(IterableDataset):
    """
    一个可迭代的数据集，它包装了两个数据加载器，并按顺序轮流从它们中产出批次。
    它会持续进行直到两个加载器都耗尽。

    Args:
        loader1 (DataLoader): 第一个数据加载器。
        loader2 (DataLoader): 第二个数据加载器。
        loader1_name (str): 第一个加载器的名称 (例如 'video')。
        loader2_name (str): 第二个加载器的名称 (例如 'audio')。
    """
    def __init__(self, loader1, loader2, loader1_name='video', loader2_name='audio'):
        super().__init__()
        self.loader1 = loader1
        self.loader2 = loader2
        self.loader1_name = loader1_name
        self.loader2_name = loader2_name
        self.max_len = max(len(loader1), len(loader2))
        
    def __iter__(self):
        # 获取 DDP 信息
        try:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        except (RuntimeError, ValueError):
            # 如果不在分布式环境中，则默认为单进程
            world_size = 1
            rank = 0

        # 创建两个无限循环迭代器
        iter1 = cycle(self.loader1)
        iter2 = cycle(self.loader2)
        
        # 核心修改：只 yield 属于当前 rank 的数据
        # 我们将总的交替流想象成一个大列表，然后对其进行切分
        # 交替流: [v1, a1, v2, a2, v3, a3, ...]
        
        # 每个 for 循环迭代产生 2 个 batch (1 个 video, 1 个 audio)
        # 总共会产生 2 * self.max_len 个 batch
        
        # for 循环负责驱动迭代
        for i in range(self.max_len):
            # 获取下一个 video batch
            v_batch = next(iter1)
            # 获取下一个 audio batch
            a_batch = next(iter2)
            
            # 这是一个交替对，我们根据索引 i 来决定哪个进程处理它
            if i % world_size == rank:
                # 只有当轮次索引 i 属于当前 rank 时，才 yield 数据
                yield v_batch
                yield a_batch

    def __len__(self):
        # 在 DDP 环境下，__len__ 应该返回单个进程处理的 batch 数量
        # 以便 Lightning 正确显示进度条
        
        try:
            world_size = dist.get_world_size()
        except (RuntimeError, ValueError):
            world_size = 1
        
        # 每个进程大致处理 1/world_size 的数据对
        # 每个数据对包含 2 个 batch
        num_pairs_per_process = self.max_len // world_size
        
        # 如果总数不能整除，最后一个 rank 会多处理一些
        # 为简化起见，我们通常可以用 ceil 来计算
        # (self.max_len + world_size - 1) // world_size 是一种高效的 ceil 写法
        num_pairs_per_process = (self.max_len + world_size - 1) // world_size
        
        return 2 * num_pairs_per_process
def get_configs(audio_configs):
    configs = []
    for config in audio_configs:
        data_dir_path = config.get("path", None)
        audio_dir_path = config.get("audio_dir", None)
        split_path = config.get("split_path", None)
        assert data_dir_path is not None, "Path must be set for local audio directory configuration"
        
        custom_metadata_fn = None
        custom_metadata_module_path = config.get("custom_metadata_module", None)
        
        if custom_metadata_module_path:
            spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
            metadata_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metadata_module)
            custom_metadata_fn = metadata_module.get_custom_metadata

        configs.append(
            LocalDatasetConfig(
                id=config["id"],
                path=data_dir_path,
                split_path=split_path,
                custom_metadata_fn=custom_metadata_fn,
                audio_dir=audio_dir_path
            )
        )
    return configs

class DataModule(L.LightningDataModule):
    def __init__(self, dataset_config, batch_size, test_batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4):
        super().__init__()
        dataset_type = dataset_config.get("dataset_type", None)
        repeat_num = dataset_config.get("repeat_num", 1)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_batch_size = test_batch_size
        self.repeat_num = repeat_num
        assert dataset_type is not None, "Dataset type must be specified in dataset config"

        if audio_channels == 1:
            force_channels = "mono"
        elif audio_channels == 2:
            force_channels = "stereo"
        else:
            force_channels = "foa"
        val_dir_configs = dataset_config.get("val_datasets", None)
        test_dir_configs = dataset_config.get("test_datasets", None)
        configs = []
        val_configs = []
        test_configs = []
        if dataset_type == "audio_dir":
            audio_dir_configs = dataset_config.get("datasets", None)
            assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"
            configs = get_configs(audio_dir_configs)
            val_configs = get_configs(val_dir_configs)
            test_configs = get_configs(test_dir_configs)
        elif dataset_type == "latent_dir" or dataset_type == "video_dataset" or dataset_type == "audio_dataset":
            audio_dir_configs = dataset_config.get("datasets", None)
            assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"
            for i, dataset in enumerate((audio_dir_configs, val_dir_configs, test_dir_configs)):
                for config in dataset:
                    data_dir_path = config.get("path", None)
                    audio_dir_path = config.get("audio_dir", None)
                    split_path = config.get("split_path", None)
                    assert data_dir_path is not None, "Path must be set for local audio directory configuration"
                    
                    content = LocalDatasetConfig(
                        id=config["id"],
                        path=data_dir_path,
                        split_path=split_path,
                        audio_dir=audio_dir_path
                    )
                    if i == 0:
                        configs.append(content)
                    elif i == 1:
                        val_configs.append(content)
                    else:
                        test_configs.append(content)
        elif dataset_type in ["multimodal_dir", "alternating_multimodal"]:
            print('##########################')
            print(f'repeat num is: {self.repeat_num}')
            self.audio_configs = []
            self.video_configs = []
            audio_dir_configs = dataset_config.get("audio_datasets", None)
            video_dir_configs = dataset_config.get("video_datasets", None)
            assert audio_dir_configs is not None and video_dir_configs is not None, "Directory configuration must be specified in video_datasets and audio_datasets"
            for i, dataset in enumerate((audio_dir_configs, video_dir_configs, val_dir_configs, test_dir_configs)):
                for config in dataset:
                    data_dir_path = config.get("path", None)
                    audio_dir_path = config.get("audio_dir", None)
                    split_path = config.get("split_path", None)
                    assert data_dir_path is not None, "Path must be set for local audio directory configuration"
                    
                    content = LocalDatasetConfig(
                        id=config["id"],
                        path=data_dir_path,
                        split_path=split_path,
                        audio_dir=audio_dir_path
                    )
                    if i == 0:
                        self.audio_configs.append(content)
                    elif i == 1:
                        self.video_configs.append(content)
                    elif i == 2:
                        val_configs.append(content)
                    else:
                        test_configs.append(content)
        self.dataset_type = dataset_type
        self.configs = configs
        self.val_configs = val_configs
        self.test_configs = test_configs
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.random_crop = dataset_config.get("random_crop", True)
        self.input_type = dataset_config.get("input_type", "video")
        self.fps = dataset_config.get("fps", 4)
        self.force_channels = force_channels


    def setup(self, stage: str):
        if self.dataset_type == 'audio_dir':
            dataset_class = SampleDataset
        elif self.dataset_type == 'latent_dir':
            dataset_class = LatentDataset
        elif self.dataset_type == 'video_dataset':
            dataset_class = VideoDataset
        elif self.dataset_type == 'audio_dataset':
            dataset_class = AudioDataset
        elif self.dataset_type in ["multimodal_dir", "alternating_multimodal"]:
            dataset_class = VideoDataset

        def create_dataset(configs, random_crop):
            return dataset_class(
                configs,
                sample_rate=self.sample_rate,
                sample_size=self.sample_size,
                random_crop=random_crop,
                input_type=self.input_type,
                fps=self.input_type,
                force_channels=self.force_channels
            )

        if stage == 'fit':
            if self.dataset_type not in ["multimodal_dir", "alternating_multimodal"]:
                self.train_set = create_dataset(self.configs, random_crop=self.random_crop)
            elif self.dataset_type == "multimodal_dir":
                self.video_set = VideoDataset(
                    self.video_configs,
                    sample_rate=self.sample_rate,
                    sample_size=self.sample_size,
                    random_crop=self.random_crop,
                    input_type=self.input_type,
                    fps=self.input_type,
                    force_channels=self.force_channels
                )
                self.audio_set = AudioDataset(
                    self.audio_configs,
                    sample_rate=self.sample_rate,
                    sample_size=self.sample_size,
                    random_crop=self.random_crop,
                    input_type=self.input_type,
                    fps=self.input_type,
                    force_channels=self.force_channels
                )
                self.train_set = MultiModalDataset([self.video_set]*self.repeat_num, [self.audio_set])
            elif self.dataset_type == "alternating_multimodal":
                self.video_set = VideoDataset(
                    self.video_configs,
                    sample_rate=self.sample_rate,
                    sample_size=self.sample_size,
                    random_crop=self.random_crop,
                    input_type=self.input_type,
                    fps=self.input_type,
                    force_channels=self.force_channels
                )
                self.audio_set = AudioDataset(
                    self.audio_configs,
                    sample_rate=self.sample_rate,
                    sample_size=self.sample_size,
                    random_crop=self.random_crop,
                    input_type=self.input_type,
                    fps=self.input_type,
                    force_channels=self.force_channels
                )
            self.val_set = create_dataset(self.val_configs, random_crop=False)
        elif stage == 'validate':
            self.val_set = create_dataset(self.val_configs, random_crop=False)
        elif stage == 'predict':
            self.test_set = create_dataset(self.test_configs, random_crop=False)



    def train_dataloader(self):
        if self.dataset_type == "alternating_multimodal":
            # 视频 DataLoader
            video_loader = DataLoader(
                self.video_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=collation_fn
            )

            # 音频 DataLoader
            audio_loader = DataLoader(
                self.audio_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=collation_fn
            )
            alternating_loader = AlternatingLoader(
                video_loader, 
                audio_loader, 
                loader1_name='video', 
                loader2_name='audio'
            )
            return DataLoader(alternating_loader, batch_size=None, num_workers=0)
        else:
            # 如果不是 alternating_multimodal，保持现有逻辑（仅用于兼容性）
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
                drop_last=True,
                collate_fn=collation_fn
            )
        

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False,
                                num_workers=self.num_workers, persistent_workers=False, pin_memory=False, drop_last=False, collate_fn=collation_fn)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False,
                                num_workers=self.num_workers, persistent_workers=False, pin_memory=False, drop_last=False, collate_fn=collation_fn)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     ...