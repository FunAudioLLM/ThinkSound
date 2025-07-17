import lightning as L
from .dataset import LatentDataset, SampleDataset, VideoDataset, AudioDataset, MultiModalDataset, LocalDatasetConfig, collation_fn
import importlib
from torch.utils.data import DataLoader


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
    def __init__(self, dataset_config, batch_size, test_batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4,repeat_num=5,latent_length=194):
        super().__init__()
        dataset_type = dataset_config.get("dataset_type", None)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_batch_size = test_batch_size
        self.repeat_num = repeat_num
        self.latent_length = latent_length
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
        elif dataset_type == "latent_dir" or dataset_type == "video_dataset":
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
                        audio_dir=audio_dir_path,
                        extra_cot=config.get("extra_cot", None)
                    )
                    if i == 0:
                        configs.append(content)
                    elif i == 1:
                        val_configs.append(content)
                    else:
                        test_configs.append(content)
        elif dataset_type == "multimodal_dir":
            self.audio_configs = []
            self.video_configs = []
            audio_dir_configs = dataset_config.get("audio_datasets", None)
            video_dir_configs = dataset_config.get("video_datasets", None)
            #assert audio_dir_configs is not None and video_dir_configs is not None, "Directory configuration must be specified in video_datasets and audio_datasets"
            for i, dataset in enumerate((audio_dir_configs, video_dir_configs, val_dir_configs, test_dir_configs)):
                if dataset is None:
                    continue
                for config in dataset:
                    data_dir_path = config.get("path", None)
                    audio_dir_path = config.get("audio_dir", None)
                    split_path = config.get("split_path", None)
                    assert data_dir_path is not None, "Path must be set for local audio directory configuration"
                    print(f'extra cot: {config.get("extra_cot", None)}')
                    content = LocalDatasetConfig(
                        id=config["id"],
                        path=data_dir_path,
                        split_path=split_path,
                        audio_dir=audio_dir_path,
                        extra_cot=config.get("extra_cot", None)
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
        elif self.dataset_type == 'multimodal_dir':
            dataset_class = VideoDataset

        def create_dataset(configs, random_crop):
            return dataset_class(
                configs,
                sample_rate=self.sample_rate,
                sample_size=self.sample_size,
                random_crop=random_crop,
                input_type=self.input_type,
                fps=self.input_type,
                force_channels=self.force_channels,
                latent_length=self.latent_length
            )

        if stage == 'fit':
            if self.dataset_type != 'multimodal_dir':
                self.train_set = create_dataset(self.configs, random_crop=self.random_crop)
            else:
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
            self.val_set = create_dataset(self.val_configs, random_crop=False)
        elif stage == 'validate':
            self.val_set = create_dataset(self.val_configs, random_crop=False)
        elif stage == 'predict':
            self.test_set = create_dataset(self.test_configs, random_crop=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True,
                                num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collation_fn)

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