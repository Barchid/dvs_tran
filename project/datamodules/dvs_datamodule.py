from typing import Optional
from cv2 import transform
import pytorch_lightning as pl
from torch import save
from torch.functional import split
from torch.utils import data
from torch.utils.data import random_split, DataLoader
import tonic
import torchvision
from torchvision import transforms
import os
import numpy as np

from project.datamodules.cifar10dvs import CIFAR10DVS


class DVSDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, dataset: str, data_dir: str = "data/", event_representation: str = "HOTS", **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.dataset = dataset  # name of the dataset

        self.event_representation = event_representation

        # create the directory if not exist
        os.makedirs(data_dir, exist_ok=True)

        # transform
        self.sensor_size = self._get_sensor_size()
        self.train_transform, self.val_transform = self._get_transforms(event_representation)

    def _get_sensor_size(self):
        if self.dataset == "n-mnist":
            return tonic.datasets.NMNIST.sensor_size
        elif self.dataset == "cifar10-dvs":
            return CIFAR10DVS.sensor_size
        elif self.dataset == "dvsgesture":
            return tonic.datasets.DVSGesture.sensor_size
        elif self.dataset == "n-caltech101":
            return (224, 224, 2)
        elif self.dataset == "asl-dvs":
            return tonic.datasets.ASLDVS.sensor_size

    def prepare_data(self) -> None:
        # downloads the dataset if it does not exist
        # NOTE: since we use the library named "Tonic", all the download process is handled, we just have to make an instanciation
        if self.dataset == "n-mnist":
            tonic.datasets.NMNIST(save_to=self.data_dir)
        elif self.dataset == "cifar10-dvs":
            CIFAR10DVS(save_to=self.data_dir)
        elif self.dataset == "dvsgesture":
            tonic.datasets.DVSGesture(save_to=self.data_dir)
        elif self.dataset == "n-caltech101":
            tonic.datasets.NCALTECH101(save_to=self.data_dir)
        elif self.dataset == "asl-dvs":
            tonic.datasets.ASLDVS(save_to=self.data_dir)

    def _get_transforms(self, event_representation: str):
        denoise = tonic.transforms.Denoise()
        if event_representation == "HOTS":
            representation = tonic.transforms.ToTimesurface(sensor_size=self.sensor_size)
        elif event_representation == "HATS":
            representation = tonic.transforms.ToAveragedTimesurface(self.sensor_size)
        elif event_representation == "frames_time":
            representation = tonic.transforms.Compose([
                tonic.transforms.ToFrame(self.sensor_size, n_time_bins=10),
                transforms.Lambda(lambda x: (x > 0).astype(np.float32))
            ])
        elif event_representation == "frames_event":
            representation = tonic.transforms.Compose([
                tonic.transforms.ToFrame(self.sensor_size, n_event_bins=10),
                transforms.Lambda(lambda x: (x > 0).astype(np.float32))
            ])

        elif event_representation == "VoxelGrid":
            representation = tonic.transforms.ToVoxelGrid(self.sensor_size, n_time_bins=10)

        val_transform = tonic.transforms.Compose([
            denoise,
            representation
        ])

        train_transform = tonic.transforms.Compose([
            tonic.transforms.RandomFlipLR(self.sensor_size),
            denoise,
            representation,
        ])

        return train_transform, val_transform

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset == "n-mnist":
            self.train_set = tonic.datasets.NMNIST(
                save_to=self.data_dir, transform=self.train_transform, target_transform=None, train=True)
            self.val_set = tonic.datasets.NMNIST(
                save_to=self.data_dir, transform=self.val_transform, target_transform=None, train=False)
        elif self.dataset == "cifar10-dvs":
            dataset_train = CIFAR10DVS(save_to=self.data_dir, transform=self.train_transform, target_transform=None)
            dataset_val = CIFAR10DVS(save_to=self.data_dir, transform=self.val_transform, target_transform=None)
            self.train_set, _ = random_split(dataset_train, lengths=[8330, 10000 - 8330])
            _, self.val_set = random_split(dataset_val, lengths=[8330, 10000 - 8330])

        elif self.dataset == "dvsgesture":
            self.train_set = tonic.datasets.DVSGesture(
                save_to=self.data_dir, transform=self.train_transform, target_transform=None, train=True)
            self.val_set = tonic.datasets.DVSGesture(
                save_to=self.data_dir, transform=self.val_transform, target_transform=None, train=False)

        elif self.dataset == "n-caltech101":
            tonic.datasets.NCALTECH101(save_to=self.data_dir)

        elif self.dataset == "asl-dvs":
            dataset = tonic.datasets.ASLDVS(save_to=self.data_dir, transform=self.train_transform)
            full_length = len(dataset)

            self.train_set, _ = random_split(dataset, [0.8 * full_length, full_length - (0.8 * full_length)])
            dataset = tonic.datasets.ASLDVS(save_to=self.data_dir, transform=self.val_transform)
            _, self.val_set = random_split(dataset, [0.8 * full_length, full_length - (0.8 * full_length)])

        self.train_set = tonic.datasets.NMNIST(save_to=self.data_dir, train=True, transform=self.transform)
        self.val_set = tonic.datasets.NMNIST(save_to=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
