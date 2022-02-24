import os
from typing import Optional
import numpy as np

from tonic.io import read_mnist_file
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive
from tonic import functional as TF
import torch.nn.functional as F
import torch
from torch.utils.data import random_split
import cv2
import random
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms


def hot_pixels(frames: np.array, severity: int):
    # frames dim : (T,C,H,W)
    # severity between 1 and 5
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    # height and with of dvs frames
    height, width = frames.shape[-2], frames.shape[-1]

    # create the mask of hot pixels (H,W) (1 where there will be a hot pixel and 0 where the pixels won't be broken)
    # from https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones
    N = frames.shape[-2] * frames.shape[-1]  # total size of mask
    K = int(c * N)  # certain proportion of broken pixels
    hot_pixels_mask = np.array([1.] * K + [0.] * (N-K))
    np.random.shuffle(hot_pixels_mask)
    hot_pixels_mask = np.reshape(hot_pixels_mask, (height, width))

    # apply hot pixels in frames
    result = []
    for i in range(frames.shape[0]):
        frame = frames[i]  # C,H,W
        frame[0][hot_pixels_mask == 1] = 1.  # for positive channel
        frame[1][hot_pixels_mask == 1] = 1.  # for negative channel
        result.append(frame)

    return np.array(result, dtype=np.float32)


def spatial_jitter(events, sensor_size, severity):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]  # c is the sigma here
    variance = c ** 2
    return TF.spatial_jitter_numpy(events, sensor_size, variance_x=variance, variance_y=variance, clip_outliers=True)


def temporal_jitter(events, severity):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]  # c is the sigma here
    return TF.time_jitter_numpy(events, c, clip_negative=True, sort_timestamps=True)


def background_activity(frames: np.ndarray, severity: int):
    # c is the average rate of background activity noise here
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    noise_mask = np.random.poisson(lam=c, size=frames.shape)

    frames = np.clip(frames + noise_mask, 0, 1).astype(np.float32)
    return frames
