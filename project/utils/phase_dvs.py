from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tonic
import numpy as np
import cv2
from dataclasses import dataclass


def to_bit_encoding(events: np.ndarray, sensor_size):
    event_frames = tonic.transforms.functional.to_frame_numpy(events, sensor_size, n_time_bins=8)
    event_frames = (event_frames > 0).astype(np.uint8)  # binary event frames
    return np.packbits(event_frames).astype(np.float32) / 255.


def to_weighted_frames(events: np.ndarray, sensor_size, timesteps: int, blur_type=None):
    event_frames = tonic.transforms.functional.to_frame_numpy(events, sensor_size, n_time_bins=timesteps)
    event_frames = (event_frames > 0).astype(np.float32)  # binary event frames

    weight = 1. / timesteps
    event_frames = event_frames * weight
    frames = event_frames.sum(0)

    if blur_type is not None:
        frames = to_blur(frames, blur_type)


@dataclass(frozen=True)
class ToBitEncoding:
    sensor_size: Tuple[int, int, int]

    def __call__(self, events):
        return to_bit_encoding(events.copy(), self.sensor_size)


@dataclass(frozen=True)
class ToWeightedFrames:
    sensor_size: Tuple[int, int, int]
    timesteps: int

    def __call__(self, events):
        return to_weighted_frames(events.copy(), self.sensor_size, self.timesteps)


def to_blur(event_frames: np.ndarray, blur_type: str = 'averaging'):
    if blur_type == 'averaging':
        return cv2.blur(event_frames, (5, 5))

    elif blur_type == 'median':
        return cv2.medianBlur(event_frames, 5)

    elif blur_type == 'gaussian':
        return cv2.GaussianBlur(event_frames, (5, 5), 0)

    elif blur_type == 'bilateral':
        return cv2.bilateralFilter(event_frames, 9, 75, 75)

    else:
        NotImplementedError('Must implement other blur strategies before using them.')
