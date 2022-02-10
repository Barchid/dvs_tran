import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tonic import transforms, functional

def dist_numpy(events, sensor_size, alpha: float = 0.4)