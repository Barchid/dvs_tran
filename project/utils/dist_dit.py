import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tonic import transforms, functional

def to_dist_numpy(events, sensor_size, alpha: float = 0.4):
    pass

def to_dit_numpy(events, sensor_size, alpha: float = 0.4, patch_size: int = 2):
    """Implementation of Discounted Timestap Image (DiT), a robust event representation which is designed for robustness against
    changes in camera trajectory and brightness. See: https://arxiv.org/pdf/2112.01041.pdf

    Args:
        events ([type]): [description]
        sensor_size ([type]): [description]
    """
    assert "x" and "y" and "t" and "p" in events.dtype.names
    assert patch_size > 1
    
    T_new = None
    
    # N.B. we use the same notation as in the paper.
    # sensor dimension
    W, H, p = sensor_size
    
    # get the timestamp image (= 2 channel image with last)
    S_0 = 
    
    
    
    
    pass