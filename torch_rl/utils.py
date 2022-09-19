import numpy as np
import random
import torch


def set_seed(seed=123, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def linear_schedule(start_e: float, end_e: float, duration: int, total_e: int):
    slope = (end_e - start_e) / duration
    return max(slope * total_e + start_e, end_e)


class LinearSchedule:

    def __init__(self, start_e: float, end_e: float, duration: int):
        self.start_e = start_e
        self.end_e = end_e
        self.duration = duration
    
    def __call__(self, total_e):
        slope = (self.end_e - self.start_e) / self.duration
        return max(slope * total_e + self.start_e, self.end_e)