# torch
import torch

# built-in library
import random
import yaml
import argparse

# external library
import numpy as np


def get_udevice():
    """사용 가능한 device를 가져옵니다. (CPU or GPU)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpu = torch.cuda.device_count()
    else:
        device = torch.device('cpu')
    
    print(f'Using device: {device}')

    if torch.cuda.is_available():
        print(f'# of GPU: {num_gpu}')

    return device


def set_seed(seed: int=2023) -> None:
    """
    실험 재현을 위해 시드를 설정합니다.

    Args:
        seed (int, optional): 시드로 사용할 값. Defaults to 2023.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# Test Code
if __name__ == "__main__":
    pass