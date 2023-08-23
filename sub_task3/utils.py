# torch
import torch

# built-in library
import os
import yaml
import random

# external library
import numpy as np


def load_cfg(cfg_name: str) -> dict:
    """
    function: 실험에 사용할 설정값들을 불러올 때 사용합니다.
    ArgParser가 아닌 yaml 파일 기반으로 실험하기 위해 추가한 함수입니다.

    Args:
        cfg_name: config 파일의 이름

    Return:
        cfg_dict: 실험에 사용할 설정값들이 담긴 dictionary
    """
    ROOT = "./configs"
    yaml_path = os.path.join(ROOT, cfg_name)

    with open(yaml_path, encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    
    return cfg_dict


def set_seed(seed: int=2023) -> None:
    """
    function: 실험의 재현 가능성을 위해 시드를 설정할 때 사용하는 함수입니다.

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