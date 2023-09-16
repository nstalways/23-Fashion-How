# torch
import torch

# built-in library
import random
import yaml
import argparse

# external library
import numpy as np


def get_udevice():
    """사용가능한 device를 가져옵니다. (CPU or GPU)
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


def str2bool(v: str) -> bool:
    """문자열을 bool로 변환할 때 사용하는 함수입니다.

    Args:
        v (str): 문자열입니다.
    """
    if isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def parse_args() -> argparse.Namespace:
    """실험을 위해 필요한 설정 값들을 불러오기 위해서,
    설정 값들이 작성되어있는 파일의 경로를 입력할 때 사용합니다.

    Returns:
        argparse.Namespace: 추가한 argument 정보가 저장되어 있습니다.
    """
    parser = argparse.ArgumentParser(description='AI Fashion Coordinator')

    parser.add_argument('--cfg_path', type=str, default='./configs/base.yaml',
                        help='실험 및 평가에 사용할 설정 값들을 기록해둔 파일의 경로를 적어주세요.')

    args = parser.parse_args()

    return args
    

def load_cfg(cfg_path: str) -> dict:
    """
    실험에 사용할 설정값들을 불러올 때 사용합니다.
    ArgParser가 아닌 yaml 파일 기반으로 실험하기 위해 추가한 함수입니다.

    Args:
        cfg_name: config 파일의 경로

    Return:
        cfg_dict: 실험에 사용할 설정값들이 담긴 dictionary
    """
    with open(cfg_path, encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    
    return cfg_dict


def set_seed(seed: int=2023) -> None:
    """
    실험의 재현 가능성을 위해 시드를 설정할 때 사용하는 함수입니다.

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


class MakeArgs:
    def __init__(self, cfgs: dict):
        """
        yaml 파일에서 불러온 설정 값들을 멤버 변수로 사용할 수 있도록 구조화하는 역할의 클래스입니다.
        Baseline으로 제공된 코드 중 모델 코드의 내부를 수정하지 않고 그대로 사용하기 위해 필요합니다.

        Args:
            cfgs (dict): 실험 설정값
        """
        for _, vals in cfgs.items():
            for cfg_name, cfg_val in vals.items():
                if isinstance(cfg_val, str):
                    var = f"self.{cfg_name} = '{cfg_val}'"
                else:
                    var = f"self.{cfg_name} = {cfg_val}"
                
                exec(var)


# Test Code
if __name__ == "__main__":
    cfgs = load_cfg(parse_args().cfg_path)
    args = MakeArgs(cfgs)
    print(args.seed)