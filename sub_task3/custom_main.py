'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2023, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2023.02.15.
'''
# torch
import torch

# built-in library
import os
import argparse

# external library
import wandb

# custom-modules
from gaia import *
from utils import load_cfg, set_seed


cores = os.cpu_count()
torch.set_num_threads(cores)


def get_udevice():
    """
    function: get usable devices(CPU and GPU)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpu = torch.cuda.device_count()
    else:    
        device = torch.device('cpu')
    print('Using device: {}'.format(device))
    if torch.cuda.is_available():
        print('# of GPU: {}'.format(num_gpu))
    return device


def str2bool(v):
    """
    function: convert into bool type(True or False)
    """
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def parse_args():
    parser = argparse.ArgumentParser(description='AI Fashion Coordinator.')

    parser.add_argument('--cfg_name', type=str, default='base.yaml',
                        help='실험 및 평가에 사용할 설정 값들을 기록해둔 파일의 이름을 적어주세요.')

    args = parser.parse_args()

    return args


class Cfg:
    def __init__(self, cfgs: dict):
        """yaml 파일에서 불러온 설정 값들을 멤버 변수로 사용할 수 있도록
        구조화하는 역할의 클래스입니다.

        Args:
            cfgs (dict): _description_

        Raises:
            ValueError: _description_
        """
        for _, vals in cfgs.items():
            for cfg_name, cfg_val in vals.items():
                if isinstance(cfg_val, str):
                    var = f"self.{cfg_name} = '{cfg_val}'"
                else:
                    var = f"self.{cfg_name} = {cfg_val}"
                
                exec(var)


if __name__ == '__main__':
    # wandb 세팅
    wandb.init(project="Sub_Task3 Continual_Learning", reinit=True)
    
    print('\n')
    print('-'*60)
    print('\t\tAI Fashion Coordinator')
    print('-'*60)
    print('\n')

    # load configs
    tmp = parse_args()
    cfgs = load_cfg(tmp.cfg_name)

    args = Cfg(cfgs)

    # 유효성 검사
    mode = args.mode
    if mode not in ['train', 'test', 'pred'] :
        raise ValueError('Unknown mode {}'.format(mode))
    
    # 문제 없으면 
    set_seed(args.seed) # seed 세팅

    # wandb에 configs, 실험 이름 등록
    wandb.config.update(args)
    wandb.run.name = tmp.cfg_name.split('.')[0]

    # configs 출력
    print('<Parsed arguments>')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('')
    
    # 객체 선언
    gaia = gAIa(args, get_udevice())
    
    # 실험 시작
    if mode == 'train':
        gaia.custom_train() # training
    elif mode == 'test':
        gaia.test() # test
    elif mode == 'pred':
        gaia.pred() # pred
