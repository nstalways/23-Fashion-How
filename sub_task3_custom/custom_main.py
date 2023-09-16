# torch
import torch

# built-in library
import os
import argparse

# external library
import wandb

# custom modules
from gaia import *
from exp_utils import *

cores = os.cpu_count()
torch.set_num_threads(cores)


# TODO: wandb 코드 완료하기
if __name__ == "__main__":
    # # wandb 세팅
    # wandb.init(project="Sub Task3: Continual Learning", reinit=True)

    print('\n')
    print('-'*60)
    print('\t\tAI Fashion Coordinator')
    print('-'*60)
    print('\n')

    # 실험 설정 값들을 불러옵니다.
    cfg_path = parse_args().cfg_path
    cfgs = load_cfg(cfg_path)
    args = MakeArgs(cfgs)

    # 유효성 검사
    mode = args.mode
    if mode not in ['train', 'test', 'pred']:
        raise ValueError(f'Unknown mode {mode}')

    # 유효성 검사에서 문제가 없다면, 실험 재현이 가능하게끔 seed 세팅
    set_seed(args.seed)

    # # wandb에 configs, 실험 이름 등록
    # wandb.config.update(args)
    # wandb.run_name = cfg_path.split('/')[-1].split('.')[0]

    # 실험이 진행되기 전 설정 값들을 확인하기 위해 configs 출력
    print('<Parsed Arguments>')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('')

    
    # 객체 선언
    gaia = gAIa(args, get_udevice())

    # TODO: 실험 코드 작성하기
    if mode == 'train':
        gaia.train()
    elif mode == 'test':
        gaia.test()
    elif mode == 'pred':
        gaia.pred()
   
