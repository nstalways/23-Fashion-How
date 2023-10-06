# torch
import torch

# built-in library
import os
import argparse

# external library
import wandb

# custom modules
from exp_utils import set_seed, get_udevice
from gaia import *


def str2bool(v: str) -> bool:
    """문자열을 boolean type으로 변환합니다.

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
    """실험을 위해 필요한 설정 값들을 입력받습니다.

    Returns:
        argparse.Namespace: 추가한 argument 정보가 저장되어 있습니다.
    """
    parser = argparse.ArgumentParser(description='AI Fashion Coordinator')
    
    ### Utility ###
    parser.add_argument('--exp_group_name', type=str,
                        default='base',
                        help='group name of whole experiments, using WandB')
    parser.add_argument('--seed', type=int,
                        default=2023,
                        help='setting a global seed for reproducibility')
    parser.add_argument('--mode', type=str,
                        default='test',
                        help='training or eval or test mode')
    parser.add_argument('--save_freq', type=int,
                        default=2,
                        help='evaluate and save results per # epochs')

    ### Data ###
    # Raw data
    parser.add_argument('--in_file_trn_dialog', type=str,
                        default='/home/suyeongp7/data/train/task1.ddata.wst.txt',
                        help='training dialog DB')
    # TODO: 성능 validation을 어떤 방식으로 구현할 지 고민해보기
    parser.add_argument('--in_file_tst_dialog', type=str, 
                        default='/home/suyeongp7/data/validation/cl_eval_task1.wst.dev', 
                        help='test dialog DB')
    parser.add_argument('--in_file_fashion', type=str,
                        default='/home/suyeongp7/data/item_metadata/mdata.wst.txt.2023.08.23',
                        help='fashion item metadata')
    parser.add_argument('--in_file_img_feats', type=str,
                        default=None,
                        help='fashion item image features')
    parser.add_argument('--subWordEmb_path', type=str,
                        default='/home/suyeongp7/data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat',
                        help='path of subword embedding')
    
    # Augmentation
    parser.add_argument('--corr_thres', type=float,
                        default=0.7,
                        help='correlation threshold')
    parser.add_argument('--permutation_iteration', type=int,
                        default=3,
                        help='# of permutation iteration')
    parser.add_argument('--num_augmentation', type=int,
                        default=3,
                        help='# of data augmentation')

    ### Model ###
    # Model path
    parser.add_argument('--model_path', type=str,
                        default='./gAIa_CL_model',
                        help='path to save/read model')
    parser.add_argument('--model_file', type=str,
                        default=None,
                        help='model file name')
    
    # MemN2N
    parser.add_argument('--hops', type=int,
                        default=3,
                        help='number of hops in the MemN2N')
    parser.add_argument('--mem_size', type=int,
                        default=16,
                        help='memory size for the MemN2N')
    parser.add_argument('--key_size', type=int,
                        default=300,
                        help='key size for the MemN2N')

    # Ranking Model
    parser.add_argument('--eval_node', type=str,
                        default='[6000,6000,200][2000]',
                        help='nodes of ranking model')
    parser.add_argument('--use_batch_norm', type=str2bool,
                        default=False,
                        help='use batch normalization')
    parser.add_argument('--use_dropout', type=str2bool,
                        default=False,
                        help='use dropout')
    parser.add_argument('--zero_prob', type=float,
                        default=0.0,
                        help='dropout probability')
    parser.add_argument('--use_multimodal', type=str2bool,
                        default=False,
                        help='use multimodal input')

    ### Optimizer ###
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001,
                        help='learning rate')

    ### Training ###
    parser.add_argument('--batch_size', type=int,
                        default=100,
                        help='batch size for training')                    
    parser.add_argument('--epochs', type=int,
                        default=10,
                        help='epochs to training')
    parser.add_argument('--max_grad_norm', type=float,
                        default=40.0,
                        help='clip gradients to this norm')
    parser.add_argument('--use_cl', type=str2bool,
                        default=True,
                        help='enable continual learning')
    
    ### Validation ###
    parser.add_argument('--evaluation_iteration', type=int,
                        default=10,
                        help='# of test iteration')
    
    ### Test ###
    parser.add_argument('--csv_path', type=str,
                        default='./eval_result',
                        help='path to save a test score')
    
    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    mode = args.mode
    project_name = 'Sub-Task3 Continual Learning'
   
    print('\n')
    print('-' * 60)
    print('\t\tAI Fashion Coordinator')
    print('-' * 60)
    print('\n')
    
    if mode not in ['train', 'eval', 'test', 'pred']:
        raise ValueError(f'Unknown mode {mode}')

    print('<Parsed arguments>')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('')

    set_seed(args.seed)

    gaia = gAIa(args, get_udevice())

    if mode == 'train':
        run_name = args.in_file_trn_dialog.split('/')[-1].split('.')[0] # task#
        run = wandb.init(project=project_name,
                         group=args.exp_group_name,
                         job_type=mode,
                         name=run_name,
                         config=args,
                         reinit=True)
        
        gaia.train()

    # TODO: 평가 코드 마무리하기
    elif mode in ['eval', 'test']:
        run_name = args.in_file_tst_dialog.split('/')[-1].split('.')[0] # cl_eval_task#
        task_num = int(run_name[-1])
        if task_num == 6:
            run = wandb.init(project=project_name,
                             group=args.exp_group_name,
                             job_type=mode,
                             name=f'{mode}_result',
                             config=args,
                             reinit=True)
            
        gaia.test(args.csv_path, args.exp_group_name, task_num)

    elif mode == 'pred':
        gaia.pred()

    
