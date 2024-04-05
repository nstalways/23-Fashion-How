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

# custom modules
from gaia import *
from exp_utils import set_seed

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


# input options
parser = argparse.ArgumentParser(description='AI Fashion Coordinator.')

parser.add_argument('--exp_name', type=str,
                    default='exp_cl',
                    help='name of experiment')
parser.add_argument('--seed', type=int,
                    default=2023,
                    help='setting a global seed for reproducibility')
parser.add_argument('--mode', type=str, 
                    default='eval',
                    help='training or eval or test mode')
parser.add_argument('--task_ids', type=str, 
                    default='/1/1',
                    help='task id of training data at last of currently evaluating model & task id of evaluation data')
parser.add_argument('--in_file_trn_dialog', type=str, 
                    default='../data/task1.ddata.wst.txt', 
                    help='training dialog DB')
parser.add_argument('--in_file_tst_dialog', type=str, 
                    default='../data/cl_eval_task1.wst.dev', 
                    help='test dialog DB')
parser.add_argument('--in_file_fashion', type=str, 
                    default='../data/mdata.wst.txt.2023.08.23', 
                    help='fashion item metadata')
parser.add_argument('--in_file_img_feats', type=str, # deprecated
                    default='../data/img_feats', 
                    help='fashion item image features')
parser.add_argument('--model_path', type=str, 
                    default='./gAIa_CL_model', 
                    help='path to save/read model')
parser.add_argument('--model_file', type=str, 
                    default=None, 
                    help='model file name')
parser.add_argument('--eval_node', type=str, 
                    default='[6000,6000,200][2000]', 
                    help='nodes of evaluation network')
parser.add_argument('--subWordEmb_path', type=str, 
                    default='./sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat', 
                    help='path of subword embedding')
parser.add_argument('--learning_rate', type=float,
                    default=0.0001, 
                    help='learning rate')
parser.add_argument('--max_grad_norm', type=float,
                    default=40.0, 
                    help='clip gradients to this norm')
parser.add_argument('--zero_prob', type=float,
                    default=0.0, 
                    help='dropout prob.')
parser.add_argument('--corr_thres', type=float,
                    default=0.7, 
                    help='correlation threshold')
parser.add_argument('--batch_size', type=int,
                    default=100,   
                    help='batch size for training')
parser.add_argument('--epochs', type=int,
                    default=10,   
                    help='epochs to training')
parser.add_argument('--save_freq', type=int,
                    default=2,   
                    help='evaluate and save results per # epochs')
parser.add_argument('--hops', type=int,
                    default=3,   
                    help='number of hops in the MemN2N')
parser.add_argument('--mem_size', type=int,
                    default=16,   
                    help='memory size for the MemN2N')
parser.add_argument('--key_size', type=int,
                    default=300,   
                    help='memory size for the MemN2N')
parser.add_argument('--permutation_iteration', type=int,
                    default=3,   
                    help='# of permutation iteration')
parser.add_argument('--evaluation_iteration', type=int,
                    default=10,   
                    help='# of test iteration')
parser.add_argument('--num_augmentation', type=int,
                    default=3,   
                    help='# of data augmentation')
parser.add_argument('--use_batch_norm', type=str2bool, 
                    default=False, 
                    help='use batch normalization')
parser.add_argument('--use_dropout', type=str2bool, 
                    default=False, 
                    help='use dropout')
parser.add_argument('--use_multimodal', type=str2bool, # TODO: 올바르게 작동하는지 확인하기
                    default=False, 
                    help='use multimodal input')
parser.add_argument('--use_cl', type=str2bool,
                    default=True,
                    help='enable continual learning')

args, _ = parser.parse_known_args()


if __name__ == '__main__':
    
    print('\n')
    print('-'*60)
    print('\t\tAI Fashion Coordinator')
    print('-'*60)
    print('\n')

    mode = args.mode    
    if mode not in ['train', 'test', 'eval', 'pred'] :
        raise ValueError('Unknown mode {}'.format(mode))

    set_seed(args.seed)

    print('<Parsed arguments>')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('')

    gaia = gAIa(args, get_udevice())

    if args.model_file:
        models = args.model_file.split(',')

    if mode == 'train':
        gaia.train() # training

    elif mode in ['eval', 'test']:
        if len(models) > 1:
            print('Using ensemble...')
            gaia.ensemble(models)
            
        else:
            gaia.test() # test
        
    elif mode == 'pred':
        gaia.pred() # pred

