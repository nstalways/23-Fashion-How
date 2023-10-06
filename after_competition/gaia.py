# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# built-in library
import os
import timeit
import re
from tqdm import tqdm
import shutil
import time
import random

# external library
import numpy as np
import pandas as pd
import wandb
from scipy import stats
from file_io import *

# custom modules
from models.model import Model
from si import surrogate_loss, update_omega
from exp_utils import set_seed

# of items in fashion coordination      
NUM_ITEM_IN_COORDI = 4
# of metadata features    
NUM_META_FEAT = 4
# of fashion coordination candidates        
NUM_RANKING = 3
# image feature size 
IMG_FEAT_SIZE = 4096
# SI parameter (얘도 성능에 영향이 있을 것 같은데..)
si_c = 0.1
epsilon = 0.001


# custom modules
class gAIa(object):
    """
    Class for AI fashion coordinator
    : 모델 학습에 필요한 변수들과 함수들을 정의한 클래스입니다.
    """
    def __init__(self, args, device, name='gAIa'):
        """필요한 변수들을 선언하고, 초기화합니다.

        Args:
            args (_type_): 모델 학습 및 평가에 필요한 인자들입니다.
            device (_type_): 학습 및 평가에 사용할 컴퓨팅 자원입니다. (CPU or GPU)
            name (str, optional): 모델 이름입니다. Defaults to 'gAIa'.
        """
        self._device = device
        self._batch_size = args.batch_size
        self._model_path = args.model_path
        self._model_file = args.model_file
        self._epochs = args.epochs
        self._max_grad_norm = args.max_grad_norm
        self._save_freq = args.save_freq
        self._num_eval = args.evaluation_iteration
        self._in_file_trn_dialog = args.in_file_trn_dialog
        self._use_cl = args.use_cl
        self._mode = args.mode
        use_dropout = args.use_dropout

        # Subword Embedding 클래스 선언
        self._swer = SubWordEmbReaderUtil(args.subWordEmb_path)
        self._emb_size = self._swer.get_emb_size()
        self._meta_size = NUM_META_FEAT
        self._coordi_size = NUM_ITEM_IN_COORDI
        self._num_rnk = NUM_RANKING
        feats_size = IMG_FEAT_SIZE

        ### fashion item metadata 전처리 ###
        metadata = make_metadata(self._swer, args.in_file_fashion, self._coordi_size,
                                 self._meta_size, args.use_multimodal, args.in_file_img_feats, feats_size)
        
        self._metadata, self._idx2item, self._item2idx, \
            self._item_size, self._meta_similarities, self._feats = [*metadata]
         
        ### 학습 및 평가에 사용할 DB 준비 ###
        if args.mode == 'train':
            self._dlg, self._crd, self._rnk = make_io_data(self._swer, self._item2idx, self._idx2item,
                                                           self._metadata, self._meta_similarities, 'prepare',
                                                           self._in_file_trn_dialog, args.mem_size, self._coordi_size,
                                                           self._num_rnk, args.permutation_iteration, args.num_augmentation,
                                                           args.corr_thres, self._feats)
            
            self._num_examples = len(self._dlg)
            
            dataset = TensorDataset(torch.tensor(self._dlg),
                                    torch.tensor(self._crd),
                                    torch.tensor(self._rnk, dtype=torch.long))
            self._dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        elif args.mode in ['eval', 'test', 'pred']:
            self._tst_dlg, self._tst_crd, _ = make_io_data(self._swer, self._item2idx, self._idx2item,
                                                            self._metadata, self._meta_similarities,
                                                            'eval', args.in_file_tst_dialog,
                                                            args.mem_size, self._coordi_size, self._num_rnk,
                                                            args.permutation_iteration, args.num_augmentation,
                                                            args.corr_thres, self._feats)
            
            self._num_examples = len(self._tst_dlg)

        ### 모델 생성 ###
        self._model = Model(self._item_size, self._emb_size,
                            args.key_size, args.mem_size, self._meta_size,
                            args.hops, self._coordi_size, args.eval_node,
                            self._num_rnk, args.use_batch_norm, use_dropout, 
                            args.zero_prob, args.use_multimodal, feats_size)
        
        # 모델을 구성하는 파라미터 출력
        print('\n<model parameters>')
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                print(name)
                n = name.replace('.', '__')
                self._model.register_buffer('{}_SI_prev_task'.format(n), 
                                            param.detach().clone())
                self._model.register_buffer('{}_SI_omega'.format(n), 
                                            torch.zeros(param.shape))

        ### Optimizer, Loss function 정의 ###
        if args.mode == 'train':
            # optimizer
            # TODO: optimizer 클래스 구현
            self._optimizer = optim.SGD(self._model.parameters(), lr=args.learning_rate)

            # loss function
            self._criterion = nn.CrossEntropyLoss()

    
    def _get_loss(self, batch):
        """_summary_

        Args:
            batch (_type_): _description_
        """
        dlg, crd, rnk = batch
        logits, _ = self._model(dlg, crd)
        loss = self._criterion(logits, rnk) * self._batch_size

        return loss


    def train(self):
        """_summary_
        """
        print('\n<Train>')
        print(f'Total examples in dataset: {self._num_examples}')

        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)

        init_epoch = 1

        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)

            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, map_location=torch.device(self._device))
                self._model.load_state_dict(checkpoint['model'])
                print(f'[*] Load Success: {file_name}\n')

                # backup
                if self._model_file == 'gAIa-final.pt':
                    print(f'time.strftime: {time.strftime("%m%d-%H%M%S")}')
                    
                    file_name_backup = os.path.join(self._model_path,
                                                    f'gAIa-final-{time.strftime("%m%d-%H%M%S")}.pt')

                    print(f'file_name_backup: {file_name_backup}')        
                    shutil.copy(file_name, file_name_backup)
                
                else:
                    init_epoch += int(re.findall('\d+', file_name)[-1])
            
            else:
                print('[!] checkpoints path does not exists...\n')

                return False

        self._model.to(self._device)

        # SI를 위해 필요한 변수 초기화
        W = {}
        p_old = {}

        for n, p in self._model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()


        ### Train ###
        end_epoch = self._epochs + init_epoch

        for curr_epoch in range(init_epoch, end_epoch):
            time_start = timeit.default_timer()
            losses = []

            losses_ce, losses_si = [], []
            iter_bar = tqdm(self._dataloader)

            for batch in iter_bar:
                self._optimizer.zero_grad()
                batch = [t.to(self._device) for t in batch]

                loss_ce = self._get_loss(batch).mean() # loss 계산
                losses_ce.append(loss_ce)
                
                if self._use_cl == True:
                    loss_si = surrogate_loss(self._model) # TODO
                    losses_si.append(loss_si)

                    loss = loss_ce + (si_c * loss_si)
                
                else:
                    loss = loss_ce

                loss.backward() # gradient 계산

                nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm) # clip gradient

                self._optimizer.step()
                losses.append(loss)

                # SI를 위해 필요한 변수 update
                for n, p in self._model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')

                        if p.grad is not None:
                            W[n].add_(-p.grad * (p.detach() - p_old[n]))
                        
                        p_old[n] = p.detach().clone()

                time_end = timeit.default_timer()

            print('-'*50)
            print('Epoch: {}/{}'.format(curr_epoch, end_epoch - 1))
            print('Time: {:.2f}sec'.format(time_end - time_start))
            print('Loss: {:.4f}'.format(torch.mean(torch.tensor(losses))))
            print('-'*50)                

            # 주기에 따른 모델 저장
            if curr_epoch % self._save_freq == 0:
                file_name = os.path.join(self._model_path, f'gAIa-{curr_epoch}.pt')
                torch.save({'model': self._model.state_dict()}, file_name)

            # wandb logging
            wandb.log({
                "Epoch": curr_epoch,
                "Train/Mean_CE_Loss": torch.mean(torch.tensor(losses_ce)),
                "Train/Mean_SI_Loss": torch.mean(torch.tensor(losses_si)),
                "Train/Total_Loss": torch.mean(torch.tensor(losses))
            })

        print(f'Done training; epoch limit {self._epochs} reached.\n')

        # SI를 위한 파라미터 업데이트
        update_omega(self._model, self._device, W, epsilon)

        # 최종 모델 저장
        file_name_final = os.path.join(self._model_path, 'gAIa-final.pt')
        torch.save({'model': self._model.state_dict()}, file_name_final)

        return True

    
    def _calculate_weighted_kendall_tau(self, pred: np.ndarray,
                                        label: np.ndarray,
                                        rnk_lst: np.ndarray):
        """Weighted kendall Tau score를 계산합니다.

        Args:
            pred (np.ndarray): 모델이 예측한 순위입니다.
            label (np.ndarray): 실제 정답 순위입니다.
            rnk_lst (np.ndarray): 배치 순서에 대한 순열입니다.

        Returns:
            _type_: _description_
        """

        total_count = 0
        total_corr = 0

        for p, l in zip(pred, label):
            corr, _ = stats.weightedtau(self._num_rnk - 1 - rnk_lst[l],
                                        self._num_rnk - 1 - rnk_lst[p])

            total_corr += corr
            total_count += 1

        return (total_corr / total_count)

    def _predict(self, eval_dlg, eval_crd):
        """_summary_

        Args:
            eval_dlg (_type_): _description_
            eval_crd (_type_): _description_
        """
        eval_num_examples = eval_dlg.shape[0]

        eval_dlg = torch.tensor(eval_dlg).to(self._device)
        eval_crd = torch.tensor(eval_crd).to(self._device)

        preds = []

        for start in range(0, eval_num_examples, self._batch_size):
            end = start + self._batch_size

            if end > eval_num_examples:
                end = eval_num_examples

            _, pred = self._model(eval_dlg[start:end], eval_crd[start:end])
            pred = pred.cpu().numpy()

            # sample별 예측 순위를 preds에 저장
            for j in range(end - start):
                preds.append(pred[j])

        return preds, eval_num_examples

    
    def _evaluate(self, eval_dlg, eval_crd):
        """_summary_

        Args:
            eval_dlg (_type_): _description_
            eval_crd (_type_): _description_
        """
        eval_num_examples = eval_dlg.shape[0]
        eval_corr = []
        
        rank_lst = np.array(list(permutations(np.arange(self._num_rnk), self._num_rnk)))

        eval_dlg = torch.tensor(eval_dlg).to(self._device)

        repeated_preds = []

        for i in range(self._num_eval):
            preds = []

            coordi, rnk = shuffle_coordi_and_ranking(eval_crd, self._num_rnk) # 평가용 데이터의 조합 및 순위를 shuffle
            coordi = torch.tensor(coordi).to(self._device)            

            for start in range(0, eval_num_examples, self._batch_size):
                end = start + self._batch_size

                if end > eval_num_examples:
                    end = eval_num_examples

                _, pred = self._model(eval_dlg[start:end], coordi[start:end])
                pred = pred.cpu().numpy()

                for j in range(end - start):
                    preds.append(pred[j])
            
            preds = np.array(preds)
            repeated_preds.append(preds)

            # metric 계산
            corr = self._calculate_weighted_kendall_tau(preds, rnk, rank_lst)
            eval_corr.append(corr)

        return repeated_preds, np.array(eval_corr), eval_num_examples

    
    def pred(self):
        """_summary_
        """
        print('\n<Predict>')

        # 모델 로드
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)

            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, map_location=torch.device('cpu'))

                self._model.load_state_dict(checkpoint['model'])
                self._model.to(self._device)
                
                print(f'[*] Load Success: {file_name}')

            else:
                print('[!] Checkpoint path does not exist...\n')

                return False

        else:
            return False

        time_start = timeit.default_timer()
        preds, num_examples = self._predict(self._tst_dlg, self._tst_crd)
        time_end = timeit.default_timer()

        print('-'*50)
        print(f'Prediction Time: {time_end-time_start:.2f}sec')
        print(f'# of Test Examples: {num_examples}')
        print('-'*50)

        return preds.astype(int)
    
    def test(self, csv_path, csv_name, task_num):
        """_summary_
        """
        print('\n<Evaluate>')

        # 모델 로드
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)

            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, map_location=torch.device('cpu'))

                self._model.load_state_dict(checkpoint['model'])
                self._model.to(self._device)
                
                print(f'[*] Load Success: {file_name}')

            else:
                print('[!] Checkpoint path does not exist...\n')

                return False

        else:
            return False

        # 예측 결과를 저장하는 csv 파일 생성 or 불러오기
        if not os.path.exists(csv_path):
            os.mkdir(csv_path)
        
        csv_file = os.path.join(csv_path, csv_name + '.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, index_col=0)
        else:
            columns = ['task1', 'task2', 'task3', 'task4', 'task5', 'task6']
            df = pd.DataFrame(index=['WKTC Avg/Best'], columns=columns)
        
        time_start = timeit.default_timer()
        repeated_preds, test_corr, num_examples = self._evaluate(self._tst_dlg, self._tst_crd)
        time_end = timeit.default_timer()

        print('-'*50)
        print(f'Prediction Time: {time_end-time_start:.2f}sec')
        print(f'# of Test Examples: {num_examples}')
        print(f'Average WKTC over iterations: {np.mean(test_corr):.4f}')
        print(f'Best WKTC: {np.max(test_corr):.4f}')
        print('-'*50)

        # 예측 결과 저장
        df[f'task{task_num}'] = f'{np.mean(test_corr):.4f}/{np.max(test_corr):.4f}'
        df.to_csv(csv_file)

        # 마지막 task data에 대한 평가가 끝나면, logging
        if task_num == 6:
            tbl = wandb.Table(dataframe=df)
            wandb.log({"Test/WKTC_Score": tbl})

    
    def ensemble(self, model_files: List[str]):
        # 모델 불러오기
        models = []

        for model_file in model_files:
            model = os.path.join(self._model_path, model_file)

            if os.path.exists(model):
                ckpt = torch.load(model, map_location=torch.device('cpu'))

                self._model.load_state_dict(ckpt['model'])
                self._model.to(self._device)

                models.append(self._model)
            
            else:
                print('[!] Checkpoint path does not exist...\n')

                return False


        # forward
        if self._mode == 'test':
            time_start = timeit.default_timer()

            eval_num_examples = self._tst_dlg.shape[0]
            eval_corr = []

            rank_lst = np.array(list(permutations(np.arange(self._num_rnk), self._num_rnk)))

            eval_dlg = torch.tensor(self._tst_dlg).to(self._device)

            repeated_preds = []

            for i in range(self._num_eval):
                preds_per_iter = []

                coordi, rnk = shuffle_coordi_and_ranking(self._tst_crd, self._num_rnk)
                coordi = torch.tensor(coordi).to(self._device)

                for start in range(0, eval_num_examples, self._batch_size):
                    end = start + self._batch_size

                    if end > eval_num_examples:
                        end = eval_num_examples

                    for model in models:
                        logits, _ = model(eval_dlg[start:end], coordi[start:end])

                        logits = logits.clone().detach().cpu().numpy()
                        
                        if start == 0:
                            logits_ensemble = logits
                        else:
                            logits_ensemble += logits    

                    # argmax를 통해 pred 구하기
                    preds_ensemble = np.argmax(logits_ensemble, 1)

                    for j in range(end - start):
                        preds_per_iter.append(preds_ensemble[j])

                preds_per_iter = np.array(preds_per_iter)
                repeated_preds.append(preds_per_iter)

                # metric 계산
                corr = self._calculate_weighted_kendall_tau(preds_per_iter, rnk, rank_lst)
                eval_corr.append(corr)

            time_end = timeit.default_timer()

            print('-'*50)
            print(f'Prediction Time: {time_end - time_start:.2f}sec')
            print(f'# of Test Examples: {eval_num_examples}')
            print(f'Average WKTC over iterations: {np.mean(np.array(eval_corr)):.4f}')
            print(f'Best WKTC: {np.max(np.array(eval_corr)):.4f}')
            print('-'*50)
        
        elif self._mode == 'pred':
            time_start = timeit.default_timer()
            
            eval_num_examples = self._tst_dlg.shape[0]

            eval_dlg = torch.tensor(self._tst_dlg).to(self._device)
            eval_crd = torch.tensor(self._tst_crd).to(self._device)

            preds = []

            for start in range(0, eval_num_examples, self._batch_size):
                end = start + self._batch_size

                if end > eval_num_examples:
                    end = eval_num_examples

                for model in models:
                    logits, _ = model(eval_dlg[start:end], eval_crd[start:end])

                    logits = logits.clone().detach().cpu().numpy()
                    
                    if start == 0:
                        logits_ensemble = logits
                    else:
                        logits_ensemble += logits    

                # argmax를 통해 pred 구하기
                preds_ensemble = np.argmax(logits_ensemble, 1)

                for j in range(end - start):
                    preds.append(preds_ensemble[j])                

            return preds, eval_num_examples
