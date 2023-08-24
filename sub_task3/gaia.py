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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# built-in library
import os
import re
import joblib
import shutil
import time
import timeit
from tqdm import tqdm

# external-library
import wandb
import numpy as np
from scipy import stats

# custom-library
from file_io import *
from requirement import *
from policy import *
from si import *

# of items in fashion coordination      
NUM_ITEM_IN_COORDI = 4
# of metadata features    
NUM_META_FEAT = 4
# of fashion coordination candidates        
NUM_RANKING = 3
# image feature size 
IMG_FEAT_SIZE = 4096
# SI parameter
si_c = 0.1
epsilon = 0.001


class Model(nn.Module):
    """ Model for AI fashion coordinator """
    def __init__(self, emb_size, key_size, mem_size, 
                 meta_size, hops, item_size, 
                 coordi_size, eval_node, num_rnk, 
                 use_batch_norm, use_dropout, zero_prob,
                 use_multimodal, img_feat_size):
        """
        initialize and declare variables

        Args:
            emb_size: 패션 아이템의 임베딩 벡터 차원. default to 128
            key_size(뭔지 모름): memory size for the MemN2N. default to 300
            mem_size:
            memory size for the MemN2N. default to 16
            데이터를 얼마나 사용할 지(기억할 지)를 결정하는 hparam. 차원의 크기라고 생각하면 됨.

            meta_size: 패션 아이템 메타데이터의 형태 특징 개수로 보임. default to 4
            hops:
            number of hops in the MemN2n. default to 3
            MemN2N 레이어를 반복하는 횟수.

            item_size: 각 카테고리별 패션 아이템의 개수가 저장되어있는 리스트.
            coordi_size: 하나의 코디를 구성할 때 필요한 패션 아이템의 개수. default to 4
            eval_node(뭔지 모름): evaluation network의 노드들. default to '[6000,6000,200][2000]'.
            num_rnk: 추천할 패션 아이템 조합의 개수. default to 3.
            use_batch_norm: batch normalization 옵션. default to False.
            use_dropout: Dropout 옵션. default to False.
            zero_prob: Dropout에 사용할 확률값. default to 0.0
            use_multimodal: multimodal input 옵션. default to False.
            img_feat_size: 이미지 feature의 크기. default to 4096
        """
        super().__init__()

        ### 분석중... ###
        # class instance for requirement estimation
        self._requirement = RequirementNet(emb_size, key_size, 
                                    mem_size, meta_size, hops)
        
        # class instance for ranking
        self._policy = PolicyNet(emb_size, key_size, item_size, 
                                 meta_size, coordi_size, eval_node,
                                 num_rnk, use_batch_norm,
                                 use_dropout, zero_prob,
                                 use_multimodal, img_feat_size)                                 

    def forward(self, dlg, crd):
        """
        build graph

        Args:
            dlg: 에피소드별 대화 임베딩 값. shape: (에피소드 개수, mem_size, emb_size)
            crd: 코디 조합. shape: (에피소드 개수, num_rnk, coordi_size * emb_size * 4)

        Return:
            logits:
            preds:
        """
        req = self._requirement(dlg) # (num_batch, key_size)
        logits = self._policy(req, crd) # (num_batch, 6)
        preds = torch.argmax(logits, 1) # (num_batch, )

        return logits, preds


class gAIa(object):
    """ Class for AI fashion coordinator """
    def __init__(self, args, device, name='gAIa'):
        """
        initialize
        """
        self._device = device
        self._batch_size = args.batch_size # default to 100
        self._model_path = args.model_path # 저장하거나 저장된 학습 모델의 경로명
        self._model_file = args.model_file # 저장하거나 저장된 학습 모델의 파일명
        self._epochs = args.epochs # default to 10
        self._max_grad_norm = args.max_grad_norm # 최대 그래디언트 값. default to 40.0
        self._save_freq = args.save_freq # 모델의 저장 주기. default to 2
        self._num_eval = args.evaluation_iteration # 입력의 순서를 바꾸어 수행할 평가 횟수. default to 10
        self._in_file_trn_dialog = args.in_file_trn_dialog # training dialog DB(학습용 대화 DB)
        self._use_cl = args.use_cl  # 연속 학습 적용 유무. default to True
        use_dropout = args.use_dropout # 드롭아웃 기법 사용 유무. default to False
        if args.mode == 'test':
            use_dropout = False
        
        # class instance for subword embedding
        self._swer = SubWordEmbReaderUtil(args.subWordEmb_path) # args.subWordEmb_path: path of subword embedding.
        self._emb_size = self._swer.get_emb_size() # 128
        self._meta_size = NUM_META_FEAT # 4
        self._coordi_size = NUM_ITEM_IN_COORDI # 4
        self._num_rnk = NUM_RANKING # 3
        feats_size = IMG_FEAT_SIZE # 4096
        
        
        # read metadata DB: fashion item metadata를 읽어서 전처리하는 과정
        self._metadata, self._idx2item, self._item2idx, \
            self._item_size, self._meta_similarities, \
            self._feats = make_metadata(args.in_file_fashion, self._swer, 
                                self._coordi_size, self._meta_size,
                                args.use_multimodal, args.in_file_img_feats,
                                feats_size)
        
        # prepare DB for training: 모델 학습에 사용할 DB 준비 과정
        if args.mode == 'train':
            self._dlg, self._crd, self._rnk = make_io_data('prepare', 
                        args.in_file_trn_dialog, self._swer, args.mem_size,
                        self._coordi_size, self._item2idx, self._idx2item, 
                        self._metadata, self._meta_similarities, self._num_rnk,
                        args.permutation_iteration, args.num_augmentation, 
                        args.corr_thres, self._feats)
            self._num_examples = len(self._dlg) # 학습 데이터 개수
            
            # dataloader
            dataset = TensorDataset(torch.tensor(self._dlg), 
                                    torch.tensor(self._crd), 
                                    torch.tensor(self._rnk, dtype=torch.long))
            self._dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

            # 학습 상태를 모니터링하기 위해 validation set도 불러옴
            self._tst_dlg, self._tst_crd, _ = make_io_data('eval', 
                    args.in_file_tst_dialog, self._swer, args.mem_size,
                    self._coordi_size, self._item2idx, self._idx2item, 
                    self._metadata, self._meta_similarities, self._num_rnk,
                    args.num_augmentation, args.num_augmentation, 
                    args.corr_thres, self._feats)
            
            # 불러온 validation set을 evaluation에 맞게 db shuffling
            self._tst_crd_per_iter, self._tst_rnk_per_iter = [], []
            for _ in range(self._num_eval):
                coordi, rnk = shuffle_coordi_and_ranking(self._tst_crd, self._num_rnk)

                self._tst_crd_per_iter.append(coordi)
                self._tst_rnk_per_iter.append(rnk)
        
        # prepare DB for evaluation: 모델 성능 평가에 사용할 DB를 준비
        # TODO: mode에 따라 불러오는 data가 다름 -> 한 번에 불러와서, 학습 과정에 성능 평가가 되도록 코드 수정
        elif args.mode in ['test', 'pred'] :
            self._tst_dlg, self._tst_crd, _ = make_io_data('eval', 
                    args.in_file_tst_dialog, self._swer, args.mem_size,
                    self._coordi_size, self._item2idx, self._idx2item, 
                    self._metadata, self._meta_similarities, self._num_rnk,
                    args.num_augmentation, args.num_augmentation, 
                    args.corr_thres, self._feats)
            self._num_examples = len(self._tst_dlg)
        
        # model
        self._model = Model(self._emb_size, args.key_size, args.mem_size, 
                            self._meta_size, args.hops, self._item_size, 
                            self._coordi_size, args.eval_node, self._num_rnk, 
                            args.use_batch_norm, use_dropout, 
                            args.zero_prob, args.use_multimodal, 
                            feats_size)
        
        ### 분석중 ... ###
        # continual learning과 관련해서 필요한 코드로 보이는데,,
        print('\n<model parameters>')
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                print(name)
                n = name.replace('.', '__')
                self._model.register_buffer('{}_SI_prev_task'.format(n), 
                                            param.detach().clone())
                self._model.register_buffer('{}_SI_omega'.format(n), 
                                            torch.zeros(param.shape))

        if args.mode == 'train':
            # optimizer
            self._optimizer = optim.SGD(self._model.parameters(),
                                        lr=args.learning_rate)
            # loss function
            self._criterion = nn.CrossEntropyLoss()

    def _get_loss(self, batch):
        """
        calculate loss

        Args:
            batch:
            (대화문 임베딩 벡터, 코디 임베딩 벡터, 랭킹)로 구성.
            대화문 임베딩 벡터: (num_batch, 16, 128)
            코디 임베딩 벡터: (num_batch, 3, 2048)
            랭킹: (num_batch,)

        Return:
            loss:
            미니배치 크기만큼 누적한 오차 수치.
            nn.CrossEntropyLoss의 경우 기본적으로 mean이 적용되는데
            batch_size를 곱한 뒤 .mean()을 적용하는 이유는 잘 모르겠음.
        """
        dlg, crd, rnk = batch
        logits, _ = self._model(dlg, crd)
        loss = self._criterion(logits, rnk) * self._batch_size

        return loss


    def train(self):
        """
        training
        """
        print('\n<Train>')
        print('total examples in dataset: {}'.format(self._num_examples))

        # gAIa_CL_model 디렉토리 만들고
        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)
        
        init_epoch = 1
        # 불러올 모델이 있는 경우 불러옴
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            if os.path.exists(file_name):
                # weights를 불러온다
                checkpoint = torch.load(file_name, map_location=torch.device(self._device))
                self._model.load_state_dict(checkpoint['model'])
                print('[*] load success: {}\n'.format(file_name))

                # 만약 불러온 모델이 이전 task의 최종 모델이었다면
                if self._model_file == 'gAIa-final.pt':
                    print('time.strftime: ')
                    print(time.strftime("%m%d-%H%M%S"))
                    file_name_backup = os.path.join(self._model_path, 
                        'gAIa-final-{}.pt'.format(time.strftime("%m%d-%H%M%S")))
                    print('file_name_backup: ')
                    print(file_name_backup)

                    # backup을 수행 -> 다른 task를 학습하면서 기존 weight 정보가 변경되기 때문
                    shutil.copy(file_name, file_name_backup)
                    
                # 아니면 그냥 불러온 모델을 사용하는데
                else:
                    # 이전 모델을 학습할 때 사용한 에폭만큼 init_epoch에 더해줌.
                    init_epoch += int(re.findall('\d+', file_name)[-1])

            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        
        # device에 model을 옮겨주고
        self._model.to(self._device)
        
        ### 분석중 ... ###
        W = {}
        p_old = {}
        
        # 현재 모델에서 parameters를 불러오는데
        for n, p in self._model.named_parameters():
            # requires_grad가 True인 경우
            if p.requires_grad:
                n = n.replace('.', '__') # 이름을 변경하고(왜?)
                W[n] = p.data.clone().zero_() # W dict에 0으로 채운 파라미터를 복사
                p_old[n] = p.data.clone() # p_old dict에 파라미터를 그대로 복사
        

        end_epoch = self._epochs + init_epoch
        for curr_epoch in range(init_epoch, end_epoch):
            time_start = timeit.default_timer()
            losses = []
            iter_bar = tqdm(self._dataloader)
            
            for batch in iter_bar:
                self._optimizer.zero_grad()
                batch = [t.to(self._device) for t in batch]
                
                # loss 계산
                loss_ce = self._get_loss(batch).mean()

                # continual learning에 따른 loss 계산
                if self._use_cl == True:
                    loss_si = surrogate_loss(self._model)
                    loss = loss_ce + si_c*loss_si
                else:
                    loss = loss_ce

                # 미분
                loss.backward()

                # gradient clip 적용
                nn.utils.clip_grad_norm_(self._model.parameters(), 
                                         self._max_grad_norm)
                
                # 업데이트
                self._optimizer.step()

                losses.append(loss)


                for n, p in self._model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            # W[n]에 저장되어있던 파라미터에 괄호 안의 값을 저장
                            # 왜 저장하는 거지??? -> update_omega 함수에 필요하므로..?
                            W[n].add_(-p.grad*(p.detach()-p_old[n]))
                        p_old[n] = p.detach().clone()
                        
            time_end = timeit.default_timer()
            print('-'*50)
            print('Epoch: {}/{}'.format(curr_epoch, end_epoch - 1))
            print('Time: {:.2f}sec'.format(time_end - time_start))
            print('Loss: {:.4f}'.format(torch.mean(torch.tensor(losses))))
            print('-'*50)

            if curr_epoch % self._save_freq == 0:
                file_name = os.path.join(self._model_path, 
                                         'gAIa-{}.pt'.format(curr_epoch))
                torch.save({'model': self._model.state_dict()}, file_name)

        print('Done training; epoch limit {} reached.\n'.format(self._epochs))
        
        ### 분석중... ###
        update_omega(self._model, self._device, W, epsilon)

        file_name_final = os.path.join(self._model_path, 'gAIa-final.pt')
        torch.save({'model': self._model.state_dict()}, file_name_final)

        return True
        
    def _calculate_weighted_kendall_tau(self, pred, label, rnk_lst):
        """
        calcuate Weighted Kendall Tau Correlation

        Args:
            pred: 대화와 코디 DB를 바탕으로 모델이 예측한 순위.
            label: 정답 순위.
            rnk_lst: 순열이 저장되어있는 배열

        Return:
            (total_corr / total_count): 
        """
        total_count = 0
        total_corr = 0
        for p, l in zip(pred, label):
            # 순서를 뒤집어주는 이유가 뭐지?????
            # p는 모델이 예측한 순위, l은 실제 순위
            # self._num_rnk-1: 3-1=2 -> 2 - np.array([0, 1, 2]) = np.array([2, 1, 0]) ????
            corr, _ = stats.weightedtau(self._num_rnk-1-rnk_lst[l], #
                                        self._num_rnk-1-rnk_lst[p]) #
            total_corr += corr
            total_count += 1

        return (total_corr / total_count)
    
    def _predict(self, eval_dlg, eval_crd):
        """
        predict
        """
        eval_num_examples = eval_dlg.shape[0]
        eval_dlg = torch.tensor(eval_dlg).to(self._device)
        eval_crd = torch.tensor(eval_crd).to(self._device)
        preds = []

        for start in range(0, eval_num_examples, self._batch_size):
            end = start + self._batch_size

            if end > eval_num_examples:
                end = eval_num_examples
            _, pred = self._model(eval_dlg[start:end],
                                  eval_crd[start:end])
            
            pred = pred.cpu().numpy()

            for j in range(end-start):
                preds.append(pred[j])

        preds = np.array(preds)

        return preds, eval_num_examples
    
    def _evaluate(self, eval_dlg, eval_crd, eval_crd_per_iter=[], eval_rnk_per_iter=[]):
        """
        evaluate

        Args:
            eval_dlg:
            에피소드별로 구분된 평가용 대화 임베딩.
            shape: (전체 에피소드 개수, 16, 128)

            eval_crd:
            에피소드별로 구분된 평가용 코디 임베딩.
            shape: (전체 에피소드 개수, 3, 2048)

            eval_crd_per_iter, eval_rnk_per_iter:
            evaluation을 동일한 데이터로 하기 위해 사용하는 변수.
            shuffle_coordi_and_ranking 함수를 호출할 때 코디와 순위가 무작위로 섞이기 때문에,
            본 함수를 호출할 때 마다 validation 데이터의 구성이 달라지는데, 이를 방지하기 위해 사용.

        Return:
            repeated_preds:
            np.array(eval_corr):
            eval_num_examples:
        """
        eval_num_examples = eval_dlg.shape[0] # 평가용 DB의 전체 개수
        eval_corr = [] # ?

        # 순열 속 element의 순서에 따라 순위를 매겨둔 리스트
        rank_lst = np.array(list(permutations(np.arange(self._num_rnk), self._num_rnk)))

        # device로 옮기고
        eval_dlg = torch.tensor(eval_dlg).to(self._device)
        
        # 평가 시작
        repeated_preds = []

        # self._num_eval: 입력의 순서를 바꿔 여러번 평가를 진행하기 위해 설정하는 hparam
        # 평가 data를 한 번만 사용하는 것이 아닌, self._num_eval만큼 반복해서 사용
        for i in range(self._num_eval):
            preds = []
            
            # 평가용 데이터가 만들어져 있다면
            if eval_crd_per_iter and eval_rnk_per_iter:
                # iteration 별 데이터를 가져와서 evaluation에 사용
                coordi = eval_crd_per_iter[i]
                rnk = eval_rnk_per_iter[i]
                
            # 평가용 데이터를 새로 만들어야 한다면
            else:
                # DB Shuflling: 패션 아이템의 다양한 조합에 대한 모델의 순위 예측 성능을 확인하기 위해 shuffling
                coordi, rnk = shuffle_coordi_and_ranking(eval_crd, self._num_rnk)

            coordi = torch.tensor(coordi).to(self._device)

            for start in range(0, eval_num_examples, self._batch_size):
                end = start + self._batch_size

                if end > eval_num_examples:
                    end = eval_num_examples

                # 배치 단위로 묶어서 forward
                _, pred = self._model(eval_dlg[start:end], coordi[start:end])
                pred = pred.cpu().numpy()

                # 배치 단위로 묶인 예측 결과를 하나씩 리스트에 저장(comprehension을 왜 안쓸까..)
                for j in range(end-start):
                    preds.append(pred[j])

            preds = np.array(preds)

            ### 분석중... ###
            # compute Weighted Kendall Tau Correlation
            repeated_preds.append(preds)
            corr = self._calculate_weighted_kendall_tau(preds, rnk, rank_lst)
            eval_corr.append(corr)

        return repeated_preds, np.array(eval_corr), eval_num_examples
    
    def pred(self):
        """
        create prediction.csv
        """
        print('\n<Predict>')

        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, map_location=torch.device('cpu'))
                self._model.load_state_dict(checkpoint['model'])
                self._model.to(self._device)
                print('[*] load success: {}\n'.format(file_name))
            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        else:
            return False
        time_start = timeit.default_timer()
        # predict
        preds, num_examples = self._predict(self._tst_dlg, self._tst_crd)
        time_end = timeit.default_timer()
        print('-'*50)
        print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('-'*50)
        return preds.astype(int)

    def test(self):
        """
        get scores using sample test set
        """
        print('\n<Evaluate>')

        # 모델 불러오기
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)

            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, 
                                        map_location=torch.device(self._device))
                self._model.load_state_dict(checkpoint['model'])
                self._model.to(self._device)
                print('[*] load success: {}\n'.format(file_name))

            else:
                print('[!] checkpoints path does not exist...\n')
                return False
            
        else:
            return False
        
        time_start = timeit.default_timer()

        # evaluation
        repeated_preds, test_corr, num_examples = self._evaluate(self._tst_dlg, self._tst_crd)

        time_end = timeit.default_timer()

        print('-'*50)
        print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('Average WKTC over iterations: {:.4f}'.format(np.mean(test_corr)))
        print('Best WKTC: {:.4f}'.format(np.max(test_corr)))
        print('-'*50)

        return np.mean(test_corr)

    # TODO: continual learning 적용이 되도록 수정하기
    def custom_train(self):
        print('\n<Train>')
        print('total examples in dataset: {}'.format(self._num_examples))

        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)

        # 모델 불러오기(self._model_file이 존재하는 경우에만)
        init_epoch = 1
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)

            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, map_location=torch.device(self._device))
                self._model.load_state_dict(checkpoint['model'])

                print('[*] Load Success: {}\n'.format(file_name))

                if self._model_file == 'gAIa-final.pt':
                    print('time.strftime: ')
                    print(time.strftime("%m%d-%H%M%S"))
                    file_name_backup = os.path.join(self._model_path,
                                                    'gAIa-final-{}.pt'.format(time.strftime("%m%d-%H%M%S")))
                    print('file_name_backup: ')
                    print(file_name_backup)

                    shutil.copy(file_name, file_name_backup)

                else:
                    init_epoch += int(re.findall('\d+', file_name)[-1])
        
            else:
                print('[!] checkpoints path does not exist...\n')
                return False
            
        self._model.to(self._device)

        # SI를 적용하기 위해 사용하는 코드 블록
        W = {}
        p_old = {}

        # 현재 모델에서 parameters를 불러오는데
        for n, p in self._model.named_parameters():
            # requires_grad가 True인 경우
            if p.requires_grad:
                n = n.replace('.', '__') # 이름을 변경하고(왜?)
                W[n] = p.data.clone().zero_() # W dict에 0으로 채운 파라미터를 복사
                p_old[n] = p.data.clone() # p_old dict에 파라미터를 그대로 복사


        end_epoch = self._epochs + init_epoch
        for curr_epoch in range(init_epoch, end_epoch):
            self._model.train()
            time_start = timeit.default_timer()
            losses = []
            iter_bar = tqdm(self._dataloader)
            
            # training
            for batch in iter_bar:
                self._optimizer.zero_grad()
                batch = [t.to(self._device) for t in batch]
                
                # loss 계산
                loss_ce = self._get_loss(batch).mean()

                # continual learning에 따른 loss 계산
                if self._use_cl == True:
                    loss_si = surrogate_loss(self._model)
                    loss = loss_ce + si_c*loss_si
                else:
                    loss = loss_ce

                # 미분
                loss.backward()

                # gradient clip 적용
                nn.utils.clip_grad_norm_(self._model.parameters(), 
                                         self._max_grad_norm)
                
                # 업데이트
                self._optimizer.step()

                losses.append(loss)

                for n, p in self._model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            # W[n]에 저장되어있던 파라미터에 괄호 안의 값을 저장
                            # 왜 저장하는 거지??? -> update_omega 함수에 필요하므로..?
                            W[n].add_(-p.grad*(p.detach()-p_old[n]))

                        p_old[n] = p.detach().clone()
            
            
            time_end = timeit.default_timer()

            print('-'*50)
            print('Epoch: {}/{}'.format(curr_epoch, end_epoch - 1))
            print('Time: {:.2f}sec'.format(time_end - time_start))
            print('Loss: {:.4f}'.format(torch.mean(torch.tensor(losses))))
            print('-'*50)

            if curr_epoch % self._save_freq == 0:
                file_name = os.path.join(self._model_path, 
                                         'gAIa-{}.pt'.format(curr_epoch))
                torch.save({'model': self._model.state_dict()}, file_name)

            # evaluation
            print('\n<Evaluate>')
            self._model.eval()

            time_start = timeit.default_timer()
            repeated_preds, test_corr, num_examples = self._evaluate(self._tst_dlg, self._tst_crd,
                                                                     self._tst_crd_per_iter, self._tst_rnk_per_iter)
            time_end = timeit.default_timer()

            print('-'*50)
            print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
            print('# of Test Examples: {}'.format(num_examples))
            print('Average WKTC over iterations: {:.4f}'.format(np.mean(test_corr)))
            print('Best WKTC: {:.4f}'.format(np.max(test_corr)))
            print('-'*50)

            # wandb logging
            wandb.log({
                "Epoch": curr_epoch,
                "Train/Mean_CE_Loss": torch.mean(torch.tensor(losses)),
                "Val/Mean_WKTC": np.mean(test_corr),
                "Val/Best_WKTC": np.max(test_corr)
            })


        print('Done training; epoch limit {} reached.\n'.format(self._epochs))
        
        ### 분석중... ###
        update_omega(self._model, self._device, W, epsilon)

        file_name_final = os.path.join(self._model_path, 'gAIa-final.pt')
        torch.save({'model': self._model.state_dict()}, file_name_final)

        return True
    