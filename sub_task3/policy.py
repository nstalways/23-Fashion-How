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

Update: 2022.06.16.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


class PolicyNet(nn.Module):
    """Class for policy network"""
    def __init__(self, emb_size, key_size, item_size, meta_size, 
                 coordi_size, eval_node, num_rnk, use_batch_norm, 
                 use_dropout, zero_prob, use_multimodal,
                 img_feat_size, name='PolicyNet'):
        """
        initialize and declare variables

        Args:
            emb_size: 패션 아이템의 임베딩 벡터 차원. default to 128
            key_size(뭔지 모름): memory size for the MemN2N. default to 300
            item_size: 각 카테고리별 패션 아이템의 개수가 저장되어있는 리스트.
            meta_size: 패션 아이템 메타데이터의 형태 특징 개수로 보임. default to 4
            coordi_size: 하나의 코디를 구성할 때 필요한 패션 아이템의 개수. default to 4
            eval_node(뭔지 모름): evaluation network의 노드들. default to '[6000,6000,200][2000]'.
            num_rnk: 추천할 패션 아이템 조합의 개수. default to 3.
            use_batch_norm: batch normalization 옵션. default to False.
            use_dropout: Dropout 옵션. default to False.
            zero_prob: Dropout에 사용할 확률값. default to 0.0
            use_multimodal: multimodal input 옵션. default to False.
            img_feat_size: 이미지 feature의 크기. default to 4096
        
        Return:

        """
        super().__init__()
        self._item_size = item_size
        self._emb_size = emb_size
        self._key_size = key_size
        self._meta_size = meta_size
        self._coordi_size = coordi_size
        self._num_rnk = num_rnk
        self._use_dropout = use_dropout
        self._zero_prob = zero_prob
        self._name = name

        ### 분석중 ...(default) ###
        buf = eval_node[1:-1].split('][') # ['6000,6000,200', '2000']
        self._num_hid_eval = list(map(int, buf[0].split(','))) # [6000, 6000, 200]
        self._num_hid_rnk = list(map(int, buf[1].split(','))) # [2000]
        self._num_hid_layer_eval = len(self._num_hid_eval) # 3
        self._num_hid_layer_rnk = len(self._num_hid_rnk) # 1
        
        mlp_eval_list = OrderedDict([])
        num_in = self._emb_size * self._meta_size * self._coordi_size \
                    + self._key_size # 128 * 4 * 4 + 300
        
        if use_multimodal:
            num_in += img_feat_size

        # self._mlp_eval 모델 선언부
        self._count_eval = 0
        for i in range(self._num_hid_layer_eval):
            num_out = self._num_hid_eval[i]
            mlp_eval_list.update({ 
                'layer%s_linear'%(i+1): nn.Linear(num_in, num_out)}) 
            mlp_eval_list.update({
                'layer%s_relu'%(i+1): nn.ReLU()})
            
            if use_batch_norm:
                mlp_eval_list.update({
                    'layer%s_bn'%(i+1): nn.BatchNorm1d(num_out)})
                
            if self._use_dropout:
                mlp_eval_list.update({
                    'layer%s_dropout'%(i+1): nn.Dropout(p=self._zero_prob)})
                
            self._count_eval += (num_in * num_out + num_out)            
            num_in = num_out
            
        self._eval_out_node = num_out 
        self._mlp_eval = nn.Sequential(mlp_eval_list) 

        # self._mlp_rnk 모델 선언부
        mlp_rnk_list = OrderedDict([])
        num_in = self._eval_out_node * self._num_rnk + self._key_size
        for i in range(self._num_hid_layer_rnk+1):
            if i == self._num_hid_layer_rnk:
                num_out = math.factorial(self._num_rnk)
                mlp_rnk_list.update({ 
                    'layer%s_linear'%(i+1): nn.Linear(num_in, num_out)}) 
            else:
                num_out = self._num_hid_rnk[i]
                mlp_rnk_list.update({ 
                    'layer%s_linear'%(i+1): nn.Linear(num_in, num_out)}) 
                mlp_rnk_list.update({
                    'layer%s_relu'%(i+1): nn.ReLU()})
                
                if use_batch_norm:
                    mlp_rnk_list.update({
                    'layer%s_bn'%(i+1): nn.BatchNorm1d(num_out)})

                if self._use_dropout:
                    mlp_rnk_list.update({
                    'layer%s_dropout'%(i+1): nn.Dropout(p=self._zero_prob)})

            self._count_eval += (num_in * num_out + num_out)
            num_in = num_out

        self._mlp_rnk = nn.Sequential(mlp_rnk_list) 

    def _evaluate_coordi(self, crd, req):
        """
        evaluate candidates

        Args:
            crd:
            전체 DB에서 n번째로 추천한 코디 임베딩 배열.
            shape: (num_batch, 2048) -> 모든 에피소드에 대해 n번째로 추천한 코디 임베딩이라는 의미

            req:
            RequirementNet의 출력으로, 사용자의 요구사항에 대한 정보가 담겨있는 벡터라고 생각할 수 있음.
            shape: (num_batch, 300)

        Return:
            evl:
        """
        # 코디 임베딩과 요구사항 임베딩을 연결
        crd_and_req = torch.cat((crd, req), 1) # (num_batch, 2348)

        # forward
        evl = self._mlp_eval(crd_and_req)

        return evl
    
    def _ranking_coordi(self, in_rnk):
        """
        rank candidates

        Args:
            in_rnk:

        Return:
            
        """
        out_rnk = self._mlp_rnk(in_rnk)

        return out_rnk
        
    def forward(self, req, crd):
        """
        build graph for evaluation and ranking

        Args:
            req:
            RequirementNet의 출력으로, 사용자의 요구사항에 대한 정보가 담겨있는 벡터라고 생각할 수 있음.

            crd:
            코디 조합의 임베딩 값이 저장된 배열.
            (전체 에피소드 개수, 에피소드 당 추천하는 코디 개수,
            하나의 코디를 이루는 아이템 개수, 임베딩 차원) -> (num_batch, 3, 2048)

        Return:
            out_rnk: crd의 순위를 나타내는 logits -> (num_batch, 6)
        """
        # 코디 임베딩에서 에피소드 차원과 추천 차원을 전치
        crd_tr = torch.transpose(crd, 1, 0)

        # 추천 차원을 반복하는데
        for i in range(self._num_rnk):
            # 
            crd_eval = self._evaluate_coordi(crd_tr[i], req)

            if i == 0:
                in_rnk = crd_eval
            else:
                in_rnk = torch.cat((in_rnk, crd_eval), 1)

        in_rnk = torch.cat((in_rnk, req), 1)
        out_rnk = self._ranking_coordi(in_rnk)

        return out_rnk


# Test Code
if __name__ == "__main__":
    model = PolicyNet(emb_size=128, key_size=300,
                      item_size=[100, 100, 100, 100], meta_size=4,
                      coordi_size=4, eval_node='[6000,6000,200][2000]',
                      num_rnk=3, use_batch_norm=True, use_dropout=True,
                      zero_prob=0.0, use_multimodal=False, img_feat_size=4096)

    print(model._mlp_eval)
    print('-' * 200)
    print(model._mlp_rnk)
    print('-' * 200)

    req, crd = torch.randn((100, 300)), torch.randn((100, 3, 2048))
    out = model(req, crd)
    print(f"Out: {out.shape}")
    
    softmax = nn.Softmax()
    print(softmax(out[0]))