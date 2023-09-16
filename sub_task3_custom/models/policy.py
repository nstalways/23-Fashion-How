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

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# built-in library
import math
from collections import OrderedDict


# TODO: docstring 마무리하기
class PolicyNet(nn.Module):
    """
    Class for policy network
    사용자의 요구 사항을 바탕으로 패션 아이템 조합의 순위를 평가합니다.
    """
    def __init__(self, item_size: list, emb_size: int=128, key_size: int=300,
                 meta_size: int=4, coordi_size: int=4, num_rnk: int=3,
                 eval_node: str='[6000,6000,200][2000]', 
                 use_batch_norm: bool=False, use_dropout: bool=True,
                 zero_prob: float=0.5, use_multimodal: bool=False,
                 img_feat_size: int=4096, name: str='PolicyNet'):
        """필요한 변수들을 선언하고, 초기화합니다.

        Args:
            item_size (list): 각 카테고리 별 패션 아이템의 개수가 저장되어 있습니다(패션 아이템은 4개의 카테고리로 구분됩니다).
            emb_size (int, optional): _description_. Defaults to 128.
            key_size (int, optional): MemN2N 출력 벡터의 차원입니다. Defaults to 300.
            meta_size (int, optional): 패션 아이템 메타데이터의 형태 특징 개수입니다(README.pdf 참고). Defaults to 4.
            coordi_size (int, optional): 하나의 코디를 구성할 때 필요한 패션 아이템의 개수를 의미합니다. Defaults to 4.
            num_rnk (int, optional): 추천할 패션 아이템 조합의 개수입니다. Defaults to 3.
            eval_node (str, optional): 
            MLP 모델을 생성할 때 사용하는 값으로, mlp layer별 노드의 개수를 의미합니다.
            Layer를 깊게 쌓고 싶은 경우, 리스트 내부에 값을 추가하면 됩니다. Defaults to '[6000,6000,200][2000]'.

            use_batch_norm (bool, optional): batch normalization을 적용할 지 선택합니다. Defaults to False.
            use_dropout (bool, optional): dropout을 적용할 지 선택합니다. Defaults to True.
            zero_prob (float, optional): Dropout 비율입니다. Defaults to 0.5.
            use_multimodal (bool, optional): 이미지 feature를 활용할 지 선택합니다. Defaults to False.
            img_feat_size (int, optional): 이미지 feature의 차원 크기입니다. Defaults to 4096.
            name (str, optional): 모델 이름입니다. Defaults to 'PolicyNet'.
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

        # 모델 생성에 필요한 hparam 정의
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
                # 'layer%s_silu'%(i+1): nn.SiLU()}) # ReLU -> SiLU 교체
            
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
        for i in range(self._num_hid_layer_rnk + 1):
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
                    # 'layer%s_silu'%(i+1): nn.SiLU()}) # ReLU -> SiLU 교체
                
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
        """_summary_

        Args:
            crd (_type_):
            전체 DB에서 n번째로 추천한 코디 임베딩 배열.
            shape: (num_batch, 2048) -> 모든 에피소드에 대해 n번째로 추천한 코디 임베딩이라는 의미

            req (_type_):
            RequirementNet의 출력으로, 사용자의 요구사항에 대한 정보가 담겨있는 벡터라고 생각할 수 있음.
            shape: (num_batch, 300)

        Returns:
            _type_: _description_
        """
        # 코디 임베딩과 요구사항 임베딩을 연결
        crd_and_req = torch.cat((crd, req), 1) # (num_batch, 2348)

        # forward
        evl = self._mlp_eval(crd_and_req) # (num_batch, 200)

        return evl
    
    def _ranking_coordi(self, in_rnk):
        """코디의 순위를 매깁니다.

        Args:
            in_rnk (_type_): 코디 별 평가 정보와 req 정보가 저장되어 있습니다. (num_batch, 900)

        Returns:
            _type_: 코디가 몇 번째 순위인지를 나타내는 행렬입니다. (num_batch, 6)
        """
        out_rnk = self._mlp_rnk(in_rnk)

        return out_rnk
        
    def forward(self, req, crd):
        """forward 연산을 정의합니다.

        Args:
            req (_type_): 패션 아이템 조합 순위를 평가할 때 사용하는 feature로 dlg에서 추출됩니다. (num_batch, key_size)
            crd (_type_): 전체 DB에서 n번째로 추천한 코디 임베딩 배열입니다. shape: (num_batch, 3, 2048)

        Returns:
            _type_: _description_
        """
        # 코디 임베딩에서 에피소드 차원과 추천 차원을 전치
        crd_tr = torch.transpose(crd, 1, 0)

        # 추천 차원을 반복
        for i in range(self._num_rnk):
            # 코디에 대한 평가 정보가 담겨있는 feature를 획득
            crd_eval = self._evaluate_coordi(crd_tr[i], req) # (num_batch, 200)

            # 반복하면서 concatenation을 수행
            if i == 0:
                in_rnk = crd_eval
            else:
                in_rnk = torch.cat((in_rnk, crd_eval), 1)

        in_rnk = torch.cat((in_rnk, req), 1) # req 정보를 추가해줌. (num_batch, 900)
        out_rnk = self._ranking_coordi(in_rnk)

        return out_rnk


# Test Code
if __name__ == "__main__":
    model = PolicyNet(emb_size=128, key_size=300,
                      item_size=[100, 100, 100, 100], meta_size=4,
                      coordi_size=4, eval_node='[6000,3000,1000,500,200][2000,1000,500,100]',
                      num_rnk=3, use_batch_norm=True, use_dropout=True,
                      zero_prob=0.0, use_multimodal=False, img_feat_size=4096)

    print(model._mlp_eval)
    print('-' * 200)
    print(model._mlp_rnk)
    print('-' * 200)

    req, crd = torch.randn((100, 300)), torch.randn((100, 3, 2048))
    out = model(req, crd)
    print(f"Out: {out.shape}")
    
    softmax = nn.Softmax(dim=0)
    print(softmax(out[0]))