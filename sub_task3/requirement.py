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


# 질문이 없는데 QA 모델을 사용하는 이유가 뭘까..?
class MemN2N(nn.Module):
    """End-To-End Memory Network."""
    def __init__(self, embedding_size, key_size, mem_size, 
                 meta_size, hops=3, nonlin=None, name='MemN2N'):
        """
        initialize and declare variables

        Args:
            embedding_size: 임베딩 벡터 차원. default to 128
            key_size(뭔지 모름): memory size for the MemN2N. default to 300:
            mem_size:
            memory size for the MemN2N. default to 16
            데이터를 얼마나 사용할 지(기억할 지)를 결정하는 hparam. 차원의 크기라고 생각하면 됨.

            meta_size: 패션 아이템 메타데이터의 형태 특징 개수로 보임. default to 4
            hops:
            number of hops in the MemN2n. default to 3
            multiple computational steps을 hops이라고 부름. hop size만큼 layer를 여러 번 쌓음.

            nonlin: 최종 출력을 계산하기 전 비선형 함수를 적용하고 싶을 때 사용하는 인자.
            nn.Linear()와 같은 레이어 객체를 넘겨주는 방식으로 사용할 수 있음.


            name: 모델 이름
        """
        super().__init__()
        self._embedding_size = embedding_size
        self._embedding_size_x2 = embedding_size * 2
        self._mem_size = mem_size
        self._meta_size = meta_size
        self._key_size = key_size
        self._hops = hops
        self._nonlin = nonlin
        self._name = name
        
        # 정규 분포를 따르는 (1, self._embedding_size) 모양의 무작위 텐서를 쿼리에 저장.
        # 본 task의 경우 질문 data가 없기 때문에 무작위로 생성한 것으로 보임.
        self._queries = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(1, self._embedding_size)), 
                requires_grad=True)
        
        # 문장의 임베딩 벡터(key)를 구하기 위해 사용
        self._A = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(self._embedding_size, self._embedding_size_x2)), 
                requires_grad=True)
        
        # 쿼리의 임베딩 벡터를 구하기 위해 사용
        self._B = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(self._embedding_size, self._embedding_size_x2)), 
                requires_grad=True)
        
        # 문장의 임베딩 벡터(value)를 구하기 위해 사용
        self._C = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(self._embedding_size, self._embedding_size_x2)), 
                requires_grad=True)
        
        # 쿼리 임베딩 벡터에 곱해지는 가중치
        self._H = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(self._embedding_size_x2, self._embedding_size_x2)), 
                requires_grad=True)
        
        # 최종 output을 계산하기 전 사용하는 가중치
        self._W = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
                size=(self._embedding_size_x2, self._key_size)), 
                requires_grad=True)
            
    def forward(self, stories):
        """
        build graph for end-to-end memory network

        Args:
            stories: 에피소드별 대화 임베딩 값. shape: (에피소드 개수, mem_size, emb_size)
        """
        # query embedding: self._queries와 self._B를 내적하여 u_0(쿼리 임베딩)를 구함
        u_0 = torch.matmul(self._queries, self._B) # (1, 256)
        u = [u_0]

        # hop size 만큼 반복
        for _ in range(self._hops):
            # key embedding: attention을 수행하기 위해 key 임베딩 벡터를 구하는 과정
            # self._A와 내적하기 위해 stories를 reshape
            m_temp = torch.matmul(torch.reshape(stories, 
                        (-1, self._embedding_size)), self._A)
            
            # 원래 형태로 reshape
            m = torch.reshape(m_temp, 
                        (-1, self._mem_size, self._embedding_size_x2))
            
            # 차원 추가 및 순서 유지
            u_temp = torch.transpose(
                        torch.unsqueeze(u[-1], -1), 2, 1) # (1, 1, 256)
            
            # get attention: 쿼리 임베딩과 키 임베딩 벡터 간의 attention matrix를 계산
            # 쿼리(질문) 임베딩과 키 임베딩 벡터 간의 성분 곱 후 합을 통해
            # 질문과 각 대화 간의 유사도를 계산, softmax를 통해 0 ~ 1 사이 확률 값으로 변경
            dotted = torch.sum(m * u_temp, 2) # (1, 16)
            probs = F.softmax(dotted, 1) # (1, 16)
            probs_temp = torch.transpose(
                            torch.unsqueeze(probs, -1), 2, 1) # (1, 1, 16)
            
            # value embedding: attention matrix와 곱할 value embedding 구하는 과정
            c = torch.matmul(torch.reshape(stories, 
                            (-1, self._embedding_size)), self._C) # (-1, 256)
            c = torch.reshape(c, 
                    (-1, self._mem_size, self._embedding_size_x2)) # (-1, 16, 256)
            c_temp = torch.transpose(c, 2, 1) # (-1, 256, 16)

            # get intermediate result: attention을 반영한 대화 임베딩 값을 구하고, matmul을 통해 중간 결과를 구함
            o_k = torch.sum(c_temp * probs_temp, 2) # (-1, 256)
            u_k = torch.matmul(u[-1], self._H) + o_k # (-1, 256)
            if self._nonlin:
                u_k = self._nonlin(u_k)
            u.append(u_k)

        # get final result: 최종 결과 출력
        req = torch.matmul(u[-1], self._W) # (-1, 300)

        return req
            

class RequirementNet(nn.Module):
    """Requirement Network"""
    def __init__(self, emb_size, key_size, mem_size, meta_size, 
                 hops, name='RequirementNet'):
        """
        initialize and declare variables

        Args:
            emb_size: 패션 아이템의 임베딩 벡터 차원. default to 128
            key_size(뭔지 모름): memory size for the MemN2N. default to 300
            mem_size:
            memory size for the MemN2N. default to 16
            데이터를 얼마나 사용할 지(기억할 지)를 결정하는 hparam.
            차원의 크기라고 생각하면 됨.
            
            meta_size: 패션 아이템 메타데이터의 형태 특징 개수로 보임. default to 4
            hops:
            number of hops in the MemN2n. default to 3
            multiple computational steps을 hops이라고 부름. hop size만큼 layer를 여러 번 쌓음.

            name: 모델 이름
        """
        super().__init__()
        self._name = name
        self._memn2n = MemN2N(emb_size, key_size, 
                              mem_size, meta_size, hops)

    def forward(self, dlg):
        """
        build graph for requirement estimation

        Args:
            dlg: 에피소드별 대화 임베딩 값. shape: (에피소드 개수, mem_size, emb_size)

        Return:
            req: 
        """
        req = self._memn2n(dlg)

        return req
    
