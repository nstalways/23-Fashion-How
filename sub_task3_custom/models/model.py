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

# custom modules
from models.policy import PolicyNet
from models.requirement import RequirementNet


# TODO: docstring 마무리하기
class Model(nn.Module):
    """
    baseline으로 제공된 모델입니다.
    RequirementNet(MemN2N) + PolicyNet으로 구성되어 있습니다.
    """
    def __init__(self, item_size: list,
                 emb_size: int=128, key_size: int=300, mem_size: int=16,
                 meta_size: int=4, hops: int=3,
                 coordi_size: int=4, eval_node: str='[6000,6000,200][2000]',
                 num_rnk: int=3, use_batch_norm: bool=False, use_dropout: bool=True,
                 zero_prob: float=0.5, use_multimodal: bool=False, img_feat_size: int=4096):
        """필요한 변수들을 선언하고, 초기화합니다.

        Args:
            item_size (list): 각 카테고리 별 패션 아이템의 개수가 저장되어 있습니다(패션 아이템은 4개의 카테고리로 구분됩니다).
            emb_size (int, optional): _description_. Defaults to 128.
            key_size (int, optional): MemN2N 출력 벡터의 차원입니다. Defaults to 300.
            mem_size (int, optional): _description_. Defaults to 16.
            meta_size (int, optional): 패션 아이템 메타데이터의 형태 특징 개수입니다(README.pdf 참고). Defaults to 4.
            hops (int, optional): MemN2N 레이어 반복 횟수입니다. Defaults to 3.
            coordi_size (int, optional): 하나의 코디를 구성할 때 필요한 패션 아이템의 개수를 의미합니다. Defaults to 4.
            eval_node (str, optional):
            _mlp_eval 모델을 생성할 때 사용하는 값으로, mlp layer별 노드의 개수를 의미합니다.
            Layer를 깊게 쌓고 싶은 경우, 리스트 내부에 값을 추가하면 됩니다. Defaults to '[6000,6000,200][2000]'.

            num_rnk (int, optional): 추천할 패션 아이템 조합의 개수입니다. Defaults to 3.
            use_batch_norm (bool, optional): PolicyNet에 batch normalization을 적용할 지 선택합니다. Defaults to False.
            use_dropout (bool, optional): PolicyNet에 dropout을 적용할 지 선택합니다. Defaults to True.
            zero_prob (float, optional): Dropout 비율입니다. Defaults to 0.5.
            use_multimodal (bool, optional): 이미지 feature를 활용할 지 선택합니다. Defaults to False.
            img_feat_size (int, optional): 이미지 feature의 차원 크기입니다. Defaults to 4096.
        """
        
        super().__init__()

        # class instance for requirement estimation: 대화문에 대한 feature를 추출하는 모델
        self._requirement = RequirementNet(emb_size, key_size,
                                           mem_size, meta_size, hops)

        # class instance for ranking: 패션 조합에 대한 순위를 산정하는 모델
        self._policy = PolicyNet(item_size, emb_size, key_size,
                                 meta_size, coordi_size, num_rnk,
                                 eval_node, use_batch_norm,
                                 use_dropout, zero_prob,
                                 use_multimodal, img_feat_size)

    def forward(self, dlg, crd) -> tuple:
        """forward 연산을 정의합니다.

        Args:
            dlg (_type_): 대화 임베딩 행렬입니다. shape: (에피소드 개수, mem_size, emb_size)
            crd (_type_): 정답 코디 조합입니다. shape: (에피소드 개수, num_rnk, coordi_size * emb_size * 4)

        Returns:
            tuple: (logits, preds)
        """
        req = self._requirement(dlg)
        logits = self._policy(req, crd)
        preds = torch.argmax(logits, 1)

        return logits, preds


# Test Code
if __name__ == "__main__":
    model = Model(item_size=[100, 100, 100, 100])

    print(model._requirement)
    print(model._policy)

    dlg = torch.randn((100, 16, 128))
    crd = torch.randn((100, 3, 2048))
    
    out = model(dlg, crd)
    print(out[0].shape, out[1].shape)