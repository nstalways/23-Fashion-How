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

# built-in library
import sys
import csv
import re
import os
import json
import random
from itertools import permutations

# external library
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

import codecs
import _pickle as pickle
from ctypes import cdll, create_string_buffer


### added ###
import pdb


def _load_fashion_item(in_file, coordi_size, meta_size):
    """
    function: load fashion item metadata
    각각의 fashion item마다 4개의 metadata를 보유하고 있음.
    고로 fashion item 개수 * 4 == metadata 개수임

    Args:
        in_file_fashion: 패션 아이템 DB. default to './data/mdata.wst.txt.2023.01.26'
        coordi_size: 하나의 코디를 구성하는 패션 아이템의 개수. default to 4
        meta_size: 패션 아이템 메타데이터의 특징 종류 개수. default to 4.

    Return:
        names: 패션 아이템의 이름을 저장해둔 리스트
        metadata:
        패션 아이템의 metadata를 저장해둔 리스트.
        여기서 metadata는 패션 아이템의 형태 특징을 텍스트로 설명한 데이터를 의미함.
    """
    print('loading fashion item metadata')
    with open(in_file, encoding='euc-kr', mode='r') as fin:
        names = []
        metadata = []
        prev_name = ''
        prev_feat = ''
        data = ''
        for l in fin.readlines(): # metadata를 한 줄씩 읽어와서
            line = l.strip() # 불필요한 문자 제거
            w = line.split() # 공백 기준 분리
            name = w[0] # fashion-item의 이름. ex) BL-001(블라우스-001)

            # 각각의 fashion-item마다 여러 개의 metadata가 있기 때문에 prev_name으로 구분
            if name != prev_name: 
                names.append(name)
                prev_name = name

            feat = w[3] # 형태 특징. F(형태), M(소재), C(색채), E(감성) 중 하나
            if feat != prev_feat: # 형태 특징이 달라지고
                if prev_feat != '': # 값이 존재한다면
                    metadata.append(data) # metadata에 data를 추가

                data = w[4] # data를 새롭게 초기화
                for d in w[5:]: # 이후 값들을 하나씩 불러와서
                    data += ' ' + d # 공백을 기준으로 합쳐줌
                prev_feat = feat # prev_feat에 현재 feat을 저장해둠

            else: # 형태 특징이 똑같으면
                for d in w[4:]: # 값들을 불러와서
                    data += ' ' + d # 기존 data에 계속 합쳐줌

        metadata.append(data) # 마지막에 남은 data를 metadata에 추가

        ### 여기서부터는 아직 역할 파악 못함 ###
        for i in range(coordi_size*meta_size):
            metadata.append('')
        # add null types    
        names.append('NONE-OUTER')
        names.append('NONE-TOP')
        names.append('NONE-BOTTOM')
        names.append('NONE-SHOES')

    return names, metadata


def _position_of_fashion_item(item):
    """
    function: get position of fashion items
    패션 아이템이 어떤 카테고리에 속하는지 알아낼 때 사용하는 함수

    Args:
        item: 패션 아이템의 이름(string)
    
    Return:
        idx: 해당 패션 아이템이 몇 번째 카테고리에 속하는지 나타내는 인덱스
    """
    prefix = item[0:2] # item이 "BL-001" 이런 식이기 때문에 하이픈 앞의 상품 카테고리를 prefix로 설정

    # prefix의 값에 따라 idx를 다르게 매김
    # idx를 다르개 매기는 이유는, idx 위치에 fashion item을 대입하기 위해 사용하는 것으로 보임.
    if prefix=='JK' or prefix=='JP' or prefix=='CT' or prefix=='CD' \
        or prefix=='VT' or item=='NONE-OUTER':
        idx = 0 
    elif prefix=='KN' or prefix=='SW' or prefix=='SH' or prefix=='BL' \
        or item=='NONE-TOP':
        idx = 1
    elif prefix=='SK' or prefix=='PT' or prefix=='OP' or item=='NONE-BOTTOM':
        idx = 2
    elif prefix=='SE' or item=='NONE-SHOES':
        idx = 3
    else:
        raise ValueError('{} do not exists.'.format(item))
    
    return idx


def _insert_into_fashion_coordi(coordi, items):
    """
    function: insert new items into previous fashion coordination
    이전 패션 코디 추천에 새로운 아이템을 추가(대체)할 때 사용하는 함수.

    초기에 코디 봇이 추천한 조합이 여러 번의 대화를 거쳐 변경되는 경우가 잦고,
    일부 아이템만 추천하는 경우가 종종 있음(4개가 아닌 2개나 3개를 추천하는 경우)
    이를 대처하기 위해 사용하는 함수로 보임.

    Args:
        coordi: 코디 리스트. e.g., [outer, top, bottom, shoes]
        items: 패션 아이템의 이름들. e.g., [JK-001, BL-001, ...]
    """
    new_coordi = coordi[:]
    for item in items:
        # item에 세미 콜론이나 언더바가 없는데, 아래 세 줄의 코드가 필요한 이유는 모르겠음.
        item = item.split(';')
        new_item = item[len(item)-1].split('_')
        cl_new_item = new_item[len(new_item)-1]

        # cl_new_item: 패션 아이템의 이름.
        # 패션 아이템의 이름을 가지고 해당 아이템이 어떤 카테고리에 속하는지 알아냄
        pos = _position_of_fashion_item(cl_new_item)

        # 예외처리: 원피스의 경우 2번 카테고리에 속하는데, 상/하의를 구분하지 않으므로
        # 상의를 'NONE-TOP'으로 초기화
        if cl_new_item[0:2]=='OP':
            new_coordi[1] = 'NONE-TOP'

        new_coordi[pos] = cl_new_item

    return new_coordi


def _load_trn_dialog(in_file):
    """
    function: load training dialog DB
    모델 학습에 사용할 대화 DB를 불러올 때 사용하는 함수

    Args:
        in_file: default to './data/task1.ddata.wst.txt'

    Return:
        data_utter: DB 속 대화들이 저장된 리스트
        data_coordi: DB 속 추천한 코디들이 저장된 리스트. 
        data_reward_last: DB 속 대화에 대한 TAG들이 저장된 리스트
        np.array(delim_utter, dtype='int32'): episode별로 대화를 나누기 위한 인덱스들을 저장한 리스트
        np.array(delim_coordi, dtype='int32'): episode별로 코디를 나누기 위한 인덱스들을 저장한 리스트
        np.array(delim_reward, dtype='int32'): episode별로 TAG를 나누기 위한 인덱스들을 저장한 리스트

    """
    print('loading dialog DB')
    with open(in_file, encoding='euc-kr', mode='r') as fin:
        data_utter = [] # DB 속 utterances를 모아두는 list
        data_coordi = [] # DB 속 추천한 아이템들을 모아두는 list
        data_reward = [] # DB 속 utterance 별 TAG를 모아두는 list
        delim_utter = [] # episode 별로 대화를 구분하기 위해 사용하는 index list
        delim_coordi = [] # episode 별로 코디를 구분하기 위해 사용하는 index list
        delim_reward = [] # episode 별로 TAG를 구분하기 위해 사용하는 index list
        num_dialog = 0 # DB 속 대화가 몇 개나 있는지 cnt
        num_turn = 1 # ???
        num_coordi = 0 # 추천한 패션 아이템의 개수 cnt
        num_reward = 0 # DB 속 TAG의 숫자를 cnt (왜?)
        is_first = True # DB 속 첫 번째 대화문인지를 판단할 때 사용

        # trn_dialog는 한 줄이 발화 번호 - <CO><US><AC> - 발화 - TAG 형식으로 구성됨
        for l in fin.readlines(): # 한 줄씩 읽어와서
            line = l.strip() # 불필요한 문자 제거 후
            w = line.split() # 공백 기준으로 분리
            ID = w[1] # <CO><US><AC> 중 하나. <CO>-코디봇, <US>-사용자, <AC>-추천한 옷을 의미

            # 아래 코드 블록은 DB에서 새로운 대화가 시작될 때 동작함
            if w[0] == '0': # 발화 번호가 0이고
                if is_first: # DB의 첫 번째 대화문이면
                    is_first = False # 나머지 w[0] == '0'인 대화문들은 첫 번째 대화문이 아니므로 False로 변환

                else: # 첫 번째 대화문이 아니면

                    # 새로운 대화가 시작되면서 이전 대화에서 저장한 값들이 아직 추가되지 않았으므로
                    # 조건에 따라 append를 모두 수행
                    data_utter.append(tot_utter.strip())

                    if prev_ID == '<CO>':
                        data_coordi.append(coordi)
                        num_coordi += 1

                    if prev_ID == '<US>':
                        data_reward.append(tot_func.strip())
                        num_reward += 1
                    
                    # training을 위한 DB를 준비할 때 data_something들을 대화 별로 나누는데,
                    # 그 때 index처럼 사용하려고 만든 리스트들. delim: delimiter(구분자)
                    delim_utter.append(num_turn)
                    delim_coordi.append(num_coordi)
                    delim_reward.append(num_reward)
                    num_turn += 1
                
                prev_ID = ID # 이전 ID를 저장해두고
                tot_utter = '' # 초기화. tot_utter은 하나의 utterance를 저장할 때 사용하는 변수
                tot_func = '' # 초기화. tot_func은 TAG를 저장할 때 사용하는 변수

                # coordination할 카테고리 종류. 초기에는 아무 옷도 추천해주지 않았으므로 'None-카테고리명'으로 초기화
                coordi = ['NONE-OUTER',
                          'NONE-TOP',
                          'NONE-BOTTOM',
                          'NONE-SHOES']
                
                # 대화 개수를 추가(문장이 아닌 대화인 것에 주의!)
                num_dialog += 1

            if ID == '<AC>': # 만약 대화 내용이 <AC>-추천한 옷이면
                items = w[2:] # w의 2번 index부터 끝까지가 옷의 이름이므로, items에 저장

                # 새로 추천한 items을 coordi에 추가함
                coordi = _insert_into_fashion_coordi(coordi, items)
                utter = ''
                continue
            
            # TAG에서 불필요한 문자들을 제거 혹은 공백으로 변환함
            func = re.sub(pattern='[^A-Z_;]', repl='', string=w[-1])
            func = re.sub(pattern='[;]', repl=' ', string=func)

            # 예외 처리
            if func == '_':
                func = ''
            
            # func에 값이 존재하면
            if func != '':
                w = w[:-1] # TAG를 제외한 나머지 데이터를 w에 저장
            
            # ID가 <CO> -> <US> 혹은 <US> -> <CO>로 바뀐 상황이면 (발화자가 변경된 상황)
            if prev_ID != ID:
                # 이전 발화자의 대화문을 data_utter에 저장함
                data_utter.append(tot_utter.strip())

                # 이전 ID가 <CO>-코디봇이었다면
                if prev_ID == '<CO>':
                    # 코디 봇이 추천한 coordi를 data_coordi에 추가
                    data_coordi.append(coordi)
                    num_coordi += 1

                # 이전 ID가 <US>-사용자였다면
                if prev_ID == '<US>':
                    # data_reward에 tot_func(발화문 TAG)를 추가하고, reward 숫자 1 증가
                    data_reward.append(tot_func.strip())
                    num_reward += 1

                # 발화자가 또 변경될 예정이므로 관련 변수들 초기화
                tot_utter = ''
                tot_func = ''
                prev_ID = ID
                num_turn += 1

            # w[2:]는 공백으로 분리되어있는 발화문 리스트를 의미함
            for u in w[2:]:
                tot_utter += ' ' + u # 공백을 기준으로 합쳐진 발화문 string이 완성됨

            tot_func += ' ' + func # 공백을 기준으로 합쳐진 TAG가 완성됨

        # 마지막 발화의 경우 append가 발생하지 않기 때문에 추가
        data_utter.append(tot_utter.strip())                  
        delim_utter.append(num_turn)

        if prev_ID == '<CO>':
            data_coordi.append(coordi)
            num_coordi += 1

        if prev_ID == '<US>':
            data_reward.append(tot_func.strip())
            num_reward += 1

        delim_coordi.append(num_coordi)
        delim_reward.append(num_reward)

        print('# of dialog: {} sets'.format(num_dialog))

        # only use last reward
        # 만약 <CO>, <US>의 발화문이 연속되는 경우 TAG가 누적되는데, 이를 분리하여 마지막 TAG만 사용하겠다는 것
        data_reward_last = []
        for r in data_reward:
            r = r.split()

            if len(r) >= 1:
                data_reward_last.append(r[len(r)-1])    
            else:
                data_reward_last.append('')

        return data_utter, data_coordi, data_reward_last, \
               np.array(delim_utter, dtype='int32'), \
               np.array(delim_coordi, dtype='int32'), \
               np.array(delim_reward, dtype='int32')


def _load_eval_dialog(in_file, num_rank):
    """
    function: load test dialog DB
    모델 성능 평가에 사용할 DB를 불러올 때 사용하는 함수.
    에피소드별로 대화와 코디, rank를 저장함.

    Args:
        in_file: test dialog DB의 경로. default to './data/cl_eval_task1.wst.dev'
        num_rank: 추천할 코디 조합의 개수. default to 3

    Return:
        data_utter: 에피소드별 대화가 저장되어있는 리스트
        data_coordi: 에피소드별 코디가 저장되어있는 리스트
        data_rank: 에피소드별 순위가 저장되어있는 리스트
    """
    print('loading dialog DB')
    with open(in_file, encoding='euc-kr', mode='r') as fin:
        data_utter = []
        data_coordi = []
        num_dialog = 0
        num_utter = 0
        is_first = True

        for line in fin.readlines(): # 한 줄씩 읽어서
            line = line.strip() # 불필요한 문자 제거

            # 평가용 DB 속 대화문의 시작은 세미콜론임
            # 마지막 줄은 ; end임
            if line[0] == ';': # 대화문의 시작 부분이고
                if line[2:5] == 'end': # DB의 마지막이면 탈출
                    break

                if is_first: # DB의 첫 번째 대화라면
                    is_first = False # 이후 대화문들을 다르게 처리하기 위해 False로 변환
                else: # DB의 첫 번째 대화가 아니라면
                    data_utter.append(tot_utter) # 이전 에피소드의 대화들을 저장
                    data_coordi.append(tot_coordi) # 이전 에피소드의 코디들을 저장

                tot_utter = [] # 대화문들을 저장하는 리스트
                tot_coordi = [] # 대화문별 코디를 저장하는 리스트
                num_dialog += 1 # 새로운 대화문이 시작되었으므로 카운팅

            # US: 사용자, CO: 코디봇.
            # 사용자 혹은 코디봇의 대화라면
            elif line[0:2] == 'US' or line[0:2] == 'CO':
                utter = line[2:].strip() 
                tot_utter.append(utter) # 대화문을 utter에 추가 
                num_utter += 1 # 개수 카운팅

            # R: Recommendation. 추천한 상품 조합을 의미
            elif line[0] == 'R':
                coordi = line[2:].strip() # 패션 코디를 가져와서
                new_coordi = ['NONE-OUTER', 
                              'NONE-TOP',  
                              'NONE-BOTTOM', 
                              'NONE-SHOES']
                
                # new_coordi에 가져온 패션 코디를 추가
                new_coordi = _insert_into_fashion_coordi(new_coordi, coordi.split())
                tot_coordi.append(new_coordi)
        
        # 추가되지 않은 마지막 대화문과 코디를 추가
        if not is_first:
            data_utter.append(tot_utter)
            data_coordi.append(tot_coordi)

        # 얘는 아직 역할 모르겠음
        data_rank = []
        rank = 0    
        
        # 모든 에피소드 개수만큼
        for i in range(len(data_coordi)):
            data_rank.append(rank) # 0을 추가. 0 추가의 의미는 아직 모름

        print('# of dialog: {} sets'.format(num_dialog))

        return data_utter, data_coordi, data_rank
        

class SubWordEmbReaderUtil:
    """
    Class for subword embedding    
    """
    def __init__(self, data_path):
        """
        initialize    
        """
        print('\n<Initialize subword embedding>')
        print ('loading=', data_path)
        with open(data_path, 'rb') as fp:
            self._subw_length_min = pickle.load(fp)
            self._subw_length_max = pickle.load(fp)
            self._subw_dic = pickle.load(fp, encoding='euc-kr')
            self._emb_np = pickle.load(fp, encoding='bytes')
            self._emb_size = self._emb_np.shape[1]

    def get_emb_size(self):
        """
        get embedding size    
        """
        return self._emb_size        

    def _normalize_func(self, s):
        """
        normalize
        왜 하는가? -> 이미지를 normalize하는 것처럼, 학습의 안정성과 용이함을 얻기 위해
        text도 정해진 규칙에 맞게 normalize해주는 것.
        """
        s1 = re.sub(' ', '', s) # 공백 없애고
        s1 = re.sub('\n', 'e', s1) # 줄바꿈을 e로 변환한 뒤에
        sl = list(s1) # 음절 단위로 분리

        # b'\xca\xa1'이랑 b'\xfd\xfe'는 'euc-kr'코덱에서 지원하는 한자 범위의 처음과 끝인데,
        # 음절을 인코딩했을 때 이 범위에 있으면 h로 다 바꿔버림(이유는 모름)
        for a in range(len(sl)):
            if sl[a].encode('euc-kr') >= b'\xca\xa1' and \
               sl[a].encode('euc-kr') <= b'\xfd\xfe': sl[a] = 'h'
        s1 = ''.join(sl) # 공백없이 합쳐준 뒤
        return s1 # 반환

    def _word2syllables(self, word):
        """
        word to syllables
        syllable: 음절 -> 발음의 기본 단위.
        단어를 음절로 변환하는 함수
        e.g., 자연어 -> '자', '연', '어'
        """
        syl_list = []

        # codecs 모듈의 'cp949' 코덱을 이용하는 incrementaldecoder를 선언하고
        dec = codecs.lookup('cp949').incrementaldecoder()

        # word를 'euc-kr' 코덱으로 인코딩한 뒤 cp949로 디코딩한 값을 normalize
        # 인코딩/디코딩 코덱을 다르게 설정한 이유는 미스테리임
        w = self._normalize_func(dec.decode(word.encode('euc-kr')))

        # 음절 단위로 다시 분리해서
        for a in list(w):
            # 'euc-kr'로 인코딩 & 디코딩한 뒤에 syl_list에 추가
            syl_list.append(a.encode('euc-kr').decode('euc-kr'))

        return syl_list # 음절 단위로 분리된 리스트를 반환

    def _get_cngram_syllable_wo_dic(self, word, min, max):
        """
        get syllables
        음절 단위의 리스트를 가지고 n-gram 형식의 subword 리스트를 만듦.
        왜 n-gram으로 만드는지는 아직 잘 모르겠음.
        
        예시) 자연어 -> ["<_자", "<_자_연", "<_자_연_어", "자_연", "자_연_어", "자_연_어_>", ...]
        """
        word = word.replace('_', '') # word에서 언더바 제거하고
        p_syl_list = self._word2syllables(word.upper()) # 대문자로 바꾼 뒤에 syllables로 변환
        subword = []
        syl_list = p_syl_list[:]

        # 음절 단위의 리스트 맨 처음과 맨 끝에다가 '<', '>' 추가. 이는 시작과 끝을 나타내기 위해 사용하는 것으로 보임
        syl_list.insert(0, '<')
        syl_list.append('>')
        
        # 원래 음절 + '<' + '>' 개수만큼 반복문을 도는데
        for a in range(len(syl_list)):
            for b in range(min, max+1): # min max 사이에서 b를 가져오고 (현재 데이터의 min, max는 2, 4임)
                if a+b > len(syl_list): break # 이 조건이면 break가 걸리고
                x = syl_list[a:a+b] # 특정 범주만큼의 음절을 가져다가
                k = '_'.join(x) # 언더바로 합치고
                subword.append(k) # subword 리스트에 추가함
        return subword

    def _get_word_emb(self, w):
        """
        do word embedding
        단어 임베딩을 수행하는 함수
        """
        word = w.strip() # 불필요한 문자 제거
        assert len(word) > 0 # 문자가 존재하면

        # cng: 음절 단위의 n-gram.
        cng = self._get_cngram_syllable_wo_dic(word, self._subw_length_min, 
                                               self._subw_length_max)
        
        # self._subw_dic: subword dictionary를 의미하며, ETRI에서 자체 개발한 임베딩 DB에서 불러와 해당 변수에 저장한 것.
        # self._subw_dic은 (cngram, 번호) 형식으로 구성
        # subw가 self._subw_dic에 존재하면 대응되는 번호를 lswi라는 리스트에 저장
        lswi = [self._subw_dic[subw] for subw in cng if subw in self._subw_dic]

        # lswi가 아무것도 없으면 임베딩 DB에 없는 데이터라는 얘기이므로, 'UNK_SUBWORD'에 대응되는 번호를 부여
        if lswi == []: lswi = [self._subw_dic['UNK_SUBWORD']]

        # self._emb_np: ETRI에서 개발한 embedding.
        # (36134, 128) shape을 가지며, np.take()를 사용해서 lswi index의 embedding 값들을 가져온 뒤
        # 0번째 축을 기준으로 모두 더해서 128 차원을 가지는 d를 구함
        d = np.sum(np.take(self._emb_np, lswi, axis=0), axis = 0)
        return d

    def get_sent_emb(self, s):
        """
        do sentence embedding
        문장 embedding을 수행하는 함수
        """
        if s != '': # 문장이 존재한다면
            s = s.strip().split() # 불필요한 문자 제거 후에 공백 기준 분리하고
            semb_tmp = []
            for a in s: # 분리된 토큰? 들을 하나씩 가져와서
                semb_tmp.append(self._get_word_emb(a)) # word embedding을 수행한 뒤 리스트에 추가함

            # 이후 semb_tmp 리스트의 모든 임베딩 값에 대한 평균을 구함
            avg = np.average(semb_tmp, axis=0)
        else:
            avg = np.zeros(self._emb_size)
        return avg


def _vectorize_sent(swer, sent):
    """
    function: vectorize one sentence
    하나의 문장을 벡터로 만들 때 사용
    """
    vec_sent = swer.get_sent_emb(sent) # 문장에 대한 임베딩 값을 구함
    return vec_sent 


def _vectorize_dlg(swer, dialog):
    """
    function: vectorize one dialog
    하나의 대화를 벡터화할 때 사용하는 함수
    """
    vec_dlg = []
    for sent in dialog: # 문장을 가져와서
        sent_emb = _vectorize_sent(swer, sent) # 문장에 대한 임베딩 값
        vec_dlg.append(sent_emb)

    # 모든 문장을 모아 하나의 대화에 대한 임베딩 array로 만듦
    vec_dlg = np.array(vec_dlg, dtype='float32')
    return vec_dlg


def _vectorize(swer, data):
    """
    function: vectorize dialogs
    모든 dialog를 벡터화하는 함수

    Args:
        swer: Subword Embedding 객체
        data: 에피소드별 대화가 저장되어있는 변수

    Return:
        vec: 에피소드별 대화를 임베딩 벡터로 변환해서 저장한 배열
    """
    print('vectorizing data')
    vec = []
    for dlg in data:
        dlg_emb = _vectorize_dlg(swer, dlg)
        vec.append(dlg_emb)

    vec = np.array(vec, dtype=object)

    return vec
    

def _memorize(dialog, mem_size, emb_size):
    """
    function: memorize dialogs for end-to-end memory network
    메모리 네트워크를 위해 dialogs를 기록하는 함수라고 적혀있는데,
    그냥 shape을 동일하게 맞춰주기 위해 패딩을 수행하는 함수라고 생각하면 됨.

    Args:
        dialog: episode 별 dialog의 embedding 값.
        e.g., dialog = np.array([np.array(ep1_embs), np.array(ep2_embs), ...])

        mem_size: memory size for the MemN2N. 역할은 아직 모름. default to 16
        emb_size: embedding DB의 차원이고, 128로 설정되어 있음.
    
    Return:
        np.array(memory, dtype='float32'):
        동일한 shape을 갖는 dialog embedding 값.
        shape: (에피소드 개수, 16, 128)
    """
    print('memorizing data')
    zero_emb = np.zeros((1, emb_size)) # (1, 128)
    memory = []

    # 모든 에피소드의 개수만큼 반복
    for i in range(len(dialog)):
        # 특정 에피소드의 대화 개수에서 mem_size를 빼고, max()를 통해 idx를 세팅
        idx = max(0, len(dialog[i]) - mem_size)

        # 만약 대화 개수가 mem_size보다 큰 경우, mem_size 크기만큼 잘라내고
        # 작은 경우 mem_size 크기에 맞도록 zero padding을 수행함
        ss = dialog[i][idx:]
        pad = mem_size - len(ss)  
        for i in range(pad):
            ss = np.append(ss, zero_emb, axis=0)

        memory.append(ss)

    return np.array(memory, dtype='float32')
    

def _make_ranking_examples(dialog, coordi, reward, item2idx, idx2item, 
                similarities, num_rank, num_perm, num_aug, corr_thres):
    """
    function: make candidates for training    
    학습에 이용할 후보군들을 만들 때 사용하는 augmentation 함수

    Args:
        dialog: episode 단위로 분리된 대화들이 저장된 리스트
        coordi: episode 단위로 분리된 옷 조합들이 저장된 리스트
        reward: episode 단위로 분리된 TAG 정보들이 저장된 리스트
        item2idx: 패션 아이템의 이름으로 idx를 찾을 수 있는 리스트
        idx2item: idx로 패션 아이템의 이름을 찾을 수 있는 리스트
        similarities: 카테고리별 패션 아이템 간의 cos_sim 값들
        num_rank: 랭킹(아마도 추천할 조합의 개수)
        num_perm: 순열 반복 횟수
        num_aug: augmentation 반복 횟수
        corr_thres: correlation threshold

    Return:
        data_dialog: 에피소드별 dialog가 저장된 리스트. e.g., [ep1_dialog, ep2_dialog, ...]
        data_coordi: 에피소드별 추천 코디가 저장된 리스트. e.g., [ep1_coordi, ep2_coordi, ...]
        data_rank: 에피소드별 위치?를 저장해둔 리스트(아직 사용하는 이유는 파악하지 못함)
    """
    print('making ranking_examples')
    data_dialog = []
    data_coordi = []
    data_rank = []
    idx = np.arange(num_rank) # num_rank: 3이므로 idx = np.array([0, 1, 2])
    rank_lst = np.array(list(permutations(idx, num_rank))) # [0, 1, 2]에서 3개로 순열을 만듦
    num_item_in_coordi = len(coordi[0][0]) # 4. 하나의 코디 내에 몇 개의 아이템이 있는지 의미

    # 전체 episode 개수만큼 반복
    for i in range(len(coordi)):
        crd_lst = coordi[i] # i번째 episode의 코디 리스트를 가져옴

        # 역순으로 뒤집는데 이는 episode별 코디 리스트의 초반에는 아무것도 추천하지 않기 때문에 무의미한 값들이 대부분이고,
        # 따라서 유의미한 값들이 많은 뒷부분을 이용하기 위해 뒤집는 것으로 보임
        crd_lst = crd_lst[::-1]
        crd = []
        prev_crd = ['', '', '', '']
        count = 0

        # crd_lst의 코디 개수만큼 반복하는데
        for j in range(len(crd_lst)):
            # 특정 코디가 prev_crd와 다르면서 아무것도 추천하지 않는 코디가 아니라면(새로운 코디라면)
            if crd_lst[j] != prev_crd and crd_lst[j] != \
                ['NONE-OUTER', 'NONE-TOP', 'NONE-BOTTOM', 'NONE-SHOES']:

                # 아래의 코드를 수행
                crd.append(crd_lst[j]) 
                prev_crd = crd_lst[j]
                count += 1

            # num_rank 만큼의 코디가 확보되면 반복문을 탈출
            if count == num_rank:
                break
        
        # 마찬가지로 특정 episode의 reward(=tag)를 가져와서 뒤집어줌.
        rwd_lst = reward[i]
        rwd_lst = rwd_lst[::-1]
        rwd = ''

        # 특이한 점은 reward의 경우 episode의 '마지막 값만' 사용함
        for j in range(len(rwd_lst)):
            if rwd_lst[j] != '':
                rwd = rwd_lst[j]
                break
        
        # count >= num_rank의 의미: num_rank개 만큼의 coordi가 확보되었다는 것
        if count >= num_rank:
            # num_perm 횟수만큼 코디와 랭킹을 shuffle(모델의 일반화 성능을 높이기 위해)
            for j in range(num_perm):
                rank, rand_crd = _shuffle_one_coordi_and_ranking(rank_lst, crd, num_rank) 
                data_rank.append(rank)
                data_dialog.append(dialog[i])
                data_coordi.append(rand_crd)

        # 만약 num_rank개의 코디가 확보되지 않았다면
        # 확보되지 않은 상황: 코디 봇이 동일한 조합을 계속해서 추천하거나, 처음 추천한 조합이 바로 마음에 든 경우를 의미
        # DB를 보면, 코디 봇은 무조건 2번 이상 조합을 추천해주고 있음.
        # (대화를 통해 초기에 추천한 조합 하나, 추천한 조합을 확정짓기 위해 마지막에 추천한 조합 하나)
        # 그래서 count >= (num_rank - 1)로 조건이 설정되어 있음
        elif count >= (num_rank - 1):
            # 하나의 코디 조합 내에 4개의 아이템이 있고, 거기서 2개를 뽑아 4*3=12, 총 12개의 순열을 만듦
            itm_lst = list(permutations(np.arange(num_item_in_coordi), 2)) 

            # 0 ~ 11 까지의 idx를 만들고
            idx = np.arange(len(itm_lst))

            # 무작위로 섞어준 뒤
            np.random.shuffle(idx)

            # 기존 코디 조합의 일부를 대체하여 새로운 코디 조합을 만들고
            crd_new = _replace_item(crd[1], item2idx, idx2item, 
                            similarities, itm_lst[idx[0]], corr_thres)
            
            # 이를 crd에 추가
            crd.append(crd_new)

            # num_perm 횟수만큼 코디와 랭킹을 shuffle(모델의 일반화 성능을 높이기 위해)
            for j in range(num_perm):
                rank, rand_crd = _shuffle_one_coordi_and_ranking(rank_lst, crd, num_rank) 
                data_rank.append(rank)
                data_dialog.append(dialog[i])
                data_coordi.append(rand_crd)

        # rwd는 대화 상태를 나타내는 TAG임
        # 'USER_SUCCESS == 사용자가 기술한 추천 의상 성공'을 의미
        # i번째 에피소드에서 코디봇이 추천한 조합이 성공했다면
        # 성공한 조합을 기반으로 num_aug만큼 데이터를 증강하겠다는 의미
        if 'USER_SUCCESS' in rwd:   
            for m in range(num_aug): # num_aug(증강할 횟수)만큼 반복함
                crd_aug = []
                crd_aug.append(crd[0]) # 성공한 조합을 crd_aug에 추가해주고

                for j in range(1, num_rank):
                    itm_lst = list(permutations(np.arange(num_item_in_coordi), j))
                    idx = np.arange(len(itm_lst))
                    np.random.shuffle(idx)

                    # 성공한 조합을 기반으로 일부 패션 아이템을 replace해서 새로운 조합을 구성한 뒤에
                    crd_new = _replace_item(crd[0], item2idx, idx2item, 
                                    similarities, itm_lst[idx[0]], corr_thres)
                    
                    # 새로운 조합을 crd_aug에 추가
                    crd_aug.append(crd_new)

                # 위 반복문이 끝나면 성공한 조합과 성공한 조합을 기반으로 증강된 조합 2개가 합쳐져
                # crd_aug에 총 3개의 조합이 들어있게됨.
                for j in range(num_perm):
                    # crd_aug를 바로 쓰지 않고, 순서를 한 번 섞어준 뒤에
                    rank, rand_crd = _shuffle_one_coordi_and_ranking(rank_lst, crd_aug, num_rank) 

                    # data_ 리스트에다가 각각을 추가
                    data_rank.append(rank)
                    data_dialog.append(dialog[i])
                    data_coordi.append(rand_crd)

    return data_dialog, data_coordi, data_rank


def _replace_item(crd, item2idx, idx2item, similarities, pos, thres):
    """
    function: replace item using cosine similarities
    cos_sim 값을 기반으로 아이템을 교체하는 함수임.
    cos_sim 값이 낮은 아이템으로 교체하는 이유에 대해서는 이해가 잘 되지 않음
    (cos_sim이 낮다 -> 유사하지 않은 아이템이다 -> 유사하지 않은 아이템으로 교체한다? 왜?)

    crd: 한 개의 패션 조합. 패션 조합은 4개의 패션 아이템으로 구성되어 있음. e.g., [outer, top, bottom, shoes]
    item2idx: 패션 아이템의 이름으로 해당 아이템의 idx를 찾을 수 있는 리스트
    idx2item: 특정 idx로 패션 아이템의 이름을 찾을 수 있는 리스트
    similarities: 패션 아이템들 간의 cos_sim 값이 저장되어 있는 배열.
    pos:
    한 개의 패션 조합은 4개의 패션 아이템으로 구성되어 있고, 각각의 패션 아이템은 서로 다른 카테고리에 속해있음.
    예를 들어, [item1, item2, item3, item4]와 같은 조합이 있다면, item 1, 2, 3, 4는 모두 다른 카테고리에 속해있는 것.
    _replace_item을 호출할 때 4개의 카테고리 중 n개를 뽑아 순열을 만들고, 만든 순열 중에 무작위로 하나를 선택해서 pos로 넘겨줌.
    즉 pos는 crd에서 replace할 패션 아이템의 카테고리 위치를 나타내는 변수임.

    thres: cos_sim에 대한 threshold 값
    """
    new_crd = crd[:] # crd를 new_crd에 복사
    for p in pos: # 카테고리를 가져옴(pos 설명 참고)
        itm = crd[p] # 해당 카테고리의 아이템을 가져옴
        itm_idx = item2idx[p][itm] # item2idx의 p번째 카테고리에서, itm의 이름으로 itm_idx를 검색
        idx = np.arange(len(item2idx[p])) # item2idx의 p번째 카테고리에 속하는 패션 아이템의 개수만큼의 배열을
        np.random.shuffle(idx) # 무작위로 섞어주고

        # p번째 카테고리에 속하는 패션 아이템의 개수만큼 반복하는데
        for k in range(len(item2idx[p])):

            # p번째 카테고리의 cos_sim 배열에서, itm_idx번째 cos_sim 행을 가져오고
            # 무작위로 섞은 idx 배열에서 k번째 idx를 가져옴.
            # 그러면 itm_idx 번째 패션 아이템과, idx[k]번째 패션 아이템 간의 cos_sim(유사도) 값을 가져오게 되는데,
            # 이게 threshold보다 작다면
            if similarities[p][itm_idx][idx[k]] < thres:
                rep_idx = idx[k] # 인덱스를 저장해두고
                rep_itm = idx2item[p][rep_idx] # p번째 카테고리의 idx2item dict에서 rep_idx로 패션 아이템이 이름을 가져옴
                break

        new_crd[p] = rep_itm # new_crd의 p번째 카테고리 아이템을 rep_itm으로 대체

    return new_crd


def _indexing_coordi(data, coordi_size, itm2idx):
    """
    function: fashion item numbering
    각각의 패션 아이템마다 번호를 매기는 함수. 사용 이유는 아직 파악 못함

    Args:
        data: episode별 코디가 담겨있음. 하나의 코디는 4개의 패션 아이템으로 구성.
        e.g., data = [ep1_coordi, ep2_coordi, ...], ep1_coordi = [itm1, itm2, itm3, itm4]

        coordi_size: hparams. default to 4
        itm2idx: 패션 아이템의 이름으로 특정 카테고리 내의 패션 아이템의 위치 index를 찾을 때 사용하는 리스트
    
    Return:
        np.array(vec, dtype='int32'): 패션 아이템의 이름에 대응되는 index들이 저장되어있는 배열
    """
    print('indexing fashion coordi')
    vec = []
    for d in range(len(data)): # 총 에피소드의 개수만큼 반복
        vec_crd = []
        
        # 특정 에피소드의 코디들을 가져옴
        # 하나의 에피소드에는 3개의 코디가 있고, 각 코디는 4개의 패션 아이템을 가지고 있음
        for itm in data[d]:
            # 하나의 코디를 구성하는 4개의 패션 아이템의 이름과 대응되는 index를 ss에 저장
            ss = np.array([itm2idx[j][itm[j]] for j in range(coordi_size)])
            vec_crd.append(ss)

        vec_crd = np.array(vec_crd, dtype='int32')
        vec.append(vec_crd)

    return np.array(vec, dtype='int32')


def _convert_one_coordi_to_metadata(one_coordi, coordi_size, 
                                    metadata, img_feats):
    """
    function: convert fashion coordination to metadata
    하나의 패션 아이템 조합을 metadata로 바꿔주는 함수

    Args:
        one_coordi: 하나의 코디 조합. e.g., [33, 254, 11, 84]
        coordi_size: default to 4
        metadata:
        패션 아이템의 임베딩 값이 저장되어있음.
        4개의 카테고리로 나뉘어져 있으며, 각 카테고리에 속하는 패션 아이템의 임베딩 값은 (1, 512) 형태를 가짐.

        img_feats: use_multimodal 옵션이 True일 때만 값이 존재. default to None.

    Return:
        items: 하나의 패션 아이템 조합에 대한 임베딩 값들.
    """
    if img_feats is None: # img_feats가 없으면
        items = None # items를 초기화해주고
        for j in range(coordi_size): # 4번 반복하는데
            
            # 특정 카테고리의 메타데이터에서 코디 index로 패션 아이템의 임베딩 값을 가져오고
            buf = metadata[j][one_coordi[j]] # (512, )

            # 첫 아이템이라면 items에 추가, 그게 아니면 concat
            if j == 0:
                items = buf[:]
            else:
                items = np.concatenate([items[:], buf[:]], axis=0) # (512 * coordi_size)

    else: # img_feats이 있는 경우(일단은 pass)
        items_meta = None
        items_feat = None

        for j in range(coordi_size):
            buf_meta = metadata[j][one_coordi[j]]
            buf_feat = img_feats[j][one_coordi[j]]

            if j == 0:
                items_meta = buf_meta[:]
                items_feat = buf_feat[:]

            else:
                items_meta = np.concatenate(
                                [items_meta[:], buf_meta[:]], axis=0)
                items_feat += buf_feat[:]
                
        # items_feat /= (float)(coordi_size)
        items_feat /= float(coordi_size)
        items = np.concatenate([items_meta, items_feat], axis=0)

    return items 
    

def _convert_dlg_coordi_to_metadata(dlg_coordi, coordi_size, 
                                    metadata, img_feats):
    """
    function: convert fashion coordinations to metadata

    Args:
        dlg_coordi: 한 개의 에피소드에 있는 코디들이 저장된 리스트. 각 코디는 패션 아이템에 대응되는 index로 이루어져 있음.
        coordi_size: default to 4
        metadata:
        패션 아이템의 임베딩 값이 저장되어있음.
        4개의 카테고리로 나뉘어져 있으며, 각 카테고리에 속하는 패션 아이템의 임베딩 값은 (1, 512) 형태를 가짐.

        img_feats: use_multimodal 옵션이 True일 때만 값이 존재. default to None.

    Return:
        scripts:
        임베딩 값들이 저장된 array.
        scripts.shape: (3, 2048), (코디 개수, 카테고리 개수 * 패션 아이템 임베딩 벡터의 차원)
    """
    # 에피소드의 첫 번째 코디를 임베딩 값으로 변환. shape: (2048, )
    items = _convert_one_coordi_to_metadata(dlg_coordi[0], coordi_size, metadata, img_feats)

    # 에피소드의 첫 번째 코디를 저장
    prev_coordi = dlg_coordi[0][:]

    # 에피소드의 첫 번째 코디에 대한 임베딩 값을 저장
    prev_items = items[:]

    # 차원 하나를 추가. scripts.shape: (1, 4, 512)
    scripts = np.expand_dims(items, axis=0)[:]

    # dlg_coordi.shape: (3, 4)
    for i in range(1, dlg_coordi.shape[0]):
        # 만약에 이전 코디와 현재 코디가 같다면
        if np.array_equal(prev_coordi, dlg_coordi[i]):
            items = prev_items[:] # 이전 코디를 items에 저장하고
        else: # 안 같으면
            # 현재 코디를 임베딩 값으로 변환한 뒤 items에 저장
            items = _convert_one_coordi_to_metadata(dlg_coordi[i], coordi_size, metadata, img_feats)
        
        # 값들을 갱신해주고
        prev_coordi = dlg_coordi[i][:]
        prev_items = items[:]

        # items 차원을 추가한 뒤에 scripts에 concat
        items = np.expand_dims(items, axis=0)
        scripts = np.concatenate([scripts[:], items[:]], axis=0)

    return scripts


def _convert_coordi_to_metadata(coordi, coordi_size, 
                                metadata, img_feats):
    """
    function: convert fashion coordinations to metadata

    Args:
        coordi: 패션 아이템에 대응되는 index가 저장되어있음.
        e.g., coordi = np.array([ep1, ep2, ep3, ...]), ep1 = [coordi1, coordi2, coordi3],
        coordi1 = [33, 204, 12, 99], coordi2 = [66, 593, 21, 33], ...

        coordi_size: default to 4
        metadata:
        패션 아이템의 임베딩 값이 저장되어있음.
        4개의 카테고리로 나뉘어져 있으며, 각 카테고리에 속하는 패션 아이템의 임베딩 값은 (1, 512) 형태를 가짐.

        img_feats: use_multimodal 옵션이 True일 때만 값이 존재. default to None.
    
    Returns:
        np.array(vec, dtype='float32'): 
    """
    print('converting fashion coordi to metadata')
    vec = []
    # 총 에피소드의 개수만큼 반복
    for d in range(len(coordi)):
        # 에피소드마다 코디들을 임베딩 값으로 변환해서 vec_meta에 저장하고
        vec_meta = _convert_dlg_coordi_to_metadata(coordi[d], coordi_size, metadata, img_feats)

        # vec에다가 추가해줌
        vec.append(vec_meta)

    return np.array(vec, dtype='float32')


def _episode_slice(data, delim):
    """
    function: divide by episode
    episode별로 나눈다 == 각 사용자마다 코디 봇과 나눈 대화 뭉탱이 별로 나눈다
    """
    episodes = []
    start = 0

    # delim에 있는 값들은 새로운 대화가 시작될 때마다 저장된 값들로,
    # delim을 사용해서 하나로 합쳐진 data를 episode별로 구분함.
    for end in delim:
        epi = data[start:end]
        episodes.append(epi)
        start = end

    return episodes


def _categorize(name, vec_item, coordi_size):
    """
    function: categorize fashion items
    -> 패션 아이템의 이름(name)과 임베딩(vec_item)을 coordi_size 개의 범주로 묶어주는 함수

    name: 패션 아이템의 이름
    vec_item: 패션 아이템 별 임베딩 값을 모아둔 array
    coordi_size: 4로 설정되어 있음
    """
    slot_item = [] 
    slot_name = []
    for i in range(coordi_size):
        # 4개의 빈 리스트를 추가
        slot_item.append([])
        slot_name.append([])
    
    # 모든 패션 아이템에 대해
    for i in range(len(name)):
        # 사전에 정의한 아이템 범주에 따른 idx 값을 가져다가
        pos = _position_of_fashion_item(name[i])

        # slot_item, slot_name의 pos 위치에 패션 아이템의 이름/임베딩 값을 저장
        slot_item[pos].append(vec_item[i])
        slot_name[pos].append(name[i])
    
    # slot_item의 경우 dtype=object로 설정하여
    # np.array 안에 np.array를 넣을 수 있도록 구성
    slot_item = np.array([np.array(s) for s in slot_item],
                         dtype=object)
    
    return slot_name, slot_item # 4개의 카테고리로 구분된 패션 아이템과 임베딩을 반환


def _shuffle_one_coordi_and_ranking(rank_lst, coordi, num_rank):
    """
    function: shuffle fashion coordinations
    패션 아이템 조합을 섞을 때 사용하는 함수.
    모델의 일반화 성능을 높이기 위해, 다양한 시나리오로 모델을 학습시키려고 사용하는 것 같음.
    (모델이 단순히 순열의 순서를 암기하는 것을 방지하는 것)

    rank_lst: np.array(list(permutations(np.arange(num_rank), num_rank)))
              num_rank 범위 내의 숫자를 가지고 num_rank개의 element를 갖는 순열들을 모아둔 변수
    coordi: num_rank 개 만큼의 코디 리스트
    num_rank: hparams. default to 3
    """
    idx = np.arange(num_rank) # np.array([0, 1, 2])
    np.random.shuffle(idx) # idx를 무작위로 섞어줌
    
    # 총 순열의 개수만큼 반복하는데
    for k in range(len(rank_lst)):
        # idx랑 rank_lst[k]랑 같으면
        if np.array_equal(idx, rank_lst[k]):
            rank = k # rank 변수에 k 값을 저장함
            # num_rank가 3일 때 rank_lst는 [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
            # 순열 기반 시스템에서 rank_lst의 순서는 사용자의 선호도가 반영되었다고 간주
            # 따라서 (0, 1, 2) 순서의 조합이 가장 올바르고, (2, 1, 0) 순서의 조합이 가장 부적절하다고 생각할 수 있음.
            # 다만 순서를 섞지 않고 모델을 학습시키는 경우 순서만 암기할 수도 있음
            # 이를 방지하기 위해 원래 코디 조합의 순서를 섞어주고, 대응되는 rank를 정답으로 사용.
            break

    rand_crd = []
    for k in range(num_rank): # num_rank만큼 반복하는데
        # idx[k]에 해당하는 코디를 rand_crd에 추가.
        # rand_crd는 무작위로 섞은 idx에 맞게 coordi를 재배열한 것(dataloader에서 shuffle하는 것과 비슷한 느낌인가?)
        rand_crd.append(coordi[idx[k]])

    return rank, rand_crd


def shuffle_coordi_and_ranking(coordi, num_rank):
    """
    function: shuffle fashion coordinations

    Args:
        coordi:
        에피소드별로 구분된 평가용 코디 임베딩.
        shape: (전체 에피소드 개수, 3, 2048)

        num_rank:
        추천할 조합의 개수. default to 3.

    Return:
        data_coordi_rand:
        data_rank:
    """
    data_rank = []
    data_coordi_rand = []

    idx = np.arange(num_rank)
    rank_lst = np.array(list(permutations(idx, num_rank)))

    # 전체 에피소드 개수만큼 반복
    for i in range(len(coordi)):
        # idx를 무작위로 섞어주고
        idx = np.arange(num_rank)
        np.random.shuffle(idx)

        # 무작위로 섞은 idx와 동일한 순열의 rank_lst 위치를 저장 -> 성능 평가를 위함
        for k in range(len(rank_lst)):
            if np.array_equal(idx, rank_lst[k]):
                rank = k
                break
        
        # 성능 평가를 위해 data_rank에 추가해두고
        data_rank.append(rank)

        # 원래 코디 조합의 순서를 무작위로 섞은 idx 순서로 변경
        coordi_rand = []
        crd = coordi[i]
        for k in range(num_rank):
            coordi_rand.append(crd[idx[k]])
        
        # 섞은 코디를 추가
        data_coordi_rand.append(coordi_rand)

    data_coordi_rand = np.array(data_coordi_rand, dtype='float32')    
    data_rank = np.array(data_rank, dtype='int32')

    return data_coordi_rand, data_rank


def _load_fashion_feature(file_name, slot_name, coordi_size, feat_size):
    """
    function: load image features
    """
    with open(file_name, 'r') as fin:
        data = json.load(fin)          
        suffix = '.jpg'
        feats = []
        for i in range(coordi_size):
            feat = []    
            for n in slot_name[i]:
                if n[0:4] == 'NONE':
                    feat.append(np.zeros((feat_size)))
                else:
                    img_name = n + suffix
                    feat.append(np.mean(np.array(data[img_name]), 
                                        axis=0))
            feats.append(np.array(feat))
        feats = np.array(feats, dtype=object)            
        return feats


def make_metadata(in_file_fashion, swer, coordi_size, meta_size,
                  use_multimodal, in_file_img_feats, feat_size):
    """
    function: make metadata for training and test
    training/test에 맞게 metadata를 만들때 사용하는 함수

    Args:
        in_file_fashion: 패션 아이템 DB. default to './data/mdata.wst.txt.2023.01.26'
        swer: Subword Embedding 객체
        coordi_size: 하나의 코디를 구성하는 패션 아이템의 개수. default to 4
        meta_size: 패션 아이템 메타데이터의 특징 종류 개수. default to 4.
        use_multimodal: multimoda input 옵션. default to False.
        in_file_img_feats: 패션 아이템의 이미지 피처. use_multimodal option이 True일 때만 사용됨.
        feat_size: 이미지 피쳐의 크기. default to 4096.

    Return:
        slot_item: 4개의 카테고리로 묶여진 패션 아이템의 임베딩 값들.
        idx2item: idx를 기반으로 패션 아이템 이름을 검색할 수 있음. List[dict] 형태.
        item2idx: 패션 아이템 이름을 기반으로 idx를 검색할 수 있음. List[dict] 형태.
        item_size: 각 카테고리마다 몇 개의 패션 아이템이 있는지에 대한 정보가 담겨있음.
        vec_similarities: 패션 아이템 간의 cos_sim 값을 계산한 배열.
        slot_feat: (아직 파악 못함)
    """
    ### added ###
    # pdb.set_trace()
    #############
    print('\n<Make metadata>')
    if not os.path.exists(in_file_fashion):
        raise ValueError('{} do not exists.'.format(in_file_fashion))
    
    # load metadata DB: fashion item metadata를 불러옴
    name, data_item = _load_fashion_item(in_file_fashion, 
                                         coordi_size, meta_size)
    
    # 불러온 메타데이터(텍스트)를 벡터(수치)로 변환
    print('vectorizing data')
    emb_size = swer.get_emb_size() # 128
    
    # embedding: data_item 리스트를 가지고 embedding 값을 구하는 과정
    vec_item = _vectorize_dlg(swer, data_item)

    # 하나의 item에 4개의 형태 특징이 있으므로, 4*emb_size 차원을 가지도록 resize를 해주면
    # (1, 512) 벡터 하나가 하나의 패션 아이템에 대한 임베딩을 의미하게 됨
    vec_item = vec_item.reshape((-1, meta_size*emb_size))

    # categorize fashion items: fashion item 관련 data들을 coordi_size 개수의 카테고리로 나눠서 묶어주는 과정
    slot_name, slot_item = _categorize(name, vec_item, coordi_size)
    slot_feat = None
    if use_multimodal: # use_multimodal은 default가 False이므로 일단 넘어감
        slot_feat = _load_fashion_feature(in_file_img_feats, 
                                    slot_name, coordi_size, feat_size)
    vec_similarities = []

    # calculation cosine similarities
    for i in range(coordi_size):
        # 특정 카테고리의 벡터들을 csr_matrix로 변환함
        # sparse matrix(값이 0인 elements가 많은 matrix를 의미)의 연산 효율성을 높이기 위해 사용한다고 생각
        item_sparse = sparse.csr_matrix(slot_item[i])

        # 같은 카테고리 내의 fashion item embedding끼리 cosine similarity를 계산
        # similarities의 shape은 해당 카테고리에 속하는 아이템의 개수로 결정됨. if 개수==n: shape == (n, n)
        similarities = cosine_similarity(item_sparse)
        vec_similarities.append(similarities)
    vec_similarities = np.array(vec_similarities, dtype=object)

    idx2item = []
    item2idx = []
    item_size = []
    for i in range(coordi_size):
        # idx를 통해 item에 접근할 수 있도록, 특정 카테고리의 패션 아이템마다 key를 부여
        idx2item.append(dict((j, m) for j, m in enumerate(slot_name[i])))

        # item을 통해 idx에 접근할 수 있도록...
        item2idx.append(dict((m, j) for j, m in enumerate(slot_name[i])))

        # 특정 카테고리에 속하는 패션 아이템의 개수를 저장
        item_size.append(len(slot_name[i]))
        
    return slot_item, idx2item, item2idx, item_size, \
           vec_similarities, slot_feat


def make_io_data(mode, in_file_dialog, swer, mem_size, coordi_size,
                 item2idx, idx2item, metadata, similarities, num_rank, 
                 num_perm=1, num_aug=1, corr_thres=1.0, img_feats=None):
    """
    function: prepare DB for training and test

    Args:
        mode: 'prepare', 'eval' 모드가 존재하며, train/validation, test DB 중 하나를 선택하는 역할.
        in_file_dialog: 학습에 사용할 대화 DB. default to ./data/task1.ddata.wst.txt
        swer: subword embedding 객체
        mem_size: memory size for the MemN2N. MemN2N 네트워크의 메모리 크기로, 모델 구조를 뜯어보면 이해할 수 있음
        coordi_size: 4로 고정되어 있음(아우터, 상의, 하의, 신발로 이루어진 4개의 카테고리)
        item2idx: fashion item으로 idx를 찾을 때 사용하는 변수
        idx2item: idx로 fashion item을 찾을 때 사용하는 변수
        metadata: fashion item의 embedding 값들이 담겨있음
        similarities: 카테고리별 아이템들 간에 연산한 cosine similarity 값들이 담겨있음
        num_rank: 추천할 코디 조합의 개수. default to 3(ArgParser의 값이 기본 3, 함수 자체는 1)
        num_perm: 순열 반복 횟수. 자세한 내용은 다른 함수 참고. default to 3
        num_aug: data augmentation 횟수. defaults to 3
        corr_thresh: correlation threshold. 데이터 증강 시 cos_sim 값에 대한 기준치로 사용됨. default to 0.7
        img_feats: use_multimodal 옵션이 False이면 None이 담겨있고, True면 다른 값(어떤 값인지는 아직 모름)이 들어감.
    
    Returns:
        mem_dialog:
        MemN2N에 사용할 대화 임베딩 값들.
        shape: (num_episodes, mem_size, emb_size)
        
        vec_coordi:
        모든 코디들에 대한 임베딩 값이 담겨있는 벡터.
        shape: (num_episodes, num_rnk, emb_size * 4)
        
        data_rank:
        에피소드별 위치를 저장해 둔 리스트인데, 모델 구조나 학습 원리까지 알아야 이해할 수 있을 듯.
        shape: (num_episodes, )
    """
    print('\n<Make input & output data>')
    
    if not os.path.exists(in_file_dialog):
        raise ValueError('{} do not exists.'.format(in_file_dialog))
    
    if mode == 'prepare':
        # load training dialog DB: 학습 대화 DB를 불러오고 
        dialog, coordi, reward, delim_dlg, delim_crd, delim_rwd = \
                                             _load_trn_dialog(in_file_dialog)
        
        # per episode: 불러온 대화 DB를 episode 별로 분리
        dialog = _episode_slice(dialog, delim_dlg)
        coordi = _episode_slice(coordi, delim_crd)
        reward = _episode_slice(reward, delim_rwd)

        # prepare DB for evaluation: 학습에 사용할 DB 생성
        data_dialog, data_coordi, data_rank = \
                    _make_ranking_examples(dialog, coordi, reward, item2idx, 
                                           idx2item, similarities, num_rank, 
                                           num_perm, num_aug, corr_thres)
    
    elif mode == 'eval':
        # load test dialog DB: 평가에 사용할 DB를 불러옴.
        data_dialog, data_coordi, data_rank = \
                                    _load_eval_dialog(in_file_dialog, num_rank)


    # 아직 역할을 잘 모르겠음.
    data_rank = np.array(data_rank, dtype='int32')
    
    # embedding: episode 단위로 잘린 data_dialog를 embedding 벡터로 변환.
    # data_dialog 형태: [[0번째 에피소드의 dialog], [1번째 에피소드의 dialog], ...]
    vec_dialog = _vectorize(swer, data_dialog)
    emb_size = swer.get_emb_size() # 128

    # memorize for end-to-end memory network:
    # MemN2N 모델 학습에 사용할 수 있게 vec_dialog의 shape을 맞춰주는 과정
    mem_dialog = _memorize(vec_dialog, mem_size, emb_size)

    # fashion item numbering: 코디 속 패션 아이템의 이름들을 index로 바꾸는 과정
    idx_coordi = _indexing_coordi(data_coordi, coordi_size, item2idx)

    # convert fashion item to metadata: 패션 아이템 index를 임베딩 값으로 바꾸는 과정
    vec_coordi = _convert_coordi_to_metadata(idx_coordi, coordi_size, 
                                             metadata, img_feats)
    
    return mem_dialog, vec_coordi, data_rank
