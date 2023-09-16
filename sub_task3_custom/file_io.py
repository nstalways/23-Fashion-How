# built-in library
import os
import re
import time
import codecs
import pickle
from typing import List
from itertools import permutations

# external-library
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class SubWordEmbReaderUtil:
    """
    Class for subword embedding
    SubWordEmbedding을 읽어오고, 재조립할 때 사용합니다.
    """
    def __init__(self, data_path: str='./sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat'):
        """

        Args:
            data_path (str, optional):
            Subword에 대한 임베딩 벡터들이 저장되어있는 파일 경로입니다.
            Defaults to './sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat'.
        """
        print('\n<Initialize subword embedding>')
        print('loading=', data_path)
        
        with open(data_path, 'rb') as fp:
            self._subw_length_min = pickle.load(fp)
            self._subw_length_max = pickle.load(fp)
            self._subw_dic = pickle.load(fp, encoding='euc-kr')
            self._emb_np = pickle.load(fp, encoding='bytes')
            self._emb_size = self._emb_np.shape[1]

    def get_emb_size(self) -> int:
        """임베딩 벡터의 크기를 반환합니다.
        128로 고정되어 있습니다.
        """
        return self._emb_size
    
    def _normalize_func(self, s):
        """단어 단위의 텍스트 메타데이터를 normalize할 때 사용합니다.
        이미지에서 normalize의 역할은 outlier로 인한 학습의 불안정성을 해소하는 것인데,
        텍스트 역시 outlier가 있을 수 있으니 동일하게 normalize한다고 생각하면 편합니다.

        Args:
            s (_type_): 단어 단위의 텍스트 메타데이터
        """
        s1 = re.sub(' ', '', s)
        s1 = re.sub('\n', 'e', s1)
        s1 = list(s1)

        for a in range(len(s1)):
            if s1[a].encode('euc-kr') >= b'\xca\xa1' and \
                s1[a].encode('euc-kr') <= b'\xfd\xfe': s1[a] = 'h'
        
        s1 = ''.join(s1) # normalized word

        return s1

    def _word2syllables(self, word):
        """단어를 syllables(음절)로 변환할 때 사용하는 함수입니다.
        e.g., 자연어 -> '자', '연', '어'

        Args:
            word (_type_): _description_
        """
        syl_list = []

        dec = codecs.lookup('cp949').incrementaldecoder()
        w = self._normalize_func(dec.decode(word.encode('euc-kr')))

        for a in list(w):
            syl_list.append(a.encode('euc-kr').decode('euc-kr'))

        return syl_list

    def _get_cngram_syllable_wo_dic(self, word, min, max):
        """n개의 음절을 가져다가 subword를 만들 때 사용하는 함수입니다.
        e.g., 자연어 -> ["<_자", "<_자_연", "<_자_연_어", "자_연", "자_연_어", "자_연_어_>", ...]

        Args:
            word (_type_): 단어 단위의 텍스트 메타데이터입니다.
            min (_type_): subword 최소 길이입니다.
            max (_type_): subword 최대 길이입니다.
        """
        word = word.replace('_', '')
        p_syl_list = self._word2syllables(word.upper())

        subword = []
        syl_list = p_syl_list[:]

        syl_list.insert(0, '<')
        syl_list.append('>')

        for a in range(len(syl_list)):
            for b in range(min, max + 1):
                if a + b > len(syl_list):
                    break

                x = syl_list[a:a + b]
                k = '_'.join(x)
                subword.append(k)

        return subword

    def _get_word_emb(self, w):
        """단어 단위의 임베딩을 수행합니다.

        Args:
            w (_type_): 단어 단위의 텍스트 메타데이터입니다.
        """
        word = w.strip()
        assert len(word) > 0

        cng = self._get_cngram_syllable_wo_dic(word, self._subw_length_min, self._subw_length_max)

        # ETRI 자체개발 임베딩에서 subw를 검색
        lswi = [self._subw_dic[subw] for subw in cng if subw in self._subw_dic]

        if lswi == []:
            lswi = [self._subw_dic['UNK_SUBWORD']] # 없으면 이거 추가

        # subword 임베딩 array에서 lswi 인덱스에 해당하는 임베딩 값들을 가져와서 다 더해줌
        d = np.sum(np.take(self._emb_np, lswi, axis=0), axis=0)

        return d

    def get_sent_emb(self, s: str):
        """문장 단위의 텍스트 메타데이터에 대한 벡터를 가져옵니다.

        Args:
            s (str): 문장 단위의 텍스트 메타데이터입니다.
        """
        if s != '':
            # 문장을 공백 단위로 다시 잘라서
            s = s.strip().split()
            s_emb_tmp = []
            
            # 단어 단위로 임베딩을 수행
            for a in s:
                s_emb_tmp.append(self._get_word_emb(a))

            # 단어 단위 임베딩을 평균내서 문장에 대한 임베딩으로 사용
            avg = np.average(s_emb_tmp, axis=0)
        else:
            avg = np.zeros(self._emb_size)

        return avg


def _vectorize_sent(swer, sent):
    """_summary_

    Args:
        swer (_type_): 벡터 변환에 사용하는 객체입니다.
        sent (_type_): 텍스트 메타데이터입니다.
    """
    vec_sent = swer.get_sent_emb(sent)
    
    return vec_sent


def _vectorize_dlg(swer, dialog):
    """텍스트 메타데이터를 벡터로 변환할 때 사용합니다.

    Args:
        swer (_type_): 벡터 변환에 사용하는 객체입니다.
        dialog (_type_): 텍스트 메타데이터입니다.
    """
    vec_dlg = []
    
    for sent in dialog:
        sent_emb = _vectorize_sent(swer, sent) # 텍스트 메타데이터 -> 벡터 변환
        vec_dlg.append(sent_emb)
    
    vec_dlg = np.array(vec_dlg, dtype='float32')

    return vec_dlg


def _vectorize(swer, data: List[List[str]]):
    """data 속 텍스트 데이터들을 모두 벡터로 변환하는 함수입니다.

    Args:
        swer (_type_): SubWordEmb 객체입니다.
        data (List[List[str]]): 에피소드별 대화문이 저장되어 있습니다.
    """
    print('vectorizing data...')

    vec = []

    for dlg in data:
        dlg_emb = _vectorize_dlg(swer, dlg)
        vec.append(dlg_emb)

    vec = np.array(vec, dtype=object)

    return vec


# TODO
def _memorize(dialog: np.ndarray, mem_size: int=16, emb_size: int=128):
    """벡터로 변환된 대화문에서 mem_size 만큼의 대화문을 추출합니다.

    Args:
        dialog (np.ndarray): 벡터로 변환된 대화문입니다.
        mem_size (int, optional): _description_. Defaults to 16.
        emb_size (int, optional): subword 임베딩 벡터의 차원을 의미합니다. Defaults to 128.
    """
    print('memorizing data...')

    zero_emb = np.zeros((1, emb_size))
    memory = []

    # # 전체 에피소드에 대해 반복
    # for i in range(len(dialog)):
    #     idx = max(0, len(dialog[i]) - mem_size)

    #     ss = dialog[i][idx:] # mem_size 만큼의 벡터를 추출

    #     # mem_size만큼의 벡터가 확보되지 않은 경우, zero padding 수행
    #     # TODO: 텍스트에서 zero padding 말고 다른 padding 방법이 없는지 찾아보기
    #     pad = mem_size - len(ss)

    #     for i in range(pad):
    #         ss = np.append(ss, zero_emb, axis=0)

    #     memory.append(ss)

    ### custom code ###
    for i in range(len(dialog)):
        idx = max(0, len(dialog[i]) - mem_size)    

        if idx == 0:
            ss = dialog[i][idx:]
        else:
            # validation data의 dialog와 비슷하게 구성
            # 초반 두 문장과 끝 부분의 유의미한 문장 일부를 붙여서 데이터를 구성
            ss = np.concatenate([dialog[i][:2], dialog[i][-mem_size:-2]], axis=0)

        # 부족하면 padding
        pad = mem_size - len(ss)

        for i in range(pad):
            ss = np.append(ss, zero_emb, axis=0)

        memory.append(ss)
    
    ### custom code ###

    return np.array(memory, dtype='float32')


def _make_one_coordis_to_all_ranking(rank_lst: np.ndarray, coordi: List[List[str]], num_rank: int=3):
    """주어진 coordi로 추천 가능한 모든 순서를 만드는 함수입니다.

    shuffle로 인해 특정 순서의 조합만 데이터에 포함되는 것을 방지합니다.

    총 factorial(num_rank) 만큼의 데이터를 만듭니다.

    Args:
        rank_lst (np.ndarray): 순위에 따라 순열들이 저장되어 있습니다.
        coordi (List[List[str]]): 하나의 에피소드에서 추천된 num_rank 개수의 코디입니다.
        num_rank (int, optional): 추천할 패션 아이템 조합의 개수입니다. Defaults to 3.
    """
    all_ranks = []
    all_coordis = []

    for rank, perm in enumerate(rank_lst):
        all_ranks.append(rank)
        
        tmp_coordi = []
        for order in perm:
            tmp_coordi.append(coordi[order])

        all_coordis.append(tmp_coordi)

    return all_ranks, all_coordis


# TODO
def _make_ranking_examples(dialog: List[List[str]], coordi: List[List[List[str]]], reward: List[List[str]],
                           item2idx: List[dict], idx2item: List[dict], similarities: np.ndarray,
                           num_rank: int=1, num_perm: int=1, num_aug: int=1, corr_thres: float=1.0):
    """순위 모델을 학습할 때 사용하는 데이터를 만듭니다.

    Args:
        dialog (List[List[str]]): 에피소드별로 분리된 대화들이 저장되어 있습니다.
        coordi (List[List[List[str]]]): 에피소드별로 분리된 코디들이 저장되어 있습니다.
        reward (List[List[str]]): 에피소드별로 분리된 TAG 정보들이 저장되어 있습니다.
        item2idx (List[dict]): 패션 아이템의 이름마다 idx가 부여되어 있습니다.
        idx2item (List[dict]): idx마다 패션 아이템의 이름이 부여되어 있습니다.
        similarities (np.ndarray): 패션 아이템 간의 cosine similarity를 계산한 결과입니다.
        num_rank (int, optional): 추천할 패션 아이템 조합의 개수입니다. Defaults to 1.
        num_perm (int, optional): _description_. Defaults to 1.
        num_aug (int, optional): 데이터 증강 횟수입니다. Defaults to 1.
        corr_thres (float, optional): 데이터 증강 시 cos_sim 값에 대한 threshold로 사용합니다. Defaults to 1.0.
    """
    print('making ranking examples...')

    data_dialog = []
    data_coordi = []
    data_rank = []

    idx = np.arange(num_rank) # np.array([0, 1, 2])

    rank_lst = np.array(list(permutations(idx, num_rank)))
    num_item_in_coordi = len(coordi[0][0]) # 4

    for i in range(len(coordi)):
        crd_lst = coordi[i] # i번째 에피소드의 코디 리스트를 가져옴
        crd_lst = crd_lst[::-1] # 유의미한 코디가 존재하는 뒷부분을 먼저 확인하도록 뒤집음

        crd = []
        prev_crd = ['', '', '', '']
        count = 0

        # 한 에피소드의 전체 코디를 확인
        for j in range(len(crd_lst)):
            # 새로운 코디 && 실제 아이템을 추천한 코디라면
            if crd_lst[j] != prev_crd and crd_lst[j] != \
                ['NONE-OUTER', 'NONE-TOP', 'NONE-BOTTOM', 'NONE-SHOES']:
                crd.append(crd_lst[j])
                prev_crd = crd_lst[j]
                count += 1
            
            # 최대 추천 개수만큼 코디를 확보하면 종료
            if count == num_rank:
                break
        
        # i번째 에피소드의 태그를 가져와서 뒤집음
        rwd_lst = reward[i]
        rwd_lst = rwd_lst[::-1]
        rwd = ''

        # 한 에피소드의 전체 태그 중 값이 존재하는 태그 하나만 저장
        for j in range(len(rwd_lst)):
            if rwd_lst[j] != '':
                rwd = rwd_lst[j]
                break
        
        # num_rank 만큼의 조합이 확보된 경우
        if count >= num_rank:
            if num_perm == 0:
                # 하나의 조합으로 모든 순열을 만든 뒤 데이터에 추가합니다.
                all_ranks, all_crds = _make_one_coordis_to_all_ranking(rank_lst, crd, num_rank)

                data_rank.extend(all_ranks)
                data_dialog.extend([dialog[i] for _ in range(len(all_ranks))])
                data_coordi.extend(all_crds)

            else:
                for j in range(num_perm):
                    ### custom code ###
                    # 섞지 않은 코디 또한 학습 데이터에 포함시킴
                    if j == 0:
                        data_rank.append(0)
                        data_dialog.append(dialog[i])
                        data_coordi.append(crd)

                        continue
                    ### custom code ###
                    
                    # shuffle한 코디 또한 학습 데이터에 포함시킴
                    rank, rand_crd = _shuffle_one_coordi_and_ranking(rank_lst, crd, num_rank)
                    
                    data_rank.append(rank)
                    data_dialog.append(dialog[i])
                    data_coordi.append(rand_crd)

        # 확보되지 않은 경우
        elif count >= (num_rank - 1):
            # 조합을 확보하기 위해 가지고 있는 코디의 일부 아이템을 대체하여 새로운 코디를 생성
            itm_lst = list(permutations(np.arange(num_item_in_coordi), 2))
            idx = np.arange(len(itm_lst))
            np.random.shuffle(idx)

            crd_new = _replace_item(crd[1], item2idx, idx2item,
                                    similarities, itm_lst[idx[0]], corr_thres)
            crd.append(crd_new)

            if num_perm == 0:
                # 하나의 조합으로 모든 순열을 만든 뒤 데이터에 추가합니다.
                all_ranks, all_crds = _make_one_coordis_to_all_ranking(rank_lst, crd, num_rank)

                data_rank.extend(all_ranks)
                data_dialog.extend([dialog[i] for _ in range(len(all_ranks))])
                data_coordi.extend(all_crds)

            else:
                for j in range(num_perm):
                    ### custom code ###
                    # 섞지 않은 코디 또한 학습 데이터에 포함시킴
                    if j == 0:
                        data_rank.append(0)
                        data_dialog.append(dialog[i])
                        data_coordi.append(crd)

                        continue
                    ### custom code ###
                    
                    # shuffle한 코디 또한 학습 데이터에 포함시킴
                    rank, rand_crd = _shuffle_one_coordi_and_ranking(rank_lst, crd, num_rank)
                    
                    data_rank.append(rank)
                    data_dialog.append(dialog[i])
                    data_coordi.append(rand_crd)

        
        # 데이터 증강
        if 'USER_SUCCESS' in rwd:
            for m in range(num_aug):
                crd_aug = []
                crd_aug.append(crd[0])

                for j in range(1, num_rank):
                    # 성공한 조합에서 j개의 아이템을 교체, 새로운 조합을 생성 및 추가
                    itm_lst = list(permutations(np.arange(num_item_in_coordi), j))
                    idx = np.arange(len(itm_lst))
                    np.random.shuffle(idx)

                    crd_new = _replace_item(crd[0], item2idx, idx2item,
                                            similarities, itm_lst[idx[0]], corr_thres)
                    crd_aug.append(crd_new)

                if num_perm == 0:
                    # 하나의 조합으로 모든 순열을 만든 뒤 데이터에 추가합니다.
                    all_ranks, all_crds = _make_one_coordis_to_all_ranking(rank_lst, crd_aug, num_rank)

                    data_rank.extend(all_ranks)
                    data_dialog.extend([dialog[i] for _ in range(len(all_ranks))])
                    data_coordi.extend(all_crds)

                else:
                    for j in range(num_perm):
                        ### custom code ###
                        # 섞지 않은 코디 또한 학습 데이터에 포함시킴
                        if j == 0:
                            data_rank.append(0)
                            data_dialog.append(dialog[i])
                            data_coordi.append(crd_aug)

                            continue
                        ### custom code ###

                        # shuffle한 코디 또한 학습 데이터에 포함시킴
                        rank, rand_crd = _shuffle_one_coordi_and_ranking(rank_lst, crd_aug, num_rank)

                        data_rank.append(rank)
                        data_dialog.append(dialog[i])
                        data_coordi.append(rand_crd)
    
    return data_dialog, data_coordi, data_rank
            

def _replace_item(crd: List[List[str]], item2idx: List[dict], idx2item: List[dict],
                  similarities: np.ndarray, pos: tuple, thres: float=1.0):
    """입력된 crd에서 pos 위치의 아이템들을 similarities & thres에 기반하여 교체합니다.

    Args:
        crd (List[List[str]]): 특정 에피소드에서 추천한 조합들이 저장되어 있습니다.
        item2idx (List[dict]): 패션 아이템의 이름마다 idx가 부여되어 있습니다.
        idx2item (List[dict]): idx마다 패션 아이템의 이름이 부여되어 있습니다.
        similarities (np.ndarray): 패션 아이템 간의 cosine similarity를 계산한 결과입니다.
        pos (tuple): 교체할 패션 아이템이 속한 카테고리의 위치를 나타냅니다.
        thres (float, optional): 데이터 증강 시 cos_sim 값에 대한 threshold로 사용합니다. Defaults to 1.0.
    """
    new_crd = crd[:]

    for p in pos:
        # 교체할 아이템의 인덱스를 검색
        itm = crd[p]
        itm_idx = item2idx[p][itm]

        # 교체할 아이템이 속해있는 카테고리의 아이템 인덱스를 shuffle
        idx = np.arange(len(item2idx[p]))
        np.random.shuffle(idx)

        ### custom code ###
        is_found = False

        # p번째 카테고리에 속하는 패션 아이템의 개수만큼 반복
        for k in range(len(item2idx[p])):
            ### original code ###
            # following codes will removed ...
            # # itm_idx 아이템과 idx[k] 아이템의 유사도가 thres보다 낮다면
            # if similarities[p][itm_idx][idx[k]] < thres:

            #     # idx[k]의 위치와 아이템 정보를 저장
            #     rep_idx = idx[k]
            #     rep_itm = idx2item[p][rep_idx]

            #     break
            ### original code ###

            candidate = similarities[p][itm_idx][idx[k]] # 대체할 후보 아이템의 유사도

            # 유사도가 일정 값 이내에 속한다면 대체
            if candidate > 0.5 * thres and candidate < thres:
                rep_idx = idx[k]
                rep_itm = idx2item[p][rep_idx]

                is_found = True
                break
        
        if is_found:
            new_crd[p] = rep_itm # idx[k] 아이템으로 대체

        else:
            rep_idx = idx[0]
            rep_itm = idx2item[p][rep_idx]

            new_crd[p] = rep_itm

        ### custom code ###

    return new_crd


def _indexing_coordi(data: List[List[List[str]]], item2idx: List[dict], coordi_size: int=4):
    """모든 코디에 대해, 코디를 구성하는 패션 아이템의 이름을 대응되는 인덱스로 변환합니다.

    Args:
        data (List[List[List[str]]]): 에피소드별 추천 코디가 저장되어 있습니다.
        item2idx (List[dict]): 패션 아이템의 이름마다 idx가 부여되어 있습니다.
        coordi_size (int, optional): 하나의 코디를 구성하는 패션 아이템의 개수입니다. Defaults to 4.
    """
    print('indexing fashion coordi...')

    vec = []

    # 전체 에피소드에 대해 반복
    for d in range(len(data)):
        vec_crd = []

        # 특정 에피소드 내의 코디에 대해 반복
        for itm in data[d]:
            ss = np.array([item2idx[j][itm[j]] for j in range(coordi_size)]) # 한 개의 조합을 이루는 아이템 별 인덱스를 저장
            vec_crd.append(ss)            

        vec_crd = np.array(vec_crd, dtype='int32')
        vec.append(vec_crd)

    return np.array(vec, dtype='int32')


def _load_fashion_item(in_file: str, coordi_size: int, meta_size: int):
    """패션 아이템의 메타데이터를 불러옵니다.

    Args:
        in_file (str): 패션 아이템의 메타데이터가 저장되어있는 파일의 경로입니다.
        coordi_size (int): 하나의 코디를 구성하는 패션 아이템의 개수입니다.
        meta_size (int): 패션 아이템 메타데이터의 특징 종류 개수입니다.
    """
    print('Loading fashion item metadata...')
    with open(in_file, encoding='euc-kr', mode='r') as fin:
        names = [] # 패션 아이템의 이름을 모아두는 리스트입니다.
        metadata = []
        prev_name = ''
        prev_feat = ''
        data = ''
        for l in fin.readlines():
            line = l.strip()
            w = line.split()
            name = w[0] # ex) BL-001 (블라우스)

            # 예외 처리(aif.submit 에러 방지)
            name = name.replace('L_', '')

            # 새로운 아이템을 불러왔다면
            if name != prev_name:
                names.append(name) # 기존 아이템의 이름을 저장
                prev_name = name

            feat = w[3] # 패션 아이템의 형태 특징. F(형태), M(소재), C(색채), E(감성) 중 하나
            
            # 같은 아이템에서 다른 형태 특징을 불러온 경우
            if feat != prev_feat:
                if prev_feat != '':
                    metadata.append(data)
                
                # 패션 아이템에 대한 설명. ex) 상의 스트레이트 실루엣
                # TODO: 불용어 처리 코드 추가하기
                data = w[4] 
                for d in w[5:]:
                    data += ' ' + d
                
                prev_feat = feat
            
            # 같은 아이템에서 같은 형태 특징을 불러온 경우
            else:
                for d in w[4:]:
                    data += ' ' + d
        
        metadata.append(data)

        # 아이템 카테고리에 속하지 않는 경우를 처리
        for _ in range(coordi_size * meta_size):
            metadata.append('')

        names.append('NONE-OUTER')
        names.append('NONE-TOP')
        names.append('NONE-BOTTOM')
        names.append('NONE-SHOES')

    return names, metadata


def _position_of_fashion_item(item: str):
    """패션 아이템이 포함된 카테고리의 위치를 알아낼 때 새용하는 함수입니다.

    Args:
        item (str): 패션 아이템의 이름입니다.
    """
    prefix = item[0:2] # BL, JK, ...

    if prefix in ['JK', 'JP', 'CT', 'CD', 'VT'] or item == 'NONE-OUTER':
        idx = 0
    elif prefix in ['KN', 'SW', 'SH', 'BL'] or item == 'NONE-TOP':
        idx = 1
    elif prefix in ['SK', 'PT', 'OP'] or item == 'NONE-BOTTOM':
        idx = 2
    elif prefix == 'SE' or item == 'NONE-SHOES':
        idx = 3
    else:
        raise ValueError(f"{item} doesn't exist.")

    return idx


def _insert_into_fashion_coordi(coordi: List[str], items: List[str]):
    """코디 리스트에 아이템을 추가하는 함수입니다.
    
    코디 봇이 항상 4개의 아이템을 추천하지 않기 때문에,
    NONE으로 초기화된 코디 리스트에 아이템을 할당하여 동일한 길이를 유지하도록 합니다.

    Args:
        coordi (List[str]): 코디 리스트입니다. e.g., ['JK-001', 'KN-001', 'PT-001', 'SE-001']
        item (List[str]): 코디 리스트에 추가할 아이템입니다. e.g., ['PT-002']
    """
    new_coordi = coordi[:]

    for item in items:
        pos = _position_of_fashion_item(item)

        if item[0:2] == 'OP': # 패션 아이템이 원피스인 경우
            new_coordi[1] = 'NONE-TOP' # 상의 None 처리

        new_coordi[pos] = item

    return new_coordi


# TODO: TAG 저장 방식을 변경해보자..
def _load_trn_dialog(in_file: str='./data/task1.ddata.wst.txt'):
    """모델 학습에 사용할 텍스트 데이터를 불러올 때 사용합니다.

    Args:
        in_file (str, optional): 텍스트 데이터의 경로입니다. Defaults to './data/task1.ddata.wst.txt'.
    """
    print('loading dialog DB for training...')

    with open(in_file, encoding='euc-kr', mode='r') as fin:
        # 각각 발화문/코디/태그들을 저장하는 리스트
        data_utter = []
        data_coordi = []
        data_reward = []
        
        # data_***를 다른 함수에서 episode별로 구분하기 위해 위치 정보를 저장하는 리스트
        delim_utter = []
        delim_coordi = []
        delim_reward = []

        # delimiter로 사용되는 값들
        num_dialog = 0
        num_turn = 1
        num_coordi = 0
        num_reward = 0

        is_first = True

        for l in fin.readlines():
            line = l.strip()
            w = line.split()
            ID = w[1] # <CO>, <US>, <AC> 중 하나. 코디 봇, 사용자, 추천 아이템을 의미하는 태그

            if w[0] == '0': # w[0]: 대화 번호로, 0번은 새로운 대화가 시작되었음을 의미함.
                if is_first: # 첫 번째 데이터라면
                    is_first = False

                else:
                    # 남아있는 값들을 추가
                    data_utter.append(tot_utter.strip())

                    if prev_ID == '<CO>':
                        data_coordi.append(coordi)
                        num_coordi += 1
                    
                    if prev_ID == '<US>':
                        data_reward.append(tot_func.strip())
                        num_reward += 1

                    # 다른 전처리 함수에서 사용하기 위해, index들을 저장
                    delim_utter.append(num_turn)
                    delim_coordi.append(num_coordi)
                    delim_reward.append(num_reward)
                    num_turn += 1

                prev_ID = ID
                tot_utter = ''
                tot_func = ''

                coordi = ['NONE-OUTER', 'NONE-TOP', 'NONE-BOTTOM', 'NONE-SHOES']

                num_dialog += 1

            # 추천 아이템인 경우, 해당 아이템을 coordi에 삽입
            if ID == '<AC>':
                items = w[2:]
                coordi = _insert_into_fashion_coordi(coordi, items) # TODO

                tot_utter = ''
                continue
            
            # TAG에서 불필요한 문자 제거
            func = re.sub(pattern='[^A-Z_;]', repl='', string=w[-1])
            func = re.sub(pattern='[;]', repl=' ', string=func)

            if func == '_':
                func = ''
            
            if func != '':
                w = w[:-1]

            ### custom code ###
            if ID == '<CO>' and func in ['SUCCESS', 'CLOSING', 'SUCCESS CLOSING', 'NONE', '']: # 불필요한 코디 봇의 발화문은 무시
                continue
            ### custom code ###

            # 발화 대상이 바뀐 경우 발화문을 저장하고, ID에 따라 코디/태그를 저장
            if prev_ID != ID:
                data_utter.append(tot_utter.strip())
                    
                if prev_ID == '<CO>':
                    data_coordi.append(coordi)
                    num_coordi += 1

                if prev_ID == '<US>':
                    data_reward.append(tot_func.strip())
                    num_reward += 1

                tot_utter = ''
                tot_func = ''
                prev_ID = ID
                num_turn += 1
            
            # tot_utter에 발화문 누적
            for u in w[2:]:
                tot_utter += ' ' + u

            # TAG 누적
            tot_func += ' ' + func

        # 남아있는 값들을 저장
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

        print(f'# of dialog: {num_dialog} sets')

        # 발화 대상이 연속하여 말하는 경우, 누적된 태그에서 마지막 태그만을 사용
        data_reward_last = []
        for r in data_reward:
            r = r.split()

            if len(r) >= 1:
                data_reward_last.append(r[len(r) - 1])
            else:
                data_reward_last.append('')

        return data_utter, data_coordi, data_reward_last, \
                np.array(delim_utter, dtype='int32'), \
                np.array(delim_coordi, dtype='int32'), \
                np.array(delim_reward, dtype='int32')


# TODO
def _load_eval_dialog(in_file: str='./data/cl_eval_task1.wst.dev'):
    """모델 성능 평가에 사용할 텍스트 데이터를 불러올 때 사용합니다.

    Args:
        in_file (str, optional): 텍스트 데이터의 경로입니다. Defaults to './data/cl_eval_task1.wst.dev'.
    """
    print('loading dialog DB for evaluation...')

    with open(in_file, encoding='euc-kr', mode='r') as fin:
        data_utter = []
        data_coordi = []

        num_dialog = 0
        num_utter = 0

        is_first = True

        for line in fin.readlines():
            line = line.strip()

            # 대화 시작
            if line[0] == ';':
                if line[2:5] == 'end': # 데이터의 마지막이면 탈출
                    break
                
                if is_first:
                    is_first = False

                # 새로운 대화가 시작되었다면, 기존 말뭉치와 코디를 저장
                else:
                    data_utter.append(tot_utter)
                    data_coordi.append(tot_coordi)

                # 필요한 변수 초기화
                tot_utter = []
                tot_coordi = []
                num_dialog += 1

            # 사용자 혹은 코디봇의 발화문인 경우
            elif line[0:2] == 'US' or line[0:2] == 'CO':
                utter = line[2:].strip()
                tot_utter.append(utter)

                num_utter += 1

            # 아이템 추천인 경우
            elif line[0] == 'R':
                # 기존 코디를 가져와서 new_coordi에 저장
                coordi = line[2:].strip()
                new_coordi = ['NONE-OUTER', 'NONE-TOP', 'NONE-BOTTOM', 'NONE-SHOES']
                new_coordi = _insert_into_fashion_coordi(new_coordi, coordi.split())

                tot_coordi.append(new_coordi)
            
        if not is_first:
            data_utter.append(tot_utter)
            data_coordi.append(tot_coordi)

        # evaluation DB는 rank가 shuffle되지 않기 때문에
        # 모든 조합에 대해 rank를 0으로 선언
        data_rank = []
        rank = 0

        for _ in range(len(data_coordi)):
            data_rank.append(rank)

        print(f'# of dialog: {num_dialog} sets')

        return data_utter, data_coordi, data_rank


# TODO: metadata가 아니라 img feature만 쓰면 어떨까?
def _convert_one_coordi_to_metadata(idx_one_coordi: List[int], metadata: np.ndarray,
                                    coordi_size: int=4, img_feats = None):
    """하나의 코디 조합에 대응되는 metadata 임베딩 벡터로 변환합니다.

    Args:
        idx_one_coordi (List[int]): 하나의 코디를 이루는 패션 아이템들의 인덱스가 저장되어 있습니다.
        metadata (np.ndarray): 패션 아이템별 임베딩 벡터들이 저장되어 있습니다.
        coordi_size (int, optional): 하나의 코디를 구성하는 패션 아이템의 개수입니다. Defaults to 4.
        img_feats (_type_, optional): image features입니다. Defaults to None.
    """
    if img_feats is None:
        items = None
        
        for j in range(coordi_size):
            # 특정 카테고리의 metadata에서 아이템 인덱스에 대응되는 임베딩을 가져옴
            buf = metadata[j][idx_one_coordi[j]] # (512, )

            if j == 0:
                items = buf[:]
            else:
                items = np.concatenate([items[:], buf[:]], axis=0) # (512 * coordi-size, )

    else:
        items_meta = None
        items_feat = None

        for j in range(coordi_size):
            buf_meta = metadata[j][idx_one_coordi[j]]
            buf_feat = img_feats[j][idx_one_coordi[j]]

            if j == 0:
                items_meta = buf_meta[:]
                items_feat = buf_feat[:]

            else:
                items_meta = np.concatenate([items_meta[:], buf_meta[:]], axis=0)
                items_feat += buf_feat[:]

        items_feat /= float(coordi_size)
        items = np.concatenate([items_meta, items_feat], axis=0) # (2, 512 * coordi_size)

    return items


def _convert_dlg_coordi_to_metadata(idx_dlg_coordi: List[List[int]], metadata: np.ndarray,
                                    coordi_size: int=4, img_feats = None):
    """하나의 에피소드를 이루는 코디들을 각 코디 조합에 대응되는 metadata 임베딩 벡터로 변환합니다.

    Args:
        idx_dlg_coordi (List[List[int]]):
        하나의 에피소드에서 추천한 코디 조합이고,
        각 코디를 이루는 패션 아이템들의 인덱스가 저장되어 있습니다.

        metadata (np.ndarray): 패션 아이템별 임베딩 벡터들이 저장되어 있습니다.
        coordi_size (int, optional): 하나의 코디를 구성하는 패션 아이템의 개수입니다. Defaults to 4.
        img_feats (_type_, optional): image features입니다. Defaults to None.
    """
    # 첫 번째 조합을 임베딩으로 변환하고, 저장하는 과정
    items = _convert_one_coordi_to_metadata(idx_dlg_coordi[0], metadata, coordi_size, img_feats) # TODO
    prev_coordi = idx_dlg_coordi[0][:]
    prev_items = items[:]
    
    scripts = np.expand_dims(items, axis=0)[:] # concat을 위해 차원 추가. (1, 2048)
    
    # 나머지 코디에 대해 반복
    for i in range(1, idx_dlg_coordi.shape[0]):
        # 이전 코디와 동일하다면 복사
        if np.array_equal(prev_coordi, idx_dlg_coordi[i]):
            items = prev_items[:]

        # 다른 코디라면 변환
        else:
            items = _convert_one_coordi_to_metadata(idx_dlg_coordi[i], metadata, coordi_size, img_feats)

        # 갱신
        prev_coordi = idx_dlg_coordi[i][:]
        prev_items = items[:]

        # scripts에 concat
        items = np.expand_dims(items, axis=0)
        scripts = np.concatenate([scripts[:], items[:]], axis=0)

    return scripts # (3, 2048)


def _convert_coordi_to_metadata(idx_coordi: List[List[List[int]]], metadata: np.ndarray,
                                coordi_size: int=4, img_feats = None):
    """모든 에피소드의 코디들을 대응되는 metadata 임베딩 벡터로 변환합니다.

    Args:
        idx_coordi (List[List[List[int]]]): 각 코디를 이루는 패션 아이템들의 인덱스가 저장되어 있습니다.
        metadata (np.ndarray): 패션 아이템별 임베딩 벡터들이 저장되어 있습니다.
        coordi_size (int, optional): 하나의 코디를 구성하는 패션 아이템의 개수입니다. Defaults to 4.
        img_feats (_type_, optional): image features입니다. Defaults to None.
    """
    print('converting fashion coordi to metadata...')

    vec = []
    
    # 모든 에피소드의 코디들을 대응되는 metadata 임베딩 벡터로 변환
    for d in range(len(idx_coordi)):
        vec_meta = _convert_dlg_coordi_to_metadata(idx_coordi[d], metadata, coordi_size, img_feats) # TODO
        vec.append(vec_meta)

    return np.array(vec, dtype='float32')


def _episode_slice(data: List[str], delim: np.ndarray):
    """delim을 이용해서 data를 에피소드 단위로 분리하는 함수입니다.

    Args:
        data (List[str]): 분리하고자 하는 대상 데이터입니다.
        delim (np.ndarray): 에피소드의 위치가 저장되어있는 배열입니다.
    """
    episodes = []
    start = 0

    for end in delim:
        epi = data[start:end]
        episodes.append(epi)
        start = end

    return episodes


def _categorize(name: list, vec_item: np.ndarray, coordi_size: int=4):
    """패션 아이템의 이름을 coordi_size 개의 category로 구분할 때 사용합니다.

    Args:
        name (list): 패션 아이템의 이름들이 저장되어있는 리스트입니다.
        vec_item (np.ndarray): 패션 아이템을 의미하는 행렬들이 저장되어 있습니다. shape: (1, 512)
        coordi_size (int, optional): 하나의 코디를 구성하는 패션 아이템의 개수입니다. Defaults to 4.
    """
    slot_item = []
    slot_name = []

    for _ in range(coordi_size):
        slot_item.append([])
        slot_name.append([])

    # 카테고리에 따라 패션 아이템을 구분하는 과정
    for i in range(len(name)):
        pos = _position_of_fashion_item(name[i])

        slot_item[pos].append(vec_item[i])
        slot_name[pos].append(name[i])

    slot_item = np.array([np.array(s) for s in slot_item], dtype=object)

    return slot_name, slot_item


# TODO
def _shuffle_one_coordi_and_ranking(rank_lst: np.ndarray, coordi: List[List[str]], num_rank: int=3):
    """코디와 코디의 순위를 섞습니다.

    랭킹 모델은 아이템들을 특정 순서로 나열하는 방법을 학습해야 합니다.
    같은 아이템들이라도 어떤 순서로 나열하느냐에 따라 점수가 달라집니다.

    shuffle을 하지 않으면 모델이 학습할 수 있는 순서가 제한적이기 때문에,
    순서를 섞음으로써 다양하게 나열된 순서를 모델이 학습할 수 있게 됩니다.

    Args:
        rank_lst (np.ndarray): _description_
        coordi (List[List[str]]): 하나의 에피소드에서 추천된 num_rank 개수의 코디입니다.
        num_rank (int, optional): 추천할 패션 아이템 조합의 개수입니다. Defaults to 3.
    """
    idx = np.arange(num_rank) # np.array([0, 1, 2])

    ### custom code ###
    # TODO: 이전에 추가되었던 순서를 제외하고 새로운 순서를 만드는 코드 추가
    idx_cp = idx.copy()
    
    # 원래 순서는 무조건 데이터에 추가되기 때문에, 이를 제외한 idx를 생성
    while True:
        np.random.shuffle(idx_cp)

        if not np.array_equal(idx, idx_cp):
            break

    ### custom code ###

    for k in range(len(rank_lst)):
        if np.array_equal(idx_cp, rank_lst[k]):
            rank = k
            break
    
    # shuffle한 순서에 맞춰 코디를 추가
    rand_crd = []
    for r in range(num_rank):
        rand_crd.append(coordi[idx_cp[r]])

    return rank, rand_crd


# TODO:
def shuffle_coordi_and_ranking(coordi: np.ndarray, num_rank: int=3):
    """평가용 데이터의 코디와 순위를 섞을 때 사용합니다.

    Args:
        coordi (np.ndarray): 임베딩 벡터로 변환된 코디 조합입니다. shape: (num_batch, 3, 2048)
        num_rank (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    data_rank = []
    data_coordi_rand = []

    idx = np.arange(num_rank)
    rank_lst = np.array(list(permutations(idx, num_rank)))

    for i in range(len(coordi)):
        idx = np.arange(num_rank)
        np.random.shuffle(idx)

        # 섞어준 순서를 rank에 저장
        for k in range(len(rank_lst)):
            if np.array_equal(idx, rank_lst[k]):
                rank = k
                break

        data_rank.append(rank)
        
        coordi_rand = []
        crd = coordi[i]

        # 섞어준 순서에 맞춰 코디 저장
        for k in range(num_rank):
            coordi_rand.append(crd[idx[k]])
        
        data_coordi_rand.append(coordi_rand)

    data_coordi_rand = np.array(data_coordi_rand, dtype='float32')
    data_rank = np.array(data_rank, dtype='int32')

    return data_coordi_rand, data_rank


def _custom_load_fashion_feature(slot_name: np.ndarray, coordi_size: int=4, feat_size: int=2048):
    """패션 아이템 이미지에 대한 feature를 불러올 때 사용합니다.

    Args:
        slot_name (np.ndarray): 패션 아이템들의 이름이 카테고리별로 구분되어있는 리스트입니다.
        coordi_size (int, optional): 하나의 코디를 구성하는 패션 아이템의 개수입니다. Defaults to 4.
        feat_size (int, optional): img feature의 차원 수입니다. Defaults to 2048.
    """
    suffix = '.npy'
    feats = []
    for i in range(coordi_size):
        feat = []

        print(f"{i} 번째 카테고리의 img_feats을 불러옵니다...", end=' ')
        start = time.time()
        for n in slot_name[i]:
            if n[0:4] == 'NONE':
                feat.append(np.zeros((feat_size)))
            else:
                img_name = n + suffix
                img_feats_path = os.path.join('./data/data_new/img_feats', img_name)
                
                # npy 파일을 가져옴
                with open(img_feats_path, 'rb') as f:
                    img_feats = np.load(f)

                feat.append(np.mean(img_feats, axis=0))
        
        end = time.time()
        print(f"완료")
        print('-'*50)
        print(f"{i} 번째 카테고리 img feats 로드 시간: {end - start}")
        print('-'*50)

        feats.append(np.array(feat, dtype=object))
    feats = np.array(feats, dtype=object)

    return feats


def make_metadata(swer, in_file_fashion: str='./data/mdata.wst.txt.2023.01.26',
                  coordi_size: int=4, meta_size: int=4, use_multimodal: bool=False,
                  in_file_img_feats: str='./data/extracted_feat.json', feat_size: int=2048):
    """Train / Test에 사용할 메타데이터를 만들때 사용하는 함수입니다.

    Args:
        swer (_type_): SubWordEmb 객체입니다.
        in_file_fashion (str, optional): 패션 아이템의 메타데이터가 저장되어있는 파일의 경로입니다. Defaults to './data/mdata.wst.txt.2023.01.26'.
        coordi_size (int, optional): 하나의 코디를 구성하는 패션 아이템의 개수입니다. Defaults to 4.
        meta_size (int, optional): 패션 아이템 메타데이터의 특징 종류 개수입니다. Defaults to 4.
        use_multimodal (bool, optional): img feature를 사용할 지 여부를 나타냅니다. Defaults to False.
        in_file_img_feats (str, optional): img feature가 저장되어있는 파일의 경로입니다. Defaults to './data/extracted_feat.json'.
        feat_size (int, optional): img feature의 차원 수입니다. Defaults to 2048.
    """

    print('\n<Make Metadata>')
    if not os.path.exists(in_file_fashion):
        raise ValueError(f'{in_file_fashion} do not exists.')

    ### fashion item metadata 불러오기 ###
    name, data_item = _load_fashion_item(in_file_fashion, coordi_size, meta_size)

    ### 메타데이터(텍스트)를 벡터로 변환 ###
    print('Vectorizing data...')
    emb_size = swer.get_emb_size()
    vec_item = _vectorize_dlg(swer, data_item)

    # 하나의 아이템에 4개의 메타데이터가 있고, 각 메타데이터별로 임베딩 값이 존재하니까
    # reshape 해서 아이템별로 하나의 벡터를 가지도록 만들어줌. e.g., BL-001: (1, 512)
    vec_item = vec_item.reshape((-1, meta_size * emb_size))

    ### fashion items을 정해진 카테고리로 묶어주기 ###
    slot_name, slot_item = _categorize(name, vec_item, coordi_size)
    slot_feat = None

    if use_multimodal: # 원하면 img feature를 불러와서 사용
        slot_feat = _custom_load_fashion_feature(slot_name, coordi_size, feat_size)

    ### 카테고리별 패션 아이템 임베딩 간의 cosine similarity 계산 ###
    vec_similarities = []

    for i in range(coordi_size):
        item_sparse = sparse.csr_matrix(slot_item[i])
        
        similarities = cosine_similarity(item_sparse)
        vec_similarities.append(similarities)

    vec_similarities = np.array(vec_similarities, dtype=object)

    ### 학습용 데이터로 변환할 때 사용할 utility 정의 ###
    idx2item = []
    item2idx = []
    item_size = []
    
    for i in range(coordi_size):
        idx2item.append(dict((j, m) for j, m in enumerate(slot_name[i])))
        item2idx.append(dict((m, j) for j, m in enumerate(slot_name[i])))
        
        item_size.append(len(slot_name[i]))

    return slot_item, idx2item, item2idx, item_size, vec_similarities, slot_feat


def make_io_data(swer, item2idx: List[dict], idx2item: List[dict],
                 metadata: np.ndarray, similarities: np.ndarray, 
                 mode: str='prepare', in_file_dialog: str='./data/task1.ddata.wst.txt',
                 mem_size: int=300, coordi_size: int=4, num_rank: int=3,
                 num_perm: int=1, num_aug: int=1, corr_thres: float=1.0, img_feats=None):
    """메타데이터를 기반으로 모델 학습 및 평가에 사용할 데이터를 생성합니다.

    Args:
        swer (_type_): subword embedding 객체입니다.
        item2idx (List[dict]): 패션 아이템의 이름마다 idx가 부여되어 있습니다.
        idx2item (List[dict]): idx마다 패션 아이템의 이름이 부여되어 있습니다.
        metadata (np.ndarray): 패션 아이템의 임베딩 값이 저장된 배열입니다.
        similarities (np.ndarray): 패션 아이템 간의 cosine similarity를 계산한 결과입니다.

        mode (str, optional): mode에 따라 데이터 생성 방식이 달라집니다. 'prepare', 'eval' 모드가 존재합니다. Defaults to 'prepare'.
        in_file_dialog (str, optional): 학습에 사용할 텍스트 데이터입니다. Defaults to './data/task1.ddata.wst.txt'.
        mem_size (int, optional): _description_. Defaults to 300.
        coordi_size (int, optional): 하나의 코디를 구성하는 패션 아이템의 개수입니다. Defaults to 4.
        num_rank (int, optional): 추천할 패션 아이템 조합의 개수입니다. Defaults to 4.
        num_perm (int, optional): _description_. Defaults to 1.
        num_aug (int, optional): 데이터 증강 횟수입니다. Defaults to 1.
        corr_thres (float, optional): 데이터 증강 시 cos_sim 값에 대한 threshold로 사용합니다. Defaults to 1.0.
        img_feats (_type_, optional): 패션 아이템 이미지별 feature입니다. Defaults to None.
    """
    print('\n<Make input & output data>')

    if not os.path.exists(in_file_dialog):
        raise ValueError(f'{in_file_dialog} does not exist.')

    ### 모델 학습용 데이터 생성 ###
    if mode == 'prepare':
        # TODO: 평가 점수 기반으로 모델을 저장할 수 있게 기존 코드를 수정해보자.
        # 학습 데이터 로드
        trn_dlg, trn_crd, trn_rwd, \
            trn_delim_dlg, trn_delim_crd, trn_delim_rwd = _load_trn_dialog(in_file_dialog)

        # 에피소드 단위로 분리
        trn_dlg = _episode_slice(trn_dlg, trn_delim_dlg)
        trn_crd = _episode_slice(trn_crd, trn_delim_crd)
        trn_rwd = _episode_slice(trn_rwd, trn_delim_rwd)

        # 학습에 사용할 DB 생성 (제일 중요)
        data_dlg, data_crd, data_rnk = _make_ranking_examples(trn_dlg, trn_crd, trn_rwd,
                                                                item2idx, idx2item, similarities,
                                                                    num_rank, num_perm, num_aug, corr_thres)

    elif mode == 'eval':
        # 평가 데이터 로드
        data_dlg, data_crd, data_rnk = _load_eval_dialog(in_file_dialog) # TODO

    ### 학습용 데이터를 벡터로 변환 ###
    data_rank = np.array(data_rnk, dtype='int32')

    vec_dialog = _vectorize(swer, data_dlg)
    emb_size = swer.get_emb_size() # 128

    mem_dialog = _memorize(vec_dialog, mem_size, emb_size)

    idx_coordi = _indexing_coordi(data_crd, item2idx, coordi_size)

    vec_coordi = _convert_coordi_to_metadata(idx_coordi, metadata, coordi_size, img_feats)

    return mem_dialog, vec_coordi, data_rank


# Test Code
if __name__ == "__main__":
    swer = SubWordEmbReaderUtil(data_path='/home/suyeongp7/sub_task3_w_new_data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat')

    _metadata, _idx2item, _item2idx, \
        _item_size, _meta_similarities, _feats = make_metadata(swer=swer,in_file_fashion="/home/suyeongp7/sub_task3_w_new_data/data/mdata.wst.txt.2023.08.23",
                                                               coordi_size=4, meta_size=4,
                                                               use_multimodal=False, in_file_img_feats=False, feat_size=2048)

    dlg, crd, rnk = make_io_data(swer=swer, item2idx=_item2idx, idx2item=_idx2item,
                                 metadata=_metadata, similarities=_meta_similarities, 
                                 mode='prepare', in_file_dialog="/home/suyeongp7/sub_task3_w_new_data/data/task1.ddata.wst.txt",
                                 mem_size=16, coordi_size=4, num_rank=3,
                                 num_perm=3, num_aug=1, corr_thres=0.7, img_feats=None)
    
    print(dlg.shape)
    print(crd.shape)
    print(rnk.shape)