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

import sys
import numpy as np
import pandas as pd
from scipy import stats
from itertools import permutations 
NUM_RANKING = 3

def calculate_weighted_kendall_tau(pred, label, rnk_lst, num_rnk):
    """
    calcuate Weighted Kendall Tau Correlation
    """
    total_count = 0
    total_corr = 0
    for i in range(len(label)):
        # weighted-kendall-tau는 순위가 높을수록 큰 숫자를 갖게끔 되어있기 때문에 
        # 순위 인덱스를 반대로 변경해서 계산함 (1위 → 가장 큰 숫자)
        corr, _ = stats.weightedtau(num_rnk-1-rnk_lst[label[i]],
                                    num_rnk-1-rnk_lst[pred[i]])
        total_corr += corr
        total_count += 1
    return (total_corr / total_count)      

y_true = pd.read_csv(sys.argv[1], header=None, encoding='utf8').to_numpy()
y_pred = pd.read_csv(sys.argv[2], header=None, encoding='utf8').to_numpy()

rnk_lst = np.array(list(permutations(np.arange(NUM_RANKING), NUM_RANKING)))

# get scores
score = np.round(
    calculate_weighted_kendall_tau(y_pred, y_true, rnk_lst, NUM_RANKING), 7)

print("score:", score)