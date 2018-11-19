# -*- coding: utf-8 -*-
from numpy import *
import scipy.io as scio

def L1_distance(vector1, vector2):
    rtn=0
    for i in range(len(vector1)):
        rtn += abs(vector1[i] - vector2[i])
    return rtn

# load data
datafile = '11182018.mat'
data = scio.loadmat(datafile)
test = data["test"]
train = data["train"]

test_sim=zeros((3,4,3,10))
accuracy = 0.0
correct_num = 0

# i,m is the person
# j is in test set
for i in range(3):
    for j in range(4):
        max_similarity = 0.0
        m_idx = -1
        k_idx = -1
        for m in range(3):
            for k in range(10):
                test_sim[i][j][m][k] = L1_distance(test[i][j], train[m][k])
                if max_similarity < test_sim[i][j][m][k]:
                    max_similarity = test_sim[i][j][m][k]
                    m_idx = m
                    k_idx = k

        if i == m_idx:
            correct_num += 1
        print(i, j, 'simi to ', m_idx, ' -> ', k_idx)


print('total accuracy : ', correct_num / 12.0)