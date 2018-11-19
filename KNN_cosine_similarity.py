# -*- coding: utf-8 -*-
import numpy
from numpy import *
import scipy.io as scio

def cosine_similarity(vector1, vector2):
    x1=numpy.array(vector1)
    x2=numpy.array(vector2)
    x3=numpy.dot(x1,x2.T)/(numpy.linalg.norm(x1)*numpy.linalg.norm(x2))
    return x3

# load data
datafile = '11182018.mat'
data = scio.loadmat(datafile)
test = data["test"]
train = data["train"]

for i in range(3):
    for j in range(10):
        for k in range(260275):
            if train[i][j][k]<128:
                train[i][j][k]=1
            else:
                train[i][j][k]=0


for i in range(3):
    for j in range(4):
        for k in range(260275):
            if test[i][j][k]<128:
                test[i][j][k]=1
            else:
                test[i][j][k]=0


test_max=zeros((3,4))
test_min=zeros((3,4))
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
                test_sim[i][j][m][k] = cosine_similarity(test[i][j], train[m][k])
                if max_similarity < test_sim[i][j][m][k]:
                    max_similarity = test_sim[i][j][m][k]
                    m_idx = m
                    k_idx = k

        if i == m_idx:
            correct_num += 1
        print(i, j, 'simi to ', m_idx, ' -> ', k_idx)


print('total accuracy : ', correct_num / 12.0)