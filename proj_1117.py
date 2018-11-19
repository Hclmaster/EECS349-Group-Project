# -*- coding: utf-8 -*-
import numpy
import json
from numpy import *
import scipy.io as scio

def cosine_similarity(vector1, vector2):
    x1=numpy.array(vector1)
    x2=numpy.array(vector2)
    x3=numpy.dot(x1,x2.T)/(numpy.linalg.norm(x1)*numpy.linalg.norm(x2))
    return x3

def L1_distance(vector1,vector2):
    rtn=0
    for i in range(len(vector1)):
        """
        if vector1[i]==vector2[i]:
            rtn=rtn
        else:
            rtn=rtn+1
        """
        rtn += abs(vector1[i] - vector2[i])
    return rtn


datafile='11182018.mat'
data=scio.loadmat(datafile)
test=data["test"]
train=data["train"]
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
test_sim=zeros((3,4,3,10))
test_max=zeros((3,4))
test_min=zeros((3,4))
for i in range(3):
    #print('i',i)
    for j in range(4):
        #print('j',j)
        max_simi = 0.0
        m_idx = -1
        k_idx = -1
        for m in range(3):
            for k in range(10):
                #test_sim[i][j][m][k] = L1_distance(test[i][j], train[m][k])
                test_sim[i][j][m][k]=cosine_similarity(test[i][j],train[m][k])
                if max_simi < test_sim[i][j][m][k]:
                    max_simi = test_sim[i][j][m][k]
                    m_idx = m
                    k_idx = k

        print(i,j,'simi to ',m_idx, ' -> ', k_idx)
