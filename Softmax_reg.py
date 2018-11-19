# -*- coding: utf-8 -*-
import numpy
import json
from numpy import *
import scipy.io as scio
import tensorflow as tf

datafile='11182018.mat'
data=scio.loadmat(datafile)
test=data["test"]
train=data["train"]

print('test shape1 -> ',test.shape)

sess=tf.InteractiveSession()#创建会话
#定义算法公式
x=tf.placeholder(tf.float32,[409019])#None表示样本的数量可以是任意的
W=tf.Variable(tf.zeros([409019,3]))#构建一个变量，代表权重矩阵，初始化为0
b=tf.Variable(tf.zeros([3]))
y=tf.nn.softmax(tf.matmul(x,W)+b)#构建一个softmax的模型，y指样本标签的预测值

y_=tf.placeholder(tf.float32, [409019])#y_指样本标签的真实值
#定义损失函数
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#用梯度下降法最小化损失函数，学习速率为0.5
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#训练数据
tf.global_variables_initializer().run()

"""
for i in range(1000):#用随机梯度下降法，即每次从训练样本中随机选择100个样本进行训练
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x: batch_xs,y_:batch_ys})
"""

test_sim=zeros((3,4,3,10))
for i in range(3):
    for j in range(4):
        for m in range(3):
            for k in range(10):
                #test_sim[i][j][m][k]=cosine_similarity(test[i][j],train[m][k])
                print('shape x -> ', test[i][j].shape)
                print('shape x2 -> ', train[m][k].shape)
                train_step.run({x: test[i][j], y_: train[m][k]})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))#打印测试信息


