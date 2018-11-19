#加载数据,手动把数据集粘贴到MNIST_data文件夹下
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)#使用one-hot编码
#评测模型效果
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

import tensorflow as tf
sess=tf.InteractiveSession()#创建会话
#定义算法公式
x=tf.placeholder(tf.float32,[None,784])#None表示样本的数量可以是任意的
W=tf.Variable(tf.zeros([784,10]))#构建一个变量，代表权重矩阵，初始化为0
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)#构建一个softmax的模型，y指样本标签的预测值

y_=tf.placeholder(tf.float32,[None,10])#y_指样本标签的真实值
#定义损失函数
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#用梯度下降法最小化损失函数，学习速率为0.5
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#训练数据
tf.global_variables_initializer().run()
for i in range(1000):#用随机梯度下降法，即每次从训练样本中随机选择100个样本进行训练
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x: batch_xs,y_:batch_ys})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))#打印测试信息