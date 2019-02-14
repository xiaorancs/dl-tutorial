#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression Example
# A linear regression learning algorithm example using TensorFlow library.
# 
# + Author: xiaoran
# + Time: 2019-02-14 PM
# + Copy: https://github.com/aymericdamien/TensorFlow-Examples/

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# 设置超参数
learning_rate = 0.01
training_epochs = 1000
display_step = 50


# In[3]:


# 训练数据
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]


# In[7]:


# 设置图中输入数据的占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 随机初始化变量W 和 b
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bais")

# 构造线性回归模型 y = w * x + b
pred_y = tf.add(tf.multiply(W, X), b)

# 计算损失函数和优化方式
# 使用最小平方误差
loss = tf.reduce_sum(tf.pow(pred_y - Y, 2)) / (2 * n_samples)

# 使用随机梯度下降进行优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# In[8]:


# 初始化随机变量
init = tf.global_variables_initializer()


# In[14]:


# satrt graph
with tf.Session() as sess:
    sess.run(init)
    
    # start training
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X:x, Y:y})
        # display log 
        if (epoch+1) % display_step == 0:
            print("Eopch:","%04d" % (epoch+1), "loss=", sess.run(loss, feed_dict={X: train_X, Y: train_Y}),"W=", sess.run(W), "b=", sess.run(b))
    print("Training Finish!")
    print("Training loss=", sess.run(loss, feed_dict={X: train_X, Y: train_Y}),"W=", sess.run(W), "b=", sess.run(b))
    print("Plot linear")
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


# In[ ]:




