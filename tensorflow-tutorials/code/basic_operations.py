
# coding: utf-8

# In[1]:


# 使用tensorflow运行基础的操作符
# Author: xiaoran


# In[2]:


import tensorflow as tf


# In[3]:


# Basic constant operations
# The value returned by the constructor represents the output od the Constant op. 
a = tf.constant(2)
b = tf.constant(3)


# In[6]:


# Launch the default graph
with tf.Session() as sess:
    print("a: %i" % sess.run(a), "b: %i" % sess.run(b))
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(a*b))


# In[8]:


# 使用基础变量作为默认图的输入
# 一般用占位符作为变量, 运行图的时候进行数据填充
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)


# In[9]:


# 定义一些操作符节点
add = tf.add(a, b)
mul = tf.multiply(a, b)


# In[10]:


# Lauch the default graph
with tf.Session() as sess:
    # 填充变量需要的数据，运行对应的操作符
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))
    


# In[11]:


# 更多的操作，和上面类似，可以使用tensor计算
# 使用tensorflow创建矩阵并进行矩阵乘法
matrix1 = tf.constant([[3., 3.]]) # 1 x 2
matrix2 = tf.constant([[2.], [2.]]) # 2 x 1
# 使用矩阵乘法操作符
product = tf.matmul(matrix1, matrix2)

# 加载默认图运行session
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

