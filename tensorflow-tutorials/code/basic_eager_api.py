
# coding: utf-8

# ## Tensorflow Eager API 的简单介绍
# + Author: xiaoran
# + Time: P.M. 2019-01-17
# 
# Eager API是直接运行操作符，不用定义图，我们可以简单的认为不需要定义Session，只需要调用tf.enable_eager_excution()
# 

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


# Set Eager API
tf.enable_eager_execution()
tfe = tf.contrib.eager


# In[3]:


# Define constant tensors
a = tf.constant(2)
print("a = %i" % a)

b = tf.constant(3)
print("b = %i" % b)


# In[4]:


# 不需要session可以直接运行操作符
c = a + b
print("a + b = %i" % c)

d = a * b
print("a * b = %i" % d)


# In[5]:


# tensor he numpy 直接兼容
a = tf.constant([[2., 2.], [1., 0.]], dtype=tf.float32)
print("Tensor: \n a = %s" % a)

b = np.array([[3., 0.], [5., 1.]], dtype=np.float32)
print("NumpyArray: \n b = %s" % b)


# In[6]:


# Tensor 支持迭代操作
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])

