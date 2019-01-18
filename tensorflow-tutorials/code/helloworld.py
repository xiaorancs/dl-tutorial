
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[3]:


# Simpale hello world using Tensorflow
# Create a Constant op 
# The op is added as a node to the defaut graph
#
# The value returned by the constructor represents the output
# of the Constant op. 
hello = tf.constant("Hello, Tensorflow!")


# In[4]:


# Start tf session
sess = tf.Session()


# In[5]:


# Run graph
print(sess.run(hello))

