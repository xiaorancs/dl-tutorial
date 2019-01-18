
# coding: utf-8

# ## linear regression
# 使用pytorch构造线性回归模型，y = W * X + b

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# 设置超参数
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001


# In[3]:


# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


# In[4]:


# Linear regression model
linear_model = nn.Linear(input_size, output_size)

# define loss and optimizer
loss_fun = nn.MSELoss()
optimizer = torch.optim.SGD(linear_model.parameters(), lr=learning_rate)


# In[5]:


# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    
    # Forwar pass
    outputs = linear_model(inputs)
    loss = loss_fun(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))


# In[7]:


# Plot the graph
predicted = linear_model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, "ro", label="Original data")
plt.plot(x_train, predicted, label="Fitted line")
plt.legend()
plt.show()


# In[8]:


# Save the model checkpoint
torch.save(linear_model.state_dict(), "../data/linear_model.ckpt")

