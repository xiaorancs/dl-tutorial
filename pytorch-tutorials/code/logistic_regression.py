
# coding: utf-8

# ## logistic regression
# 使用MNIST数据测试torch的softmax回归
# 

# In[2]:


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# In[3]:


# 超参数
inputs_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


# In[4]:


# MNIST dataset(images and labels) 
# 使用前面提到过的数据加载的输入pipeline
train_dataset = torchvision.datasets.MNIST(root="../data/", 
                                           train=True, 
                                           transform=transforms.ToTensor(), 
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root="../data/", 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data Loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# In[5]:


# logistic regression model, 和 线性回归一样，只是损失函数不一样
model = nn.Linear(inputs_size, num_classes)

# Loss and optimizer
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[6]:


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)

        # Forward pass
        outputs = model(images)
        loss = loss_fun(outputs, labels)
        
        # Bachward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# In[7]:


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    corrent = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        corrent += (predicted == labels).sum()
    print("Accuracy of the model on the 10000 test iamges: {} %".format(100 * corrent / total))


# In[9]:


# Save the model checkpoint
torch.save(model.state_dict(), "../model/lr_model.ckpt")

