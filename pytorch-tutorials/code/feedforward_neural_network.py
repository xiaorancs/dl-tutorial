
# coding: utf-8

# ## pytorch 前馈神经网络
# pytorch教程，实现前馈神经网络.

# In[1]:


import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


# In[3]:


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[6]:


# set Hyper-parameters
input_size = 784
hidden1_size = 500
hidden2_size = 600
hidden_size = [hidden1_size, hidden2_size]

num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


# In[5]:


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="../data/", 
                                           train=True, 
                                           transform=transforms.ToTensor(), 
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root="../data/",
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# In[20]:


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu1 = nn.ReLU()

#         self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
#         self.relu2 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size[0], num_classes)
    
    def forward(self, x):
        out1 = self.fc1(x)
        out1 = self.relu1(out1)

#         out2 = self.fc2(out1)
#         out2 = self.relu2(out2)

        out3 = self.fc2(out1)
        return out3
model = NeuralNet(input_size, hidden_size, num_classes).to(device)


# In[21]:


# Loss and optimizer
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[22]:


# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fun(outputs, labels)
        
        # Bcakward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# In[29]:


# Test the model
# In test phase, we don't need to copute gradients (for memory efficiency)
with torch.no_grad():
    corrent = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        lables = labels.to(device)
        outputs = model(images)
#         print(outputs)
        _, predicted = torch.max(outputs.data, 1)
#         print(predicted)
        total += labels.size(0)
        corrent += (predicted == labels).sum()
    print("Accuracy of the network on the 10000 test images: {} %".format(100 * corrent / total))

