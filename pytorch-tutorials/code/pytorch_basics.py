
# coding: utf-8

# ## pytorch basics
# + Author: xiaoran
# + Time: p.m. 2019-01-17 
# 
# 展示pytorch的自动求导机制，和与numpy之间的交互
# 

# In[1]:


import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# 1. 自动求导的例子1

# In[7]:


# 1. 使用变量创建tensor，可以使用向量创建
x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# 2. 建立一个图表达式，torch会帮我们构建一个图
# 默认的表达式是 y = 2 * x + 3
y = w * x + b 

# 3. 计算y关于所有变量(x, w, b)的梯度
y.backward()

# 4. 打印出所有的梯度
print(x.grad)    # x.grad = w = 2
print(w.grad)    # w.grad = x = 1
print(b.grad)    # b.grad = 1


# 2. 自动求导例子2

# In[13]:


# 1. 随机创建二维的tensor，shape input x (10,2) and output y (10, 2)
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# 2. 建立一个全连接层， y = w * x + b, w 是权重 shape (3, 2)， b 是偏差 shape 2, 这是是默认参数还没有优化
linear = nn.Linear(3, 2)
print("w: ", linear.weight)
print("b: ", linear.bias)

# 3. 前面就是一个只有一层的MLP，定义损失函数和优化器
loss_fun = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# 4. 预测的时候就是，前向传播
pred = linear(x)

# 5. 计算损失
loss = loss_fun(pred, y)

# 6. 根据loss的反向传播，求导优化参数
loss.backward()

# 6.1. 打印出损失函数的梯度
print("dL/dw: ", linear.weight.grad)
print("dL/db: ", linear.bias.grad)
# 7. 梯度下降, 使用学习率 0.01, 这里值执行一部
optimizer.step()
# 7.1 上面的基于优化函数的梯度下降，可以用下面的两句替代
# linear.weight.data,sub_(0.01 * linear.weight.grad.data)
# linear.bias.data,sub_(0.01 * linear.bias.grad.data)

# 8. 打印执行一次梯度下降的损失函数
pred = linear(x)
loss = loss_fun(pred, y)
print("loss after 1 step optimization: ", loss.item())


# 循环方式进行梯度优化

# In[19]:


iter_k = 10
for i in range(iter_k):
    # 4. 预测的时候就是，前向传播
    pred = linear(x)

    # 5. 计算损失
    loss = loss_fun(pred, y)

    print("loss after %d step optimization: %s" % (i+1, loss.item()))

    # 6. 根据loss的反向传播，求导优化参数
    loss.backward()

    # 6.1. 打印出损失函数的梯度
#     print("dL/dw: ", linear.weight.grad)
#     print("dL/db: ", linear.bias.grad)
    # 7. 梯度下降, 使用学习率 0.01, 这里值执行一部
    optimizer.step()
    # 7.1 上面的基于优化函数的梯度下降，可以用下面的两句替代
    # linear.weight.data,sub_(0.01 * linear.weight.grad.data)
    # linear.bias.data,sub_(0.01 * linear.bias.grad.data)

    # 8. 打印执行一次梯度下降的损失函数


# ### 2 从numpy中得到数据

# In[21]:


# numpy array
x = np.array([[1,2],[3,4]])

# convert numpy array to a touch tensor
y = torch.from_numpy(x)

# convert the torch tensor to a numpy array
z = y.numpy()


# ### 3 Input pipline

# In[24]:


# Download and construct CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root="../data/",
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
image, label = train_dataset[0]
print(image.size())
print(label)


# In[25]:


# 数据架子（pytorch提供多线程和队列加载）
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 当迭代开始的时候，队列和多线程开始从文件中加载数据
data_iter = iter(train_loader)

# 每次得到小批量的数据和label
images, labels = data_iter.next()
print(labels)

# 在实际的使用时候，一般用for循环
for images, labels in train_loader:
    # Train code should be written here.
    pass


# ### 3.1 Input pipline for custom dataset(自定制数据集)
# 1. 使用下面给出的类参考的格式定义自己任务的数据
# 2. 之后，使用上面的的方式，指定batch_size,
# 3. 使用for循环或者iter的next

# In[26]:


# 定制自己的客户数据集

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file namse
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (e.g. torchvision.Transform)
        # 3. Return a data pair (e.g. image and label)
        pass
    def __len__(self):
        # the total size of your dataset
        size = 100
        return size
# 使用方式
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=64, shuffle=True)


# ### 4. 基于迁移学习的预训练模型
# 1. 例子 resnet-18
# 2. 去掉顶层网络，根据自己的数据重新定义
# 3. 设置层的状态是否支持微调
# 

# In[29]:


# Download and load the pretrained ReNet-18. 下载并预训练模型
resnet = torchvision.models.resnet18(pretrained=True)

# 设置参数，仅仅微调顶层，将其他层冻结
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetunig(根据自己的数据替换顶层网络结构，并记性微调)
label_size = 100 # 你的数据的类别的个数
resnet.fc = nn.Linear(resnet.fc.in_features, label_size)

# Forward pass, 前向传播(这里可以设置epoch，batch，iteator)
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs) # (64, 100)


# ### 5. save and load the entire model. (保存和加载模型)

# In[31]:


# Save and load the entir model, (保存加载整个模型)
torch.save(resnet, "resnet_model.ckpt")
model = torch.load("resnet_model.ckpt")

# Save and load only the model parameters (recommend) 推荐仅仅保存模型的参数
torch.save(resnet.state_dict(), "resnet_params.ckpt")
resnet.load_state_dict(torch.load("resnet_params.ckpt"))


# In[32]:


resnet

