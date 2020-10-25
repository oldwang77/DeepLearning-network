import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter

# 定义全局变量
modelPath = './model.pkl'
batchSize = 5
# 一共迭代的轮数
nEpochs = 20
numPrint = 1000

# 定义Summary_Writer
writer = SummaryWriter('./Result')  # 数据存放在这个文件夹

# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)

cnt = 0;

for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)  # 将输入和目标在每一步都送入GPU
    print(labels.size(0));
    cnt = cnt + 1;

    # total += labels.size(0)
print(cnt)
