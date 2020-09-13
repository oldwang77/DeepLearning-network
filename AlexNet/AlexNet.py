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


# 定义神经网络
# 训练AlexNet
class Net(nn.Module):
    '''
    三层卷积，三层全连接  (应该是5层卷积，由于图片是 32 * 32，且为了效率，这里设成了 3 层)
    '''

    def __init__(self):
        super(Net, self).__init__()
        # 五个卷积层
        self.conv1 = nn.Sequential(  # 输入 32 * 32 * 3
            # kernel_size 卷积核大小，正方形卷积只为单个数字
            # 结合out_channels可以推测出卷积核的大小是3*3*3，一共有6个卷积核
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(  # 输入 16 * 16 * 6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (16-2)/2+1 = 8
        )
        self.conv3 = nn.Sequential(  # 输入 8 * 8 * 16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (8-2)/2+1 = 4
        )
        self.conv4 = nn.Sequential(  # 输入 4 * 4 * 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (4-2)/2+1 = 2
        )
        self.conv5 = nn.Sequential(  # 输入 2 * 2 * 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (2-2)/2+1 = 1
        )  # 最后一层卷积层，输出 1 * 1 * 128

        # 全连接层
        self.dense = nn.Sequential(
            # class torch.nn.Linear(in_features, out_features, bias=True)
            nn.Linear(128,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # 在函数的参数中经常可以看到-1例如x.view(-1, 4)
        # 这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成4列，那不确定的地方就可以写成-1
        x = x.view(-1, 128)
        # dense表示全联接层
        x = self.dense(x)
        return x

net = Net().to(device)


# 使用测试数据测试网络
def Accuracy():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集不需要反向传播
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # 将输入和目标在每一步都送入GPU
            outputs = net(images)
            _, pridected = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            total += labels.size(0)
            correct += (pridected == labels).sum().item();
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return 100.0 * correct / total


# 训练函数
def train():
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)     # lr:学习率,momentum:动量
    iter = 0
    num = 1

    # 训练网络
    for epoch in range(nEpochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            iter = iter + 1
            # 取数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和目标在每一步都送入GPU
            # 将梯度设置为0
            optimizer.zero_grad()
            # 训练
            outputs = net(inputs)
            loss = criterion(outputs, labels).to(device)
            loss.backward()  # 反向传播
            writer.add_scalar('loss', loss.item(), iter)
            # 一旦梯度被如backward()之类的函数计算好后，我们就可以调用step这个函数。
            optimizer.step()  # 优化

            # 统计数据
            running_loss += loss.item()
            if i % numPrint == 999:  # 每 batchsize * numPrint 张图片，打印一次
                print('epoch: %d\t batch: %d\t loss: %.6f' % (epoch + 1, i + 1, running_loss / (batchSize*numPrint)))
                running_loss = 0.0
                writer.add_scalar('accuracy', Accuracy(), num + 1)
                num = num + 1
    # 保存模型
    torch.save(net, './model.pkl')


if __name__ == '__main__':
    # 如果模型存在，就加载模型
    if os.path.exists(modelPath):
        print('model exists')
        net = torch.load(modelPath)
        print('model loaded')
    else:
        print('model not exists')
    print('Train Started')
    train()
    writer.close()
    print('Training Finished')
