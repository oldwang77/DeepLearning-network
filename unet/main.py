import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms

from mino import get_iou
from unet import Unet
from dataset import LiverDataSet

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 指定使用哪一个显卡
torch.cuda.set_device(0)

# PyTorch框架中有一个非常重要且好用的包：torchvision，
# 该包主要由3个子包组成，分别是：torchvision.datasets、torchvision.models、torchvision.transforms

# transforms.ToTensor()将一个取值范围是[0,255]的PLT.Image或者shape为(H,W,C)的NUMPY.ndarray,
# 转换成形式为(C,H,W)，取值范围在(0,1)的torch.floatTensor
# 而后面的transform.Normalize()则把0-1变换到(-1,1)

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    # 不明白可以参考：https://blog.csdn.net/qq_38765642/article/details/109779370
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()


def train_model(model, critersion, optimizer, dataload, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = critersion(outputs, labels)
            # 反向传播，参数优化
            loss.backward()
            optimizer.step()
            # 计算loss的值
            # pytorch中，.item()方法 是得到一个元素张量里面的元素值
            # 具体就是 用于将一个零维张量转换成浮点数，比如计算loss，accuracy的值
            epoch_loss += loss.item()

            # step/第几批数据
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))

    # 保存model
    torch.save(model.state_dict(),'weights_%d.pth' % epoch)
    return model


def train(args):
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    # 损失函数，该类主要用来创建衡量目标和输出之间的二进制交叉熵的标准。
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    # 初始化
    liver_dataset = LiverDataSet("/home/ming/code/u-net-liver-pytorch/data/train", transform=x_transforms, target_transform=y_transforms)

    # 读取数据集
    # 它为我们提供的常用操作有：batch_size(每个batch的大小),
    # shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)
    # shuffle是否将数据打乱；
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_model(model, criterion, optimizer, dataloaders)


def test(args):
    model = Unet(3,1)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    liver_dataset = LiverDataSet("/home/ming/code/u-net-liver-pytorch/data/val",transform = x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset,batch_size=1)

    #不启用 BatchNormalization 和 Dropout
    #训练完 train 样本后，生成的模型 model 要用来测试样本。在 model(test) 之前，
    #需要加上model.eval()，否则只要有输入数据，即使不训练，model 也会改变权值。
    # 这是model中含有的 batch normalization 层所带来的的性质。
    model.eval()

    import matplotlib.pyplot as plt
    #使matplotlib的显示模式转换为交互（interactive）模式。
    # 即使在脚本中遇到plt.show()，代码还是会继续执行
    plt.ion()

    # 在测试阶段使用with torch.no_grad()可以对整个网络都停止自动求导，
    # 可以大大加快速度，也可以使用大的batch_size来测试
    # 当然，也可以不使用with torch.no_grad
    with torch.no_grad():
        for x,_ in dataloaders:
            # sigmod把值域在0和1之间，sigmod是为了后面用imgshow方法画热图
            y = model(x).sigmoid()
            img_y = torch.squeeze(y).numpy()

            # get_iou("data/val/000_mask.png",img_y)

            # imshow方法首先将二维数组的值标准化为0到1之间的值，
            # 然后根据指定的渐变色依次赋予每个单元格对应的颜色，就形成了热图。
            plt.imshow(img_y)
            plt.pause(0.1)
        plt.show()

if __name__ == '__main__':
    # 参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    # 这种是属于可选参数
    # python test.py --batch_size=10
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")

    # 获取全部的参数
    args = parse.parse_args()

    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)
