from torch.utils.data import Dataset
import PIL.Image as Image
import os


def make_dataset(root):
    # imgs就是用来存储输入图像和mask的路径
    # 存储格式为[(输入图像,mask),...]
    imgs = []
    n = len(os.listdir(root)) // 2
    for i in range(n):
        # 组成路径的形式
        img = os.path.join(root, "%03d.png" % i)
        mask = os.path.join(root, "%03d_mask.png" % i)
        imgs.append((img, mask))
    return imgs


class LiverDataSet(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    # 凡是在类中定义了这个__getitem__ 方法，那么它的实例对象（假定为p），可以像这样
    # p[key] 取值，当实例对象做p[key] 运算时，会调用类中的方法__getitem__
    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]

        # 利用 img = Image.open(ImgPath) 打开的图片是PIL类型的，它自带resize函数。
        # 由于pytorch的顺序是(batch,c,h,w)，所以需要进行PIL类型到numpy类型转换，
        # tensorflow，numpy的顺序是(batch,h,w,c)：

        # PIL通过Image类的open()函数读取图像，返回值image有以下几个属性:
        #
        # 1.image是PIL自定义类型，需要通过numpy.array()函数转化为numpy可以处理的类型
        # 2.image.shap表示图像的形状(H W C),其中C表示通道数/深度，它的顺序为RGB
        # 3.image.size可以获取图像的高和宽
        # ４.iamge.dtype表示图像像素的数据类型unit8(0-255)

        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
