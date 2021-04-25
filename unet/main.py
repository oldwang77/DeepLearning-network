import torch
import argparse

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 指定使用哪一个显卡
torch.cuda.set_device(0)

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

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
