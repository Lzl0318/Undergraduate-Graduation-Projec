import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pickle
import PIL.Image as Image
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] ='True'
# 参数定义
parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')

parser.add_argument('--index', type=int, default="4", help='the index for leaking images on Dataset.')
parser.add_argument('--image', type=str, default="", help='the path to customized image.')
parser.add_argument('--dataset', type=str, default="MNIST", help='choose your dataset')
parser.add_argument('--noise_mutipler', type=float, default=0.005, help='noise mutipler')
args = parser.parse_args()


class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


# define net
class LeNet(nn.Module):
    def __init__(self, channel=3, hidden=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act()
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    # 设计前向传播算法
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):  # hasattr：函数用于判断对象是否包含对应的属性。
            m.weight.data.uniform_(-0.5, 0.5)  # 对m.weight.data进行均值初始化。m.weights.data指的是网络中的卷积核的权重
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"): # 对偏置进行初始化
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

def main():
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = args.dataset
    root_path = '.'
    data_path = os.path.join(root_path, './data').replace('\\', '/')  # 指定数据存放的路径地址， replace是进行转义
    save_path = os.path.join(root_path, 'results/DLG_%s' % dataset).replace('\\', '/')  # 图片保存的路径

    lr = 0.5
    num_dummy = 1  # 一次输入还原的图片数量
    iteration = 200  # 一张图片迭代的次数
    num_exp = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tt = transforms.Compose([transforms.ToTensor()]) # 将图像类型数据（PILImage）转换成Tensor张量
    tp = transforms.Compose([transforms.ToPILImage()]) # 将Tensor张量转换成图像类型数据

    '''
    打印路径而已
    '''
    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'): # 判断是否存在results文件夹，没有就创建，Linux中mkdir创建文件夹
        os.mkdir('results')
    if not os.path.exists(save_path): # 是否存在路径， 不存在则创建保存图片的路径
        os.mkdir(save_path)

    '''
    加载数据
    '''
    if dataset == 'MNIST' or dataset == 'mnist':  # 判断是什么数据集
        image_shape = (28, 28)  # mnist数据集图片尺寸是28x28
        num_classes = 10  # mnist数据分类为十分类： 0 ～ 9
        channel = 1  # mnist数据集是灰度图像所以是单通道
        hidden = 588
        dst = datasets.MNIST(data_path, download=False)

    elif dataset == 'cifar10' or dataset == 'CIFAR10':
        image_shape = (32, 32)  # cifar10数据集图片尺寸是32x32
        num_classes = 10  # cifar10数据分类为十分类：卡车、 飞机等
        channel = 3  # cifar10数据集是RGB图像所以是三通道
        hidden = 768
        dst = datasets.CIFAR10(data_path, download=False)

    elif dataset == 'cifar100' or dataset == 'CIFAR100':
        image_shape = (32, 32)  # cifar100数据集图片尺寸是32x32
        num_classes = 100  # cifar100数据分类为一百个分类
        channel = 3  # cifar100数据集是灰度图像所以是单通道
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=False)
    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, './data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)
    else:
        exit('unkown dataset')  # 未定义的数据集

    for idx_net in range(num_exp):
        net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
        net.apply(weights_init)

        print('running %d|%d experiment' % (idx_net, num_exp))
        net = net.to(device)

        print('%s, Try to generate %d images' % ('DLG', num_dummy))

        criterion = nn.CrossEntropyLoss().to(device)
        imidx_list = []  # 用于记录当前还原图片的下标

        for imidx in range(num_dummy):
            idx = args.index
            imidx_list.append(idx)
            tmp_datum = tt(dst[idx][0]).float().to(device) # 将数据集中index对应的图片数据拿出来转换成Tensor张量
            tmp_datum = tmp_datum.view(1, *tmp_datum.size()) # 将tmp_datum数据重构形状， 可以用shape打印出来看看
            tmp_label = torch.Tensor([dst[idx][1]]).long().to(device) # 将数据集中index对应的图片的标签拿出来转换成Tensor张量
            tmp_label = tmp_label.view(1, ) # 将标签重塑为列向量形式
            if imidx == 0:  # 如果imidx为0， 代表只处理一张图片
                gt_data = tmp_datum  # gt_data表示真实图片数据
                gt_label = tmp_label  # gt_label 表示真实图片的标签
            else:
                gt_data = torch.cat((gt_data, tmp_datum), dim=0)  # 如果是多张图片就要将数据cat拼接起来
                gt_label = torch.cat((gt_label, tmp_label), dim=0)

            # compute original gradient
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())

            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            # gaussian noise with specific variance
            x = []
            for i in range(len(original_dy_dx)):
                r = args.noise_mutipler * torch.randn_like(original_dy_dx[i])
                x.append(r)

            # adding noise to gradient
            noise_dy_dx = []
            for i in range(len(original_dy_dx)):
                noise_dy_dx.append(original_dy_dx[i] + x[i])

            # generate dummy data and label。 生成假的数据和标签
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)  # 设置优化器为拟牛顿法

        history = []  # 全部的假的数据（这里假的数据指的是随机产生的假图像）
        history_iters = []  # 画图使用的迭代次数
        grad_difference = []  # 真实梯度和虚假梯度的差
        data_difference = []  # 真实图片和虚假图片的差
        train_iters = []

        print('lr =', lr)
        for iters in range(iteration): # 开始训练迭代

            def closure(): # 闭包函数
                optimizer.zero_grad()
                pred = net(dummy_data)

                # 将假的预测进行softmax归一化，转换为概率
                dummy_loss = -torch.mean(
                        torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                    # dummy_loss = criterion(pred, gt_label)

                # 对假的数据进行自动微分， 求出假的梯度
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                grad_diff = 0 # 定义真实梯度和假梯度的差值
                for gx, gy in zip(dummy_dy_dx, noise_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure) # 优化器更新梯度
            current_loss = closure().item()
            train_iters.append(iters)
            grad_difference.append(current_loss)
            data_difference.append(torch.mean((dummy_data - gt_data) ** 2).item())

            if iters % int(iteration / 20) == 0:  # 这一行是代表多少个iters画一张图
                current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())) # 每次迭代打印出来时间
                print(current_time, iters, '梯度差 = %.8f, 数据差 = %.8f' % (current_loss, data_difference[-1])) # 打印出梯度差和数据差
                history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)]) # history 记录的是假的图片数据
                history_iters.append(iters)  # 记录迭代次数用于画图使用

                for imidx in range(num_dummy):  # 这个循环是迭代有多少张图片输入
                    plt.figure(figsize=(12, 4))
                    plt.subplot(2, 12, 1) # 在figure画布上画子图的意思
                    # plt.imshow(tp(gt_data[imidx].cpu())) # 这一行是显示真实图片的意思, 如果是mnist数据集，将这一行改为如下
                    plt.imshow(tp(gt_data[imidx].cpu()), cmap='gray') # 灰度图像
                    for i in range(min(len(history), 19)): # 这一行是迭代画出子图的意思
                        plt.subplot(2, 12, i + 2)
                        # plt.imshow(history[i][imidx]) # 在figure显示history存储假的图片数据
                        plt.imshow(history[i][imidx], cmap='gray') # 显示灰度图像
                        plt.title('iter=%d' % (history_iters[i])) # 第几次迭代
                        plt.axis('off')

                    plt.savefig('%s/DLG_on_%s_%05d_noise_%05f.png' % (save_path, imidx_list, imidx_list[imidx], args.noise_mutipler)) # 保存图片地址
                    plt.close()

                if current_loss < 0.000001:  # converge
                    break

            loss_DLG = grad_difference
            label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
            mse_DLG = data_difference

    print('imidx_list 图片的index :', imidx_list)
    print('梯度差 :', loss_DLG[-1])
    print('数据差 :', mse_DLG[-1])
    print('gt_label 真实标签 :', gt_label.detach().cpu().data.numpy(), '虚假的还原数据标签: ', label_DLG)

    print('----------------------\n\n')


if __name__ == '__main__':
    main()



