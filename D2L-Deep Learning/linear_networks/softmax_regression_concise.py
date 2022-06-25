import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256  # 批大小

import torchvision
from torch.utils import data
from torchvision import transforms


def get_dataloader_workers():
    return 0


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]  # 数据转换
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=False
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=False
    )
    return (
        data.DataLoader(
            mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()
        ),
        data.DataLoader(
            mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()
        ),
    )


train_iter, test_iter = load_data_fashion_mnist(batch_size)  # 定义数据迭代器
# pytorch不会隐式地调整输入的形状，因此在线性层前定义平展层平展层(flatten)，来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)  # 以均值为0，标准差为0.01初始化网络权重


net.apply(init_weights)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# ? 分类精度
def accuracy(y_hat, y):
    if (
        len(y_hat.shape) > 1 and y_hat.shape[1] > 1
    ):  # y_hat.shape 矩阵大小 | len(y_hat.shape) 维度数 | y_hat.shape[1] 第二维度的长度
        y_hat = y_hat.argmax(axis=1)  # 找出第二维度上所有最大值 | 返回的是索引，所以下面才可以进行比较
    cmp = y_hat.type(y.dtype) == y  # 将预测值与标签进行比较 | 结果为布尔类型
    return float(cmp.type(y.dtype).sum())  # 返回预测正确的个数（1为正确）


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)  # 计算预测值 | 前向传播
        l = loss(y_hat, y)  # 计算损失
        if isinstance(updater, torch.optim.Optimizer):
            # 使用pytorch内置的优化器和损失函数
            updater.zero_grad()  # 清除梯度
            l.mean().backward()  # 反向传播，计算梯度
            updater.step()  # 根据梯度优化模型参数
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, loss, num_epochs, updator):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updator)
        print(train_metrics)


num_epochs = 10
train_ch3(net, train_iter, loss, num_epochs, trainer)
