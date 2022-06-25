import torch
from IPython import display
from d2l import torch as d2l
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


batch_size = 256  # 设置批大小
train_iter, test_iter = load_data_fashion_mnist(batch_size)  # 这里用多线程读取会暴毙
# ? 初始化模型参数

num_inputs = 28 * 28  # 输入图像大小 | 平铺为向量
num_outputs = 10  # 输出数量，与类别相同

W = torch.normal(
    0, 0.01, size=(num_inputs, num_outputs), requires_grad=True
)  # 初始化权重 | 10个类别，28*28个像素
b = torch.zeros(num_outputs, requires_grad=True)

# ? softmax操作
def softmax(X):
    X_exp = torch.exp(X)  # 对每个数进行指数运算
    partition = X_exp.sum(1, keepdim=True)  # 按列求和，计算每行的总和
    return X_exp / partition  # 这里使用了广播机制


# ? 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)  # 这里计算的是点积


# ? 定义损失函数
def cross_entropy(y_hat, y):  # 交叉熵损失
    return -torch.log(y_hat[range(len(y_hat)), y])  # -y*log(y^)


# ? 分类精度
def accuracy(y_hat, y):
    if (
        len(y_hat.shape) > 1 and y_hat.shape[1] > 1
    ):  # y_hat.shape 矩阵大小 | len(y_hat.shape) 维度数 | y_hat.shape[1] 第二维度的长度
        y_hat = y_hat.argmax(axis=1)  # 找出第二维度上所有最大值
    cmp = y_hat.type(y.dtype) == y  # 将预测值与标签进行比较 | 结果为布尔类型
    return float(cmp.type(y.dtype).sum())  # 返回预测正确的个数（1为正确）


# ? 计算指定数据集上模型的精度
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):  # 判断net是不是torch.nn.Module的实例
        net.eval()  # 将模型设置为评估模式
    metric = d2l.Accumulator(2)  # 创建实例，用于存储正确预测的数量和预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)  # 计算预测值 | 正向推演
        l = loss(y_hat, y)
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


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updator):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updator)
        print(train_metrics)


lr = 0.1


def updator(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updator)
