# %matplotlib inline
import random
import torch
from d2l import torch as d2l

# 人造数据生成函数
# 输入： w: 权重 | b: 偏置 | num_examples: 样本数量
def synthetic_data(w, b, num_examples):
    X = torch.normal(
        0, 1, (num_examples, len(w))
    )  # 生成服从正态分布的特征集 | (num_example, len(w)) 分别是样本数和特征数量
    y = torch.matmul(X, w) + b  # 调用点积函数，生成标签
    y += torch.normal(0, 0.01, y.shape)  # 往标签中加入噪声
    return X, y.reshape((-1, 1))  # 返回数据集 | 把y的尺寸调整为[num_examples, 1]


# 随机小批量样本生成器
# 输入：batch_size: 生成的数据数 | features: 特征集 | labels: 标签集
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 获取特征数量
    indices = list(range(num_examples))  # 生成索引
    random.shuffle(indices)  # 打乱索引
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# ? 生成数据集
true_w = torch.tensor([2, -3.4])  # 生成真正的权重
true_b = 4.2  # 真正的偏置
features, labels = synthetic_data(true_w, true_b, 1000)  # 使用权重和偏置生成1000个样本

# ? 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print(w)
batch_size = 10

# ? 定义模型
def linear_regression(X, w, b):
    return torch.matmul(X, w) + b


# ? 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# ? 更新参数 | 优化算法
# 传入参数：params: 要更新的参数 | learning_rate: 学习率 | batch_size: 小批量随机样本采集数
def sgd(params, learning_rate, batch_size):
    with torch.no_grad():
        for param in params:  # 根据梯度、学习率、采样数更新每一个参数
            param -= learning_rate * param.grad / batch_size
            param.grad.zero_()  # 清空梯度


# ? 超参数
learning_rate = 0.03  # 学习率
num_epochs = 3  # 迭代周期个数
net = linear_regression  # 线性回归方程 | 单层网络
loss = squared_loss  # 损失函数
# ? 训练网络
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):  # 获取小批量样本
        l = loss(net(X, w, b), y)  # 计算小批量样本的损失函数 | 是一个矢量，包含小批量样本中每一个样本的损失
        l.sum().backward()  # 计算梯度（注意，梯度保存在自变量那，这里的自变量是w和b）
        sgd([w, b], learning_rate, batch_size)  # 更新参数 | 优化
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)  # 每次迭代后的损失函数
        print(f"epoch{epoch+1}, loss{float(train_l.mean()):f}")

print(f"w的估计误差{true_w-w.reshape(true_w.shape)}")
print(f"b的估计误差{true_b-b}")
