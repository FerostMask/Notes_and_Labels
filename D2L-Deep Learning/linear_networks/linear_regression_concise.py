import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):  # 创建迭代器
    dataset = data.TensorDataset(*data_arrays)  # 转换数据集格式
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 返回小批量样本


batch_size = 10
data_iter = load_array((features, labels), batch_size)  # 创建一个迭代器，返回随机小批量样本

# ? 定义模型
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))  # Sequential将多个层连接起来，Linear是全连接层，2表示有两个特征，1表示有一个输出
# ? 初始化模型参数
net[0].weight.data.normal_(
    0, 0.01
)  # 初始化权重，通过net[0]来访问网络的第一层，weight.data访问权重，normal_重写参数（从正态分布中取随机数）
net[0].bias.data.fill_(0)  # 初始化偏置，bias.data访问偏置，fill_重写参数
# ? 定义损失函数
loss = nn.MSELoss()  # 平方L_2范数
# ? 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# ? 训练模型
num_epochs = 3
for epoch in range(num_epochs):  # 训练周期
    for X, y in data_iter:  # 生成小批量样本
        l = loss(net(X), y)  # net(X)在计算模型预测值，y是标签（真实值） | 前向传播
        trainer.zero_grad()  # 清除梯度
        l.backward()  # 计算梯度 | 反向传播
        trainer.step()  # 根据梯度优化模型参数
    l = loss(net(features), labels)  # 计算损失，表示当前模型训练效果
    print(f"epoch{epoch+1}, loss{l:f}")

w = net[0].weight.data
print("w的估计误差：", true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print("b的估计误差：", true_b - b)
print(net[0].weight.data.grad)
