# !pip install git+https://github.com/d2l-ai/d2l-zh@release  # installing d2l
# %matplotlib inline

import random
import torch
from d2l import torch as d2l

# 给定 w 和 b 参数，随机生成指定数目的数据和根据 w 和 b 生成的标签 y
def synthetic_data(w, b, num_examples):  
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# 自己设置 w 和 b 参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 查看随机生成的第一个数据和对应 label
print('features:', features[0],'\nlabel:', labels[0])

# 读取 batch_size 大小的数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # yield：返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后开始
        yield features[batch_indices], labels[batch_indices]

# 自定义 batch 大小
batch_size = 10

# 读取第一个小批量数据样本并打印
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，并将偏置初始化为0
# requires_grad=True 自动微分来计算梯度
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 线性回归模型
def linreg(X, w, b):  
    return torch.matmul(X, w) + b

# 均方损失
def squared_loss(y_hat, y):  
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 小批量随机梯度下降
def sgd(params, lr, batch_size): 
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            # 清除 param 的梯度值
            param.grad.zero_()

# 学习率设为0.03，3个epoch
lr = 0.03
num_epochs = 3

net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')