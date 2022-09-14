# !pip install git+https://github.com/d2l-ai/d2l-zh@release  # installing d2l
# %matplotlib inline

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 给定 w 和 b 参数，随机生成指定数目的数据和根据 w 和 b 生成的标签 y
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
# 查看随机生成的前 batch 个数据和标签
next(iter(data_iter))

# nn是神经网络的缩写
from torch import nn
# nn.Sequential() 作为一个容器放各个层
# nn.Linear(in_features, out_features) 指定输入输出的维数的全连接层
net = nn.Sequential(nn.Linear(2, 1))

# net[0] 获取第一层
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数 Mean Squared Error
loss = nn.MSELoss()

# 定义SGD优化方法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
      # 每次取出的 X 放入 net 并求损失
      l = loss(net(X) ,y)
      # 梯度清零
      trainer.zero_grad()
      # 求梯度
      l.backward()
      # 模型更新
      trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
