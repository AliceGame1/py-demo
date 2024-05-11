import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
X = np.random.uniform(-1, 1, (500, 2))
y = np.zeros((500, 1))
y[(X[:, 0] > 0) & (X[:, 1] > 0)] = 1
y[(X[:, 0] < 0) & (X[:, 1] < 0)] = 1

# 转换为Tensor
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# 定义神经网络类
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 设置输入、隐藏和输出的维度
input_size = 2
hidden_size = 100
output_size = 10

# 创建神经网络实例
model = NeuralNetwork(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 定义批量大小和迭代次数
batch_size = 10
num_epochs = 10000

# 训练神经网络
for epoch in range(num_epochs):
    # 随机打乱数据
    indices = torch.randperm(X.size(0))
    X = X[indices]
    y = y[indices]
    
    # 分批训练
    for i in range(0, X.size(0), batch_size):
        # 获取当前批次的数据
        inputs = X[i:i+batch_size]
        targets = y[i:i+batch_size]
        
        # 前向传播
        output = model(inputs)
        
        # 计算损失
        loss = criterion(output, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 打印损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.item()}")

# 打印神经网络的参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'{name}: {param.data}')
