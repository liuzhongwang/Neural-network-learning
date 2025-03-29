'''
import torch.nn as nn
import torch.nn.functional as F



class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 输入通道1，输出通道32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)  # 输出层（10类）

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 激活函数
        x = F.max_pool2d(x, 2)  # 最大池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 1600)  # 展平张量
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  
'''

import torch.nn as nn
import torch.nn.functional as F


class Fashion_CNN(nn.Module):
    def __init__(self):
        super(Fashion_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 2, padding=1)
        # 添加 BN 层
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 2, padding=1)
        # 添加 BN 层
        self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64,128,2,padding=1)
        # self.bn3=nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(3136, 128)
        # 添加 BN 层
        self.bn_fc1 = nn.BatchNorm1d(128)
        # 添加 Dropout 层，概率为 0.5
        self.dp1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x=self.conv3(x)
        # x=self.bn3(x)
        # x=F.relu(x)
        # x=F.max_pool2d(x,2)

        x = x.view(-1, 3136)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        # 应用 Dropout 层
        x = self.dp1(x)
        x = self.fc2(x)
        return x










