
import torch
import torch.nn as nn

#第一步：使用类继承 nn.Module 是构建神经网络的标准方式，建立自己的神经网络模型类

class MyModel(nn.Module):
    # 初始化，使用父类的初始换函数就行
    def __init__(self):
        super(MyModel,self).__init__()
        # 创建两层的神经网络
        self.fc1=nn.Linear(3,30)
        self.fc2=nn.Linear(30,2)


    # 前向传播函数
    def forward(self,x):
        x=self.fc1(x)
        x=torch.relu(x)
        x=self.fc2(x)
        x=torch.sigmoid(x)
        return x


