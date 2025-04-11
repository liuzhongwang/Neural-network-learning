import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

class CIFAR10_VGG(nn.Module):
    def __init__(self, vgg_name):
        super(CIFAR10_VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # 计算经过特征提取后输出的特征图尺寸
        self.classifier = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':  # 如果是M就是池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:  # 否则是卷积层，每个卷积层跟一个ReLu激活函数(还可以加一个BN层优化)
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]  # 使用自适应平均池化确保输出尺寸为 1x1
        return nn.Sequential(*layers)




'''import torch.nn as nn
import torch.nn.functional as F

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 添加 BN 层
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 添加 BN 层
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # 添加 BN 层
        self.pool3 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # 在卷积层后使用 BN 层
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 在全连接层之间应用 Dropout
        x = self.fc2(x)
        return x'''
