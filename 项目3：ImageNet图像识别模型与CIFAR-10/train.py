'''import torch
import torch.nn as nn
import torch.optim as optim
# from Model import CIFAR10_VGG
from Model import CIFAR10_CNN
from data_generator import train_loader, test_loader

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移动到 GPU 上
model = CIFAR10_CNN().to(device)
# 添加 L2 正则化，设置 weight_decay 参数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    for i, (data, label) in enumerate(train_loader):
        # 将数据和标签移动到 GPU 上
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        predict = model(data)
        train_loss = criterion(predict, label)
        train_loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch {epoch} | Batch {i} | Loss: {train_loss.item():.4f}')


def train_test():
    model.eval()
    loss = 0.0
    correct_num = 0
    with torch.no_grad():
        for data, label in train_loader:
            # 将数据和标签移动到 GPU 上
            data, label = data.to(device), label.to(device)
            predict = model(data)
            loss += criterion(predict, label).item()
            y = predict.argmax(dim=1)
            correct_num += y.eq(label).sum().item()
    loss /= len(train_loader)
    accuracy = 100. * correct_num / len(train_loader.dataset)
    print(f'Loss:{loss} | accuracy{accuracy:.4f}')

def test():
    model.eval()
    correct_num=0
    with torch.no_grad():
        for data,label in test_loader:
            data,label=data.to(device),label.to(device)
            predict=model(data)
            y=predict.argmax(dim=1)
            correct_num+=y.eq(label).sum().item()
    accuracy=100.*correct_num/len(test_loader.dataset)
    print(f'accuracy{accuracy:.4f}')

for i in range(10):
    train(i)
    train_test()
    test()'''

import torch
import torch.nn as nn
import torch.optim as optim
from Model import CIFAR10_VGG
from data_generator import train_loader, test_loader

from torchvision.models import vgg16, VGG16_Weights

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移动到 GPU 上
# model = CIFAR10_VGG('VGG16').to(device)

# 加载预训练的 VGG16 模型，使用 weights 参数
model = vgg16(weights=VGG16_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False
# 修改模型的全连接层
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 10)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    for i, (data, label) in enumerate(train_loader):
        # 将数据和标签移动到 GPU 上
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        predict = model(data)
        train_loss = criterion(predict, label)
        train_loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch {epoch} | Batch {i} | Loss: {train_loss.item():.4f}')


def train_test():
    model.eval()
    loss = 0.0
    correct_num = 0
    with torch.no_grad():
        for data, label in train_loader:
            # 将数据和标签移动到 GPU 上
            data, label = data.to(device), label.to(device)
            predict = model(data)
            loss += criterion(predict, label).item()
            y = predict.argmax(dim=1)
            correct_num += y.eq(label).sum().item()
    loss /= len(train_loader)
    accuracy = 100. * correct_num / len(train_loader.dataset)
    print(f'Loss:{loss} | accuracy{accuracy:.4f}')

def test():
    model.eval()
    correct_num=0
    with torch.no_grad():
        for data,label in test_loader:
            data,label=data.to(device),label.to(device)
            predict=model(data)
            y=predict.argmax(dim=1)
            correct_num+=y.eq(label).sum().item()
    accuracy=100.*correct_num/len(test_loader.dataset)
    print(f'accuracy{accuracy:.4f}')


for i in range(5):
    train(i)
    train_test()
    test()