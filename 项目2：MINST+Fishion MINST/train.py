'''
import torch.optim as optim
import torch
from Model import MNIST_CNN
import torch.nn as nn
from data_generator import train_loader,test_loader

model = MNIST_CNN()

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练循环
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}')


# 测试循环
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'测试集平均损失: {test_loss:.4f}, 准确率: {accuracy:.2f}%')


# 训练5个Epoch
for epoch in range(1, 6):
    train(epoch)
    test()

'''


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Model import Fashion_CNN
from data_generator import train_loader,test_loader

model=Fashion_CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
# 当验证集损失在 5 个 epoch 内没有下降时，学习率乘以 0.1
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        predict=model(data)
        loss=criterion(predict,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100==0:
            print(f'Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}')

def train_test():
    model.eval()
    correct=0
    train_loss=0
    with torch.no_grad():
        for data,target in train_loader:
            predict=model(data)
            train_loss+=criterion(predict,target).item()
            y=predict.argmax(dim=1)
            correct+=y.eq(target).sum().item()
    accuracy = 100.*correct/len(train_loader.dataset)
    print(f'accuracy{accuracy:.4f}')
    return train_loss



def test():
    model.eval()
    correct=0
    test_loss=0
    with torch.no_grad():
        for data, target in test_loader:
            predict = model(data)
            test_loss += criterion(predict, target).item()
            y = predict.argmax(dim=1)
            correct += y.eq(target).sum().item()
    test_loss/=len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'loss {test_loss} | accuracy {accuracy:.4f}')


for epoch in range(1,30):
    train(epoch)
    train_loss=train_test()
    scheduler.step(train_loss)
    test()
