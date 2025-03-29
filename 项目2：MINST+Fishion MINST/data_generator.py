'''
import torch
from torchvision import datasets, transforms

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

'''


import torch
from torchvision import datasets, transforms
'''
# 定义数据增强的转换操作
train_transform = transforms.Compose([
    # 随机旋转图像，旋转角度范围为 -10 到 10 度
    transforms.RandomRotation(10),
    # 随机水平翻转图像
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(0.286, 0.353)
])
'''

print(torch.cuda.is_available())


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.286, 0.353)
])

# 加载训练集并应用数据增强
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# 加载测试集，测试集通常不进行数据增强
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)




'''
# 定义数据转换，仅将图像转换为张量
transform = transforms.Compose([transforms.ToTensor()])

# 加载数据集
dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# 初始化变量
num_channels = 1  # 假设是 RGB 图像
total_sum = torch.zeros(num_channels)
total_squared_sum = torch.zeros(num_channels)
total_count = 0

# 遍历数据集
for data, _ in dataset:
    total_sum += torch.sum(data, dim=(1, 2))
    total_squared_sum += torch.sum(data ** 2, dim=(1, 2))
    total_count += data.size(1) * data.size(2)

# 计算均值和方差
mean = total_sum / total_count
std = torch.sqrt((total_squared_sum / total_count) - (mean ** 2))

print("Mean:", mean)
print("Std:", std)

'''