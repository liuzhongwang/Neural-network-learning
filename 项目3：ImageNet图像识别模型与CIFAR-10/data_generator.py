import torch
from torchvision import datasets,transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图片尺寸调整为 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图片尺寸调整为 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.4942, 0.4851, 0.4504),(0.2467, 0.2429, 0.2616))
])

train_set = datasets.CIFAR10(
    root = './root',
    train = True,
    download = True,
    transform = train_transform
)
test_set = datasets.CIFAR10(
    root = './root',
    train = False,
    download = True,
    transform = test_transform
)

train_loader = torch.utils.data.DataLoader(train_set,32,True)
test_loader = torch.utils.data.DataLoader(test_set,32,False)