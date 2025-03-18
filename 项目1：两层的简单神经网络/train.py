import torch
from Model import MyModel
import random

x = []
y = []
# 第二步：读取data,这里使用with，而不使用data=list(datafile)
# 将数据以list的形式存放在x和y中
with open(r'C:\Users\Liu\Desktop\项目1：两层的简单神经网络\data1.txt', 'r', encoding='utf-8') as datafile:
    for line in datafile:
        line = line.replace('\n', '')
        val = line.split(',')
        x.append([float(val[0]), float(val[1]), float(val[2])])
        y.append(int(val[3]))  # 转换为整数类型

# 数据提取完成之后，由于机器学习常用tensor变量存数据，因此我们要将其换成tensor型
x = torch.tensor(x)
y = torch.tensor(y)  # 转换为整数类型的张量

# 第三步:建立模型，确定各个参数
model = MyModel()
# 确定优化算法和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fc = torch.nn.CrossEntropyLoss()


# 第四步：开始训练
for i in range(200):
    y_pred = model(x)  # 直接调用模型实例
    loss = loss_fc(y_pred, y)  # 注意输入顺序
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 5 == 0:
        print(f'Epoch [{i + 1}/{200}], Loss: {loss.item():.4f}')



# 第五步：使用模型进行预测

# 测试train集的正确率
train_pre=model(x)
train_pred_labels = torch.argmax(train_pre, dim=1)  # 获取预测的类别索引
num1 = 0
for i in range(len(y)):
    if train_pred_labels[i] != y[i]:
        num1 += 1

print(num1)

# 测试集的正确率
num = 0
for _ in range(500):
    num1 = round(random.uniform(0, 1000), 2)
    num2 = round(random.uniform(0, 1000), 2)
    num3 = round(random.uniform(0, 1000), 2)
    label = 1 if num1 + num2 + num3 > 2000 else 0
    # max_val = max(num1, num2, num3)
    # label = 1 if max_val > 750 else 0
    line = torch.tensor([[num1, num2, num3]], dtype=torch.float32)  # 扩展为二维张量
    pred = model(line)
    pred_label = torch.argmax(pred, dim=1).item()  # 获取预测的类别索引
    if pred_label != label:
        num += 1

print(num)