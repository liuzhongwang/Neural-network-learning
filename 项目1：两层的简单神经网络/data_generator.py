
import random

# 三种数据创建方式
# 第一种：当三个特征值的总和大于某个阈值时，标签设为 1；反之，标签设为 0。
with open(r'C:\Users\Liu\Desktop\项目1：两层的简单神经网络\data1.txt','w') as file:
    for _ in range(1000):
        num1=round(random.uniform(0,1000),2)
        num2 = round(random.uniform(0, 1000), 2)
        num3 = round(random.uniform(0, 1000), 2)
        label = 1 if num1+num2+num3>2000 else 0
        line=f"{num1},{num2},{num3},{label}\n"
        file.write(line)

# 第二种：若三个特征中的最大值大于某个阈值，标签设为 1；否则，标签设为 0。
with open(r'C:\Users\Liu\Desktop\项目1：两层的简单神经网络\data2.txt','w') as file:
    for _ in range(500):
        num1 = round(random.uniform(0, 1000),2)
        num2 = round(random.uniform(0,1000),2)
        num3 = round(random.uniform(0,1000),2)
        max_val = max(num1, num2, num3)
        label = 1 if max_val>750 else 0
        line = f"{num1},{num2},{num3},{label}\n"
        file.write(line)

# 第三种：若三个特征中奇数的个数为奇数，标签设为 1；否则，标签设为 0。
def is_odd(num):
    return int(num) % 2 != 0

with open(r'C:\Users\Liu\Desktop\项目1：两层的简单神经网络\data3.txt','w') as file:
    for _ in range(500):
        num1 = round(random.uniform(0, 1000),2)
        num2 = round(random.uniform(0,1000),2)
        num3 = round(random.uniform(0,1000),2)
        odd_count = sum([is_odd(num1), is_odd(num2), is_odd(num3)])
        label = 1 if odd_count % 2 != 0 else 0
        line = f"{num1},{num2},{num3},{label}\n"
        file.write(line)








