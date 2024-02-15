import matplotlib.pyplot as plt

# 从文件中读取数据
with open("log/log.txt", "r", encoding='utf-8') as file:
    lines = file.readlines()

loss_values = []
entropy_values = []

# 提取loss和entropy的数值
for line in lines:
    if line.startswith('loss'):
        print(line)
        loss_values.append(float(line.split(',')[0].split(':')[1]))
        entropy_values.append(float(line.split(',')[1].split(':')[1]))

# 绘制曲线
plt.plot(loss_values, label='Loss')
plt.plot(entropy_values, label='Entropy')

plt.xlabel('Batch')
plt.ylabel('Value')
plt.title('Loss and Entropy Over Batches')
plt.legend()
plt.grid(True)

plt.savefig('log/log.png')

plt.show()