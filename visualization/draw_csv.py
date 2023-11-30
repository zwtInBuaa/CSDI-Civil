import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('../data/ours/re15.csv')

# 将'Datetime'列转换为日期时间格式
data['Datetime'] = pd.to_datetime(data['Datetime'])

# 设置图形大小
plt.figure(figsize=(10, 6))

# 遍历每个特征列并绘制折线图
for column in data.columns[1:]:
    plt.plot(data['Datetime'], data[column], label=column)

# 设置图例和标签
plt.legend()
plt.xlabel('Datetime')
plt.ylabel('Feature Value')

# 根据需要自定义标题等

# 显示图形
plt.show()