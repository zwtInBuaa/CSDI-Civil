# import pandas as pd
#
# # 读取CSV文件
# data = pd.read_csv('re15.csv')
#
# # 删除"datetime"列
# data = data.drop('Datetime', axis=1)
#
# # 保存处理后的数据到新的CSV文件
# data.to_csv('new.txt', index=False)

# 读取文本文件
with open('new.txt', 'r') as file:
    lines = file.readlines()

# 翻转数据
transposed_lines = []
for line in lines:
    elements = line.strip().split(',')
    transposed_lines.append(elements)

transposed_data = []
for i in range(len(transposed_lines[0])):
    transposed_row = [transposed_lines[j][i] for j in range(len(transposed_lines))]
    transposed_data.append(','.join(transposed_row))

# 将翻转后的数据写入新文件
with open('transposed_file.txt', 'w') as file:
    file.write('\n'.join(transposed_data))