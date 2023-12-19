import pickle

path = '../data/ours/our_meanstd.pk'
# 读取 .pk 文件
with open(path, 'rb') as f:
    data = pickle.load(f)

# 输出数据
print(data)