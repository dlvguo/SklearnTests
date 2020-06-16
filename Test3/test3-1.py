import numpy as np

# 实验3 第（1）步骤 读取移动客户数据

mobilePath = "../实验数据/移动客户数据表.tsv"
# 设置numpy的小数点类型以及禁用科学计数法
np.set_printoptions(precision=2, suppress=True)
# usecols 代表取的列数 (2, 3, 4, 5, 8, 10, 11, 12)
row_tag = np.genfromtxt(mobilePath, max_rows=1, dtype=str, delimiter='\t')
print(row_tag[4:])
x_feature = np.genfromtxt(mobilePath, skip_header=1, delimiter='\t')[:, 4:]
for i in range(10):
    print(x_feature.tolist()[i])
