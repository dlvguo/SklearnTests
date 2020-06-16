import numpy as np
from sklearn import preprocessing

eleRecPath = "./实验1数据/电费回收数据.csv"
# 设置numpy的小数点类型以及禁用科学计数法
np.set_printoptions(precision=2, suppress=True)
# usecols 代表取的列数
row_data = np.genfromtxt(eleRecPath, names=True, skip_header=1, delimiter=',', usecols=(2, 3, 4, 5, 8, 10, 11, 12, 13))
print(row_data.dtype)
print(row_data[:10])
