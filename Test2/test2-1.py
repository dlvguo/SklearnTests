import numpy as np

# 实验2 第（1）步骤 读取网抢修数据

repairPath = "../实验数据/配网抢修数据.csv"
np.set_printoptions(precision=2, suppress=True)
row_tag = np.genfromtxt(repairPath, max_rows=1, dtype=str, delimiter=',', usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
# 输出Tag
print(row_tag)
x_feature = np.genfromtxt(repairPath, skip_header=1, delimiter=',', usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
for i in range(10):
    print(x_feature.tolist()[i])
