import numpy as np
from sklearn import preprocessing

# 实验2 第（2）a步骤规范化数据

repairPath = "../实验数据/配网抢修数据.csv"
np.set_printoptions(precision=2, suppress=True)
row_tag = np.genfromtxt(repairPath, max_rows=1, dtype=str, delimiter=',', usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10))
print(row_tag)
min_max_scaler = preprocessing.MinMaxScaler()
x_feature = min_max_scaler.fit_transform(
    np.genfromtxt(repairPath, skip_header=1, delimiter=',', usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11)))
for i in range(10):
    print(x_feature.tolist()[i])
