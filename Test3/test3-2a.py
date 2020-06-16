import numpy as np
from sklearn import preprocessing

# 实验3 第（2）a步骤规范化数据

mobilePath = "../实验数据/移动客户数据表.tsv"
np.set_printoptions(precision=2, suppress=True)
row_tag = np.genfromtxt(mobilePath, max_rows=1, dtype=str, delimiter='\t')
print(row_tag[4:])
min_max_scaler = preprocessing.MinMaxScaler()
x_feature = min_max_scaler.fit_transform(np.genfromtxt(mobilePath, skip_header=1, delimiter='\t')[:, 4:])
print(x_feature[:10])
