import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

# 实验1 第（2）b步骤1  使用sklearn.feature_selection.VarianceThreshold进行特征值选择

eleRecPath = "../实验数据/电费回收数据.csv"
# 设置numpy的小数点类型以及禁用科学计数法
np.set_printoptions(precision=2, suppress=True)
# usecols 代表取的列数
row_tag = np.genfromtxt(eleRecPath, max_rows=1, dtype=str, delimiter=',', usecols=(2, 3, 4, 5, 8, 10, 11, 12, 13))
# 输出数据Tag
print(row_tag)
row_data = np.genfromtxt(eleRecPath, skip_header=1, delimiter=',', usecols=(2, 3, 4, 5, 8, 10, 11, 12, 13))
# 规范化数据并计算方差
min_max_scaler = preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(row_data.tolist())
selector = VarianceThreshold(0)
selector.fit(x_train_minmax)
print('Variances is %s' % selector.variances_)
