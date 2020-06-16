import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

# 实验3 第（2）b步骤 使用sklearn.feature_selection.VarianceThreshold进行特征值选择

mobilePath = "../实验数据/移动客户数据表.tsv"
# 设置numpy的小数点类型以及禁用科学计数法
np.set_printoptions(precision=2, suppress=True)
# usecols 代表取的列数 (2, 3, 4, 5, 8, 10, 11, 12)
row_tag = np.genfromtxt(mobilePath, max_rows=1, dtype=str, delimiter='\t')
print(row_tag[4:])
min_max_scaler = preprocessing.MinMaxScaler()
x_feature = min_max_scaler.fit_transform(np.genfromtxt(mobilePath, skip_header=1, delimiter='\t')[:, 4:])
selector = VarianceThreshold(0)
selector.fit(x_feature)
# 对特征值排序
arr = np.argsort(-selector.variances_)
# 输出分数前20个特征值
for i in range(20):
    print(row_tag[arr[i]], ':', selector.variances_[arr[i]])
