import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import chi2
from sklearn.feature_selection import GenericUnivariateSelect

# 实验2 第（2）b步骤2  使用sklearn.feature_selection.GenericUnivariateSelec进行特征值选择

repairPath = "../实验数据/配网抢修数据.csv"
np.set_printoptions(precision=2, suppress=True)
row_tag = np.genfromtxt(repairPath, max_rows=1, dtype=str, delimiter=',', usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10))
print(row_tag)
min_max_scaler = preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(
    np.genfromtxt(repairPath, skip_header=1, delimiter=',', usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10)))
y_target = np.genfromtxt(repairPath, skip_header=1, delimiter=',', usecols=(11))
# 将target值按val/50分组划分
for i in range(len(y_target)):
    y_target[i] = int(y_target[i] / 50)
s = GenericUnivariateSelect(score_func=chi2, mode='k_best', param=2)
s.fit_transform(x_train_minmax, y_target)
print(s.scores_)
