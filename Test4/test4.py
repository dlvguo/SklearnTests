import numpy as np
from apyori import apriori
from sklearn import preprocessing

# 实验4 利用Apriori进行关联规则分析 采用Apyori库 使用 pip install apyori 下载类库

repairPath = "../实验数据/配网抢修数据.csv"
np.set_printoptions(precision=2, suppress=True)
min_max_scaler = preprocessing.MinMaxScaler()
x_feature = min_max_scaler.fit_transform(
    np.genfromtxt(repairPath, skip_header=1, delimiter=',', usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11)))
# 数据规范化为0-9
for i in range(len(x_feature)):
    for j in range(10):
        x_feature[i][j] = int(x_feature[i][j] * 10)
# 这边设置可信度0.1-0.9
res = apriori(transactions=x_feature, min_confidence=0.1)
for rule in res:
    print(str(rule))
