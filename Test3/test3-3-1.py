import numpy as np
import time
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from Test3.S_Dbw import S_Dbw

# 实验3 第（3）步骤1 使用KMeans计算

mobilePath = "../实验数据/移动客户数据表.tsv"
np.set_printoptions(precision=2, suppress=True)
min_max_scaler = preprocessing.MinMaxScaler()
x_feature = min_max_scaler.fit_transform(np.genfromtxt(mobilePath, skip_header=1, delimiter='\t')[:, 4:])
selector = VarianceThreshold(0)
selector.fit(x_feature)
arr = np.argsort(-selector.variances_)
row_tag = np.genfromtxt(mobilePath, max_rows=1, dtype=str, delimiter='\t', usecols=arr[:20])
x_feature = min_max_scaler.fit_transform(np.genfromtxt(mobilePath, skip_header=1, delimiter='\t', usecols=arr[:20]))
time_start = time.time()
clf = KMeans(n_clusters=10)
clf.fit(x_feature)
print('聚类质量SSE:', clf.inertia_)
time_end = time.time()
print('聚类运算时间 {:.2f}'.format(time_end - time_start), 's')
print('Silhouette:', silhouette_score(x_feature, clf.predict(x_feature), metric='euclidean'))
s_dbw = S_Dbw(x_feature, clf.predict(x_feature))
print('S_Dbw', s_dbw.result())
