import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
from sklearn.neural_network import MLPClassifier

# 实验1 第（3）步骤3 使用神经网络计算

eleRecPath = "../实验数据/电费回收数据.csv"
# 设置numpy的小数点类型以及禁用科学计数法
np.set_printoptions(precision=2, suppress=True)
min_max_scaler = preprocessing.MinMaxScaler()
time_start = time.time()
x_feature = min_max_scaler.fit_transform(np.genfromtxt(eleRecPath, skip_header=1, delimiter=',', usecols=(8, 12)))
# 目标值标签
y_target = np.genfromtxt(eleRecPath, skip_header=1, delimiter=',', usecols=13)
# 切割数据集
x_train, x_test, y_train, y_test = train_test_split(x_feature, y_target, test_size=0.4, random_state=0)
# 神经网络计算
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1)
clf.fit(x_train, y_train)
target_names = ['未欠费', '欠费']
print(classification_report(y_test, clf.predict(x_test), target_names=target_names))
print('混淆矩阵如下:')
print(confusion_matrix(y_test, clf.predict(x_test)))
time_end = time.time()
print('运算时间 {:.2f}'.format(time_end - time_start), 's')
