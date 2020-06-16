import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 实验2 第（3）步骤1 使用随机森林回归

repairPath = "../实验数据/配网抢修数据.csv"
np.set_printoptions(precision=2, suppress=True)
min_max_scaler = preprocessing.MinMaxScaler()
time_start = time.time()
x_feature = min_max_scaler.fit_transform(
    np.genfromtxt(repairPath, skip_header=1, delimiter=',', usecols=(5, 6, 7, 8, 9)))
y_target = np.genfromtxt(repairPath, skip_header=1, delimiter=',', usecols=(11))
# 将target值按val/50分组划分
for i in range(len(y_target)):
    y_target[i] = int(y_target[i] / 50)
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x_feature, y_target.flatten(), test_size=0.4, random_state=0)
# 随机森林回归
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
print('mean_squared_error:', mean_squared_error(y_test, rfr.predict(x_test)))
e = mean_absolute_error(y_test, rfr.predict(x_test))
print('mean_absolute_error:', e)
# Sklearn无rae 根据mae计算
sum = 0.
for i in range(len(y_test)):
    if y_test[i] == 0:
        sum += 1
    else:
        sum += (e / y_test[i])
print('relative absolute error:', sum / len(y_test))
time_end = time.time()
print('运算时间 {:.2f}'.format(time_end - time_start), 's')
