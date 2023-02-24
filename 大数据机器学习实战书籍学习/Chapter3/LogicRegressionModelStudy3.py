import matplotlib.pylab as plt
import numpy as np

# 1、读取数据
import pandas as pd

df = pd.read_excel('股票客户流失.xlsx')

# 2、划分特征变量和目标变量
X = df.drop(columns='是否流失')
y = df['是否流失']

# 3、划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 4、模型搭建
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# 5、模型使用1：预测数据结果
y_pred = model.predict(X_test)
print(y_pred[:100])

# 6、模型使用2：预测概率
y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba)
