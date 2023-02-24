import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1,读取数据与数据预处理
from Chapter3.LogicRegressionModelStudy3 import y_pred_proba

df = pd.read_excel('股票客户流失.xlsx')

# 2,划分特征变量和目标变量
X = df.drop(columns='是否流失')
y = df['是否流失']

# 3,划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 4,模型搭建
model = LogisticRegression()
model.fit(X_train, y_train)

# 5,模型预测结果
y_pred = model.predict(X_test)
print(y_pred[:20])

a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head(10))

# 6,模型预测概率
# 两种查看准确率的方式
score = accuracy_score(y_pred, y_test)
print(score)
print(model.score(X_test, y_test))

# 7,模型使用-预测概率
y_pred_proba = model.predict_proba(X_test)
b = pd.DataFrame(y_pred_proba, columns=['不流失概率', '流失概率'])
print(b.head(10))

# P=1/(1+e^（-y）)   y=k1X+K2X+...
print(model.coef_)#K1 K2 K3等的值

