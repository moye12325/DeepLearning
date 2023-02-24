import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
# 1、数据读取与预处理
from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel('员工离职预测模型.xlsx')
df = df.replace({'工资': {'低': 0, '中': 1, '高': 2}})
print(df.head(20))

# 2、提取特征变量和目标变量
X = df.drop(columns='离职')
y = df['离职']

# 3、划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 4、模型训练及搭建
model = DecisionTreeClassifier(max_depth=3, random_state=123)
model.fit(X_train, y_train)

# 5、模型预测于评估
print(model.score(X_test, y_test))

y_pred_proba = model.predict_proba(X_test)
a = pd.DataFrame(y_pred_proba, columns=['离职概率', '不离职概率'])
fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:, 1])
b = pd.DataFrame()
b['阈值'] = list(thres)
b['假警报率'] = list(fpr)
b['命中率'] = list(tpr)
print(b.head(20))

plt.plot(fpr, tpr)
plt.show()
score = roc_auc_score(y_test, y_pred_proba[:, 1])
print(score)

# 查看各个特征的重要性
features = X.columns  # 获取特征名称
importances = model.feature_importances_  # 获取特征重要性
# 以二维表格形式展示
importances_df = pd.DataFrame()
importances_df['特征重要性'] = importances
importances_df['特征名称'] = features
importances_df.sort_values('特征重要性', ascending=False)
print(importances_df)

# 6、决策树模型可视化
import os
from sklearn.tree import export_graphviz
import graphviz

os.environ['PATH'] = os.pathsep + r'C:\Program Files\Graphviz\bin'
dot_data = export_graphviz(model, out_file=None, class_names=['0', '1'])
graph = graphviz.Source(dot_data)
graph.render("result1")

# 7、参数调优---K折交叉验证与GridSearch网格搜索
# 多参数调优
parameters = {'max_depth': [4, 5, 6, 7, 8, 9, 10, 11],
              'criterion': ['gini', 'entropy'],
              'min_samples_split': np.arange(10, 40, 3),
              }
model2 = DecisionTreeClassifier()
grid_search = GridSearchCV(model2, parameters, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

print(type(grid_search.best_estimator_))
print(grid_search.best_score_)
print(grid_search.best_index_)

model3 = grid_search.best_estimator_
# model3.fit(X_train, y_train)
os.environ['PATH'] = os.pathsep + r'C:\Program Files\Graphviz\bin'
dot_data = export_graphviz(model3, out_file=None, class_names=['0', '1'])
graph = graphviz.Source(dot_data)
graph.render("result2")
