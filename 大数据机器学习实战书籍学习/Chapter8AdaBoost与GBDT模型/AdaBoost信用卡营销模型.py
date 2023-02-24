import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_excel('信用卡精准营销模型.xlsx')
X = df.drop(columns='响应')
y = df['响应']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(random_state=123)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_pred, y_test)
print(score)

y_pred_proba = clf.predict_proba(X_test)
from sklearn.metrics import roc_curve

fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:, 1])

plt.plot(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score

score = roc_auc_score(y_test, y_pred_proba[:, 1])
print(clf.feature_importances_)

features = X.columns
importances = clf.feature_importances_
importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)
print(importances_df)
