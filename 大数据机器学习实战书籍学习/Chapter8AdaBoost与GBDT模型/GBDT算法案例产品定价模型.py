import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('产品定价模型.xlsx')

# 分类统计图书类别
a = df['类别'].value_counts()
b = df['彩印'].value_counts()
c = df['纸张'].value_counts()
print(a)
print(b)
print(c)
le = LabelEncoder()
df['类别'] = le.fit_transform(df['类别'])
df['类别'] = df['类别'].replace({'办公类': 0, '技术类': 1, '教辅类': 2})
le = LabelEncoder()
df['纸张'] = le.fit_transform(df['纸张'])

print(df['类别'].value_counts())
print(df['纸张'].value_counts())
print(df['彩印'].value_counts())

X = df.drop(columns='价格')
y = df['价格']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
model = GradientBoostingRegressor(random_state=123)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

d = pd.DataFrame()
d['预测值'] = list(y_pred)
d['实际值'] = list(y_test)
# print(d.head(20))

score = model.score(X_test, y_test)
from sklearn.metrics import r2_score

r2_score = r2_score(y_test, model.predict(X_test))
print(score)
print(r2_score)

features = X.columns
importances = model.feature_importances_
importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)
print(importances_df)
