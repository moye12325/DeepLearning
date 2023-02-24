# 朴素贝叶斯模型原理
from sklearn.naive_bayes import GaussianNB

X = [[1, 2], [5, 3], [6, 0], [3, 4], [1, 6]]
y = [1, 0, 1, 1, 0]

model = GaussianNB()

model.fit(X, y)
print(model.predict([[3, 4], [1, 6]]))
