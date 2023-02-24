from sklearn.ensemble import AdaBoostClassifier

X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 0, 0, 1, 1]
model = AdaBoostClassifier(random_state=123)
model.fit(X, y)
print(model.predict([[5, 5]]))

from sklearn.ensemble import AdaBoostRegressor

X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [1, 2, 3, 4, 5]
model1 = AdaBoostRegressor(random_state=123)
model1.fit(X, y)
print(model1.predict([[5, 5]]))
