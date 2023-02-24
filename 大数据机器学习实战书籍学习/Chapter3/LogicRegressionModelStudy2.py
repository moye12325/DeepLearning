import matplotlib.pylab as plt
import numpy as np

# 逻辑回归模型数学原理
X = [[1, 0], [5, 1], [6, 4], [4, 2], [3, 2]]
Y = [0, 1, 1, 0, 0]
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, Y)
print(model.predict([[2, 2]]))
