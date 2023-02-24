import matplotlib.pylab as plt
import numpy as np

# 逻辑回归模型数学原理
X = np.linspace(-6, 6)
Y = 1.0 / (1.0 + np.exp(-X))
plt.plot(X, Y)
plt.show()
