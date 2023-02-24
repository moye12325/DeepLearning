import matplotlib.pylab as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x = [1, 2, 3]
y = ['A', 'B', 'C']

ax1 = plt.subplot(221)
plt.plot(x, y, color="red", linewidth=2)
plt.title('图1')
plt.xlabel("X")
plt.ylabel("Y")

ax2 = plt.subplot(222)
plt.bar(x, y, color="red", linewidth=2., )
plt.title('Test2')
plt.xlabel("X")
plt.ylabel("Y")

m = np.random.rand(10)
n = np.random.rand(10)

ax3 = plt.subplot(223)
plt.scatter(m, n, color="red", linewidth=2)
plt.title('Test3')
plt.xlabel("X")
plt.ylabel("Y")

x1 = np.array([1, 2, 3])
y1 = x1 + 1
y2 = x1 * 2
ax4 = plt.subplot(224)
plt.plot(x1, y1, label="y = x + 1")
plt.plot(x1, y2, color='red', linewidth=3, linestyle='--', label="y = 2x")
plt.legend(loc = 'upper left')
plt.title('Test4')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
