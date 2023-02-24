import matplotlib.pyplot as plt
import numpy as np
import pandas

# 读取excel
df = pandas.read_excel("IT行业收入表.xlsx")
# 显示前五行
# print(df.head())

# excel为二维数据，故X需写成二维
X = df[["工龄"]]
Y = df[["薪水"]]
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.subplot(211)
plt.title("IT行业一元二次回归模型")
plt.scatter(X, Y)
plt.xlabel("工龄")
plt.ylabel("薪水")
# plt.show()

from sklearn.linear_model import LinearRegression  # 引入线性回归模块
from sklearn.preprocessing import PolynomialFeatures  # 引入多次项内容模块

poly_reg = PolynomialFeatures(degree=2)  # 设置最高次项为2
X_ = poly_reg.fit_transform(X)  # 将原有的X转为一个新的二维数组X_,包含无意义的常数项、一次项、二次项系数

regr = LinearRegression()  # 构建一个初始的线性回归模型
regr.fit(X_, Y)  # fit()函数完成函数模型搭建

# 系数类型为numpy.ndarray，将其保留两位小数
# around保留两位小数，tolist转为list类型
print(regr.coef_)
list1 = regr.coef_[0]

a1 = np.around(list1[1], 2)
a2 = np.around(list1[2], 2)

print(a1)
print(a2)

b = np.around(regr.intercept_[0], 2)

str1 = "函数为：Y = " + str(a1) + "X\u00b2 + " + str(a2) + "X +" + str(b)
print(str1)

plt.subplot(212)  # 两行一列的第二列
plt.title("图2：一元二次函数")
plt.scatter(X, Y)
plt.plot(X, regr.predict(X_), color="red", label=str1)
plt.legend(loc='upper right')
plt.xlabel("工龄")
plt.ylabel("薪水")
plt.tight_layout()
plt.show()
