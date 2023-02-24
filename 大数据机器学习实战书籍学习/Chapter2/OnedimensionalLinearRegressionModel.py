import matplotlib.pyplot as plt
import numpy as np
import pandas

# 读取excel
df = pandas.read_excel("金融行业收入表.xlsx")
# 显示前五行
print(df.head())

# excel为二维数据，故X需写成二维
X = df[["工龄"]]
Y = df[["薪水"]]
ax1 = plt.subplot(211)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.title("工龄-薪水一元一次回归模型")
plt.scatter(X, Y)
plt.xlabel("工龄")
plt.ylabel("薪水")
# plt.show()

from sklearn.linear_model import LinearRegression  # 引入线性回归模块

regr = LinearRegression()  # 构建一个初始的线性回归模型
regr.fit(X, Y)  # fit()函数完成函数模型搭建

# 系数类型为numpy.ndarray，将其保留两位小数
# around保留两位小数，tolist转为list类型
a = np.around(regr.coef_[0], 2).tolist()

print(type(a[0]))
# print(a[0])
b = np.around(regr.intercept_[0], 2)
# print(type(b))
# print(b)

print("函数为：Y = " + str(a[0]) + "X + " + str(b))
str1 = "函数为：Y = " + str(a[0]) + "X + " + str(b)
# print("函数为：Y = " + str(regr.coef_[0]) + "X + " + str(regr.intercept_))

# print(a + "-------" + b)
# 打印截距

# print(type(regr.intercept_))
# print(type(regr.coef_[0]))
print("系数a:" + str(regr.coef_[0]))
print("截距b：" + str(regr.intercept_))

plt.subplot(212)  # 两行一列的第二列
plt.title("图2：一元函数")
plt.scatter(X, Y)
plt.plot(X, regr.predict(X), color="red", label=str1)
plt.legend(loc='upper right')
plt.xlabel("工龄")
plt.ylabel("薪水")
plt.tight_layout()
plt.show()
