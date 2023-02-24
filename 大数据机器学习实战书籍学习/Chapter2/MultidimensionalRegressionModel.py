import numpy as np
import pandas

df = pandas.read_excel("客户价值数据表.xlsx")
X = df[['历史贷款金额', '贷款次数', '学历', '月收入', '性别']]
Y = df['客户价值']

from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(X, Y)

list1 = regr.coef_.tolist()

b = [float('{:.3f}'.format(i)) for i in list1]

str2 = []
for i in range(len(b)):
    str2.append(str(b[i]) + "X" + str(i + 1))
print(str2)
# str1 = "Y = "+b[0]
print(type(str2))
str3 = ""
# print(type(regr.intercept_))
for i in range(len(str2)):
    str3 = str3 + " + " + str(str2[i])
str4 = "Y = " + str3[2:]
print(str4)

# 引入相关性评估
import statsmodels.api as sm

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2).fit()
print(est.summary())
