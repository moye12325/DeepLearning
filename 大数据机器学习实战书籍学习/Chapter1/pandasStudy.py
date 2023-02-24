import pandas as pd
import numpy as np
# s1 = pd.Series(['zhangsan','lisi','wangwu'])
# print(s1)

b = pd.DataFrame([['zhangsan',2],[3,'4'],[5,6]],columns=['data','score'],dtype=str,index=['A','B','C'])
# print(b)

a = pd.DataFrame(np.arange(12).reshape(3,4),index=['A','B','C'],columns=['a','b','c','d'])
print(a)
# a.to_excel('write.xlsx')
#
# data = pd.read_excel('data.xlsx')
# print(data.head())
print("----------------------------------------")
data = a[1:3]
print(data)
print("----------------------------------------")
data = a[['c','d','a']]
print(data)
print("----------------------------------------")
data = a.iloc[1:3]
print(data)