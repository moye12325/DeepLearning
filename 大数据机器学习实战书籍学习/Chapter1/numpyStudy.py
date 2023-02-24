import numpy as np
a = [1,2,3,4]
b = np.array([1,2,3,4])
print(a)
print(b)
print(type(a))
print(type(b))

#array和list的区别
c = a*2
d = b*2
print(c)
print(d)

#创建二维数组
f = np.array([[1,2],[3,4],[4,4]])
print(type(f))

g = np.arange(12).reshape(3,4)
print(g)
h = np.random.randint(1,8,(2,9),int)
print(h)