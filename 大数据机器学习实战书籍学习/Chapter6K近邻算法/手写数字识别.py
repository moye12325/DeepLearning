import pandas as pd
from networkx import neighbors
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

# 1、读取数据

df = pd.read_excel('手写字体识别.xlsx')

# 2、提取特征变量和目标变量
X = df.drop(columns='对应数字')
y = df['对应数字']

# 样本矩阵标准化处理
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)

# 3、划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 4、模型搭建
from sklearn.neighbors import KNeighborsClassifier as KNN

knn = KNN(n_neighbors=3)
knn.fit(X_train, y_train)

# 5、模型预测与评估
y_pred = knn.predict(X_test)

a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
score = accuracy_score(y_pred, y_test)

from PIL import Image

# 1、图片大小调整
img = Image.open('数字4.png')
img = img.resize((32, 32))
# img.show()
# 2、图片灰度处理
img = img.convert('L')
# 3、图片二值化处理
import numpy as np

img_new = img.point(lambda x: 0 if x > 128 else 1)
arr = np.array(img_new)

8

# 4、将二维数组转换为一维数组(1, 1024)
arr_new = arr.reshape(1, -1)
print(arr_new.shape)
answer = knn.predict(arr_new)
print("结果为：" + str(answer[0]))
