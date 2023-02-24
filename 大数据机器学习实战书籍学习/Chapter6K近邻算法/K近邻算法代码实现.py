import pandas as pd

df = pd.read_excel('葡萄酒.xlsx')
X_train = df[['酒精含量(%)', '苹果酸含量(%)']]
y_train = df['分类']

from sklearn.neighbors import KNeighborsClassifier as KNN

knn = KNN(n_neighbors=3)
knn.fit(X_train, y_train)

X_test = [[2, 1], [4, 1], [8, 3]]
ans = knn.predict(X_test)
print(ans)
