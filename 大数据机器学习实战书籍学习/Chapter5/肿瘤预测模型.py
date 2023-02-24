import pandas as pd

df = pd.read_excel('肿瘤数据.xlsx')

X = df.drop(columns='肿瘤性质')
y = df['肿瘤性质']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)
print(nb_clf.score(X_test,y_test))

