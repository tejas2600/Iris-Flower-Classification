from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np

df = load_iris()

X_train, X_test, y_train, y_test = train_test_split(df['data'], df['target'])

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("target predicted {}".format(y_pred))
print("target predicted Name {}".format(df['target_names'][y_pred]))
accuracy=clf.score(X_test, y_test)
print("KNN Accuracy (k=3) : {}".format(accuracy))

#(clf.predict([[4.4, 2.9,4.6,1.0]]))



