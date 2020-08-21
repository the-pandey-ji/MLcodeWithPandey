from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
print(iris.keys())
print(iris.data[0], iris.target[0])

features = iris.data
labels = iris.target

clf = KNeighborsClassifier()
clf.fit(features, labels)

prediction = clf.predict([[1, 1, 2, 3]])
print(prediction)
