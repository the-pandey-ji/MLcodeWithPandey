from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
print(iris.keys())
print(iris.data[0], iris.target[0])

features = iris.data
labels = iris.target
