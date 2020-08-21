import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])

X_train = X
X_test = X

y_train = np.array([3,2,4,6,7,8,9,8,9])
y_test = np.array([3,2,4,6,7,8,9,8,9])

model = linear_model.LinearRegression()

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

print("Mean squared error is: ", mean_squared_error(y_test, y_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_predicted)

plt.show()

# Mean squared error is:  3035.0601152912695
# Weights:  [941.43097333]
# Intercept:  153.39713623331698
