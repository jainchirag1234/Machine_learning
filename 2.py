import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
# Features: x1 and x2
X = np.array([[1, 2],
              [2, 3],
              [4, 5],
              [3, 6],
              [5, 8]])

# Target
y = np.array([2, 3, 4, 5, 6])

# Create model
model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Prediction
X_new = np.array([[6, 9]])
y_pred = model.predict(X_new)
print("Prediction for input [6, 9]:", y_pred[0])
