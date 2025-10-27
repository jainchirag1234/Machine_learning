import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])  # Clearly y = x^2 pattern

# Convert to polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# Fit model
model = LinearRegression()
model.fit(x_poly, y)

# Prediction
x_range = np.linspace(0, 6, 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_pred = model.predict(x_range_poly)

# Plot
plt.scatter(x, y, color='blue', label='Original')
plt.plot(x_range, y_pred, color='green', label='Polynomial Fit')
plt.title("Polynomial Curve Fitting")
plt.legend()
plt.show()

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
