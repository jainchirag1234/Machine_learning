import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# OLS method formulas
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)

# Slope (m)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
m = numerator / denominator

# Intercept (c)
c = y_mean - m * x_mean

# Predict
y_pred = m * x + c

# Plot
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, y_pred, color='red', label='OLS Fitted Line')
plt.legend()
plt.title("OLS - Simple Linear Regression")
plt.show()

print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")
