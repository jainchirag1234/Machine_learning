import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 1.3, 3.75, 2.25])

# Initialize parameters
m = c = 0
L = 0.01  # Learning rate
epochs = 1000

n = float(len(x))  # Number of elements

# Gradient Descent
for i in range(epochs):
    y_pred = m * x + c
    D_m = (-2/n) * sum(x * (y - y_pred))
    D_c = (-2/n) * sum(y - y_pred)
    m = m - L * D_m
    c = c - L * D_c

# Prediction
y_final = m * x + c

# Plot
plt.scatter(x, y, color="blue")
plt.plot(x, y_final, color="red")
plt.title("Line Fit using Gradient Descent")
plt.show()

print(f"Learned Slope: {m}")
print(f"Learned Intercept: {c}")
