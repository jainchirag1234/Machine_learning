# Q6: KNN without using library
import numpy as np
from collections import Counter

# Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Custom KNN class
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Calculate distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get k nearest labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example dataset
X_train = np.array([[1,2],[2,3],[3,4],[6,7],[7,8],[8,9]])
y_train = np.array([0,0,0,1,1,1])
X_test = np.array([[5,5]])

# Train and predict
clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("Prediction for test data:", predictions)
