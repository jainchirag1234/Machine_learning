# KNN Algorithm without using library
# Only basic Python (no numpy, no sklearn)

import math
from collections import Counter

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1) - 1):  # exclude the label
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# Function to get K nearest neighbors
def get_neighbors(training_data, test_instance, k):
    distances = []
    for train_instance in training_data:
        dist = euclidean_distance(test_instance, train_instance)
        distances.append((train_instance, dist))
    distances.sort(key=lambda x: x[1])  # sort by distance
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

# Function to predict the class
def predict_classification(training_data, test_instance, k):
    neighbors = get_neighbors(training_data, test_instance, k)
    output_labels = [row[-1] for row in neighbors]  # take last column as label
    prediction = Counter(output_labels).most_common(1)[0][0]
    return prediction

# Main
if __name__ == "__main__":
    # Example dataset [feature1, feature2, label]
    dataset = [
        [2.7, 2.5, 0],
        [1.3, 3.1, 0],
        [3.5, 4.2, 0],
        [8.2, 7.8, 1],
        [7.5, 8.3, 1],
        [9.1, 6.9, 1]
    ]

    test_sample = [7.0, 7.5]  # test data (without label)
    k = 3  # number of neighbors to check

    prediction = predict_classification(dataset, test_sample, k)
    print(f"Test Sample: {test_sample}")
    print(f"Predicted Class: {prediction}")
