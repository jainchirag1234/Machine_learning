import numpy as np
import matplotlib.pyplot as plt

# Euclidean distance function
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# K-Means algorithm
def kmeans(X, k, max_iters=100):
    # Step 1: Initialize centroids randomly
    np.random.seed(42)  # for reproducibility
    random_indices = np.random.choice(len(X), size=k, replace=False)
    centroids = X[random_indices]

    for _ in range(max_iters):
        # Step 2: Assign clusters
        clusters = [[] for _ in range(k)]
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

        # Step 3: Update centroids
        new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroids[i]
                                  for i, cluster in enumerate(clusters)])

        # Step 4: Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, clusters


# Example usage
if __name__ == "__main__":
    # Generate some random data points
    X = np.random.rand(100, 2)  # 100 points in 2D space

    # Apply K-Means
    k = 3
    centroids, clusters = kmeans(X, k)

    # Plot the clusters
    colors = ['r', 'g', 'b']
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='y', marker='*', s=200, label='Centroids')
    plt.title("K-Means Clustering")
    plt.legend()
    plt.show()
