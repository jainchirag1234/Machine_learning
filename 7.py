# Q7: Agglomerative Clustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# Create dataset
X, y = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)

# Apply Agglomerative Clustering
clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = clustering.fit_predict(X)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title("Agglomerative Clustering")
plt.show()
