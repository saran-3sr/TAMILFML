import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

data = pd.read_csv("Overrun.csv")
X = np.array(data)
K = 3
max_iters = 100
centroids = X[random.sample(range(len(X)), K)]
for i in range(max_iters):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    clusters = np.argmin(distances, axis=0)
    for k in range(K):
        centroids[k] = X[clusters == k].mean(axis=0)
plt.scatter(X[:, 0], X[:, 1], c=clusters)

plt.title("K-means Clustering")

plt.show()