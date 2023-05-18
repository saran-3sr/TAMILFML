import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv("Overrun.csv")

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3, random_state=0).fit(data_scaled)

data["cluster"] = kmeans.labels_

plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data["cluster"])
plt.xlabel("TV")
plt.ylabel("Sales")
plt.title("K-means Clustering")
plt.show()