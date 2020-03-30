import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
# data kemaren
dataset = make_blobs(
    n_samples=150, n_features=2,
    centers=3, cluster_std=0.5,
    shuffle=True, random_state=0
)
points = dataset[0]


#plt.scatter(dataset[0][0:, 0], dataset[0][:, 1])

# perform the clustering
hc = AgglomerativeClustering(
    n_clusters=3, affinity="euclidean", linkage='ward')

y_hc = hc.fit_predict(points)

plt.scatter(points[y_hc == 0, 0], points[y_hc == 0, 1], s=100, c='lightgreen')
plt.scatter(points[y_hc == 1, 0], points[y_hc == 1, 1], s=100, c='black')
plt.scatter(points[y_hc == 2, 0], points[y_hc == 2, 1], s=100, c='blue')
plt.show()
