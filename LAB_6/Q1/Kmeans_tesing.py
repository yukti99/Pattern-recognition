import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from Kmeans_clustering import KMeans

print("\n************************************KMEANS CLUSTERING BY YUKTI KHURANA****************************")
sample_no = 500
center_no = 4
feature_no = 2
X, y = make_blobs(centers=center_no , n_samples=sample_no, n_features=feature_no, shuffle=True, random_state=42)
print("Number of samples = ", sample_no)
print("Number of features = ", feature_no)
print("Number of cluster centers/k = ", center_no)

clusters_len = len(np.unique(y))
k = KMeans(K=clusters_len, max_iters=150, plot_steps=True)
y_pred = k.predict(X)
print("\nThe class assigned to each datapoint : ")
print(y_pred)
































