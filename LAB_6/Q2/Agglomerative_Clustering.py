
"""
Consider a dataset of your choice and implement agglomerative clustering algorithm
"""

import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# creating dataset to apply clustering on
X, y = make_blobs(centers=4, n_samples=15, n_features=3, shuffle=True, random_state=42)
X = np.array(X)

print("************************************** Agglomerative Clustering Algorithm *****************************************************\n")

def compute_distance(samples):
    # Creates a matrix of distances between individual samples and clusters attained at a particular step
    Distance_mat = np.zeros((len(samples), len(samples)))
    for i in range(Distance_mat.shape[0]):
        for j in range(Distance_mat.shape[0]):
            if i != j:
                Distance_mat[i, j] = float(distance_calculate(samples[i], samples[j]))
            else:
                Distance_mat[i, j] = 10 ** 4
    return Distance_mat

def intersampledist(s1, s2):
    if str(type(s2[0])) != '<class \'list\'>':
        s2 = [s2]
    if str(type(s1[0])) != '<class \'list\'>':
        s1 = [s1]
    m = len(s1)
    n = len(s2)
    dist = []
    if n >= m:
        for i in range(n):
            for j in range(m):
                if (len(s2[i]) >= len(s1[j])) and str(type(s2[i][0]) != '<class \'list\'>'):
                    dist.append(interclusterdist(s2[i], s1[j]))
                else:
                    dist.append(np.linalg.norm(np.array(s2[i]) - np.array(s1[j])))
    else:
        for i in range(m):
            for j in range(n):
                if (len(s1[i]) >= len(s2[j])) and str(type(s1[i][0]) != '<class \'list\'>'):
                    dist.append(interclusterdist(s1[i], s2[j]))
                else:
                    dist.append(np.linalg.norm(np.array(s1[i]) - np.array(s2[j])))
    return min(dist)

def interclusterdist( cl, sample):
    if sample[0] != '<class \'list\'>':
        sample = [sample]
    dist = []
    for i in range(len(cl)):
        for j in range(len(sample)):
            dist.append(np.linalg.norm(np.array(cl[i]) - np.array(sample[j])))
    return min(dist)

def distance_calculate(sample1, sample2):
    dist = []
    for i in range(len(sample1)):
        for j in range(len(sample2)):
            try:
                dist.append(np.linalg.norm(np.array(sample1[i]) - np.array(sample2[j])))
            except:
                dist.append(intersampledist(sample1[i], sample2[j]))
    return min(dist)


t = [[i] for i in range(X.shape[0])]
samples = [[list(X[i])] for i in range(X.shape[0])]
m = len(samples)

while m > 1:
    print('Current number of Clusters   :- ', m)
    Distance_mat = compute_distance(samples)
    sample_ind_needed = np.where(Distance_mat == Distance_mat.min())[0]
    value_to_add = samples.pop(sample_ind_needed[1])
    samples[sample_ind_needed[0]].append(value_to_add)
    print("Combining Clusters:  ")
    print(t[sample_ind_needed[0]]," and ", t[sample_ind_needed[1]])
    t[sample_ind_needed[0]].append(t[sample_ind_needed[1]])
    t[sample_ind_needed[0]] = [t[sample_ind_needed[0]]]
    v = t.pop(sample_ind_needed[1])
    m = len(samples)
    print("Current Status         :", t)
    print("Cluster attained:      :", t[sample_ind_needed[0]])
    print("Size after clustering  :", m)
    print('\n')
    
print("The algorithm has converged!!")
Z = linkage(X, 'single')
dn = dendrogram(Z)
plt.title("Agglomerative Clustering - Dendogram")
plt.show()
