import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("Principal Component Analysis by Yukti Khurana")
df = pd.read_csv("Ionosphere.csv",header=None)
df = df[1:]
# split into training and testing sets
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
# Splitting the training and testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.70, random_state = 41, stratify = y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# finding the Covariance Matrix of training data - X
cov_mat = np.cov(X_train_std.T)
# Eigen values and Eigen vectors using the covariance matrix
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# calculate cumulative sum of explained variances
total_evals = sum(eigen_vals)
var_exp = [(i / total_evals) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Feature Extraction
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low (in decreasing order of eigen values)
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# we collect the two eigenvectors that correspond to the two largest eigenvalues, 
# to capture about 60% of the variance in this dataset
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
#print('Matrix W:\n', w)

# Using the projection matrix, we can now transform a sample x (represented as a 1 x 34-dimensional row vector) 
# onto the PCA subspace (the principal components one and two) obtaining x′
# now a two-dimensional sample vector consisting of two new features
# x' = xW
X_train_std[0].dot(w)

# we can transform the entire 245 X 34-dimensional training dataset 
# onto the two principal components by calculating the matrix dot product
# X' = XW
X_train_pca = X_train_std.dot(w)
print()
print("Training Dataset after reduction of Dimenstions : ")
print(X_train_pca)
# Hence the dimensions of dataset were reduced from 34 to 2
# visualizing the reduced Ionosphere training dataset
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], 
                X_train_pca[y_train==l, 1], 
                c=c, label=l, marker=m) 
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

print()
print("Dimensions Before PCA of Ionosphere Training dataset = ", X_train.shape)
print("Dimensions After PCA of Ionosphere Training dataset = ", X_train_pca.shape)