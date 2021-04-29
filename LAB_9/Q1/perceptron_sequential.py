import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

print("Perceptron Implementation using Python by Yukti Khurana")
# loading dataset
df = pd.read_csv("Iris.csv", usecols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])
df.columns = range(df.shape[1])
X = df.iloc[0:100, [0,2]].values

# Visualizing the dataset
plt.scatter(X[:50, 0], X[:50, 1], label = 'setosa',marker='x', color='purple')
plt.scatter(X[50:100, 0], X[50:100, 1], label = 'versicolor',color='green')
plt.title("Iris Dataset Visualization by Yukti Khurana")
plt.xlabel("SepalLengthCm")
plt.ylabel("PetalLengthCm")
plt.show()

y = df.iloc[0:100, 4].values
# class label for setosa flower = -1
# class label for versicolor flower = 1
y = np.where(y == 'Iris-setosa', -1, 1)

# Initialising the model parameters
learn_rate = 0.001
epochs = 50
errors = []
# initializing an array for weights which will get updated in each iteration
weights = np.zeros(1 + X.shape[1])

# function for summing the given matrix inputs and their corresponding weights.
def net_input(x, weights):
  return np.dot(x, weights[1:]) + weights[0]

# prediction function
def predict(x, weights):
  return np.where(net_input(x, weights) >= 0.0, 1, -1)

# model fitting
# creating an numpy array of the size of train data 
weights = np.zeros(1 + X.shape[1])
for i in range(epochs):
  err = 0
  for xi, target in zip(X,y):
    update = learn_rate * (target - predict(xi, weights))
    weights[1:] += update*xi
    weights[0] += update
    err += int(update != 0)
  errors.append(err)
print(errors)

# observing the drop in misclassification error after each epoch
plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

# Visualization of the perceptron model
resolution = 0.02
# setup marker generator and color map
markers = ('x', 'o')
colors = ('purple', 'green')
cmap = ListedColormap(colors[:len(np.unique(y))])

# plot the decision surface
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

Z = predict(np.array([xx1.ravel(), xx2.ravel()]).T, weights)
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

class_names = ['setosa','versicolor']
# plot class samples
for idx, cl in enumerate(np.unique(y)):
  plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.7, c=cmap(idx),marker=markers[idx], label=class_names[idx])

plt.title("Perceptron Model on Iris dataset")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()