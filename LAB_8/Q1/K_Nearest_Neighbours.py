import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import matplotlib.pyplot as plt


# HELPER FUNCTIONS

def Euclidean_distance(a, b):
    # just in case, if the instances are lists or tuples:
    a = np.array(a)
    b = np.array(b)
    #ed = np.sqrt(np.sum((a - b)**2)) # or ed = np.linalg.norm(a - b)
    ed = np.linalg.norm(a - b)
    return ed

def get_K_neighbours(X_train, y_train, test_sample, k):
    # calculate the distances of the test sample from each training sample
    distances = [Euclidean_distance(x, test_sample) for x in X_train]

    # sort the distances and get the k neighbours with smallest distance
    k_nearest = np.argsort(distances)[:k]

    # get the class labels of all k-nearest neighbours
    k_nearest_labels = [y_train[i] for i in k_nearest]
    return k_nearest_labels

def get_majority_class(k_nearest_labels):
    # select the most frequent class labels amongst the k-nearest neighbours of the test sample
    most_common_class = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common_class

def knn_predict(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_sample in X_test:
        k_nearest_labels = get_K_neighbours(X_train, y_train,test_sample, k)
        prediction = get_majority_class(k_nearest_labels)
        y_pred.append(prediction)
    return np.array(y_pred)

def sklearn_knn_algo(X_train, y_train, X_test, k=3):
    # using the sklearn library for comparison
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred

def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    a =  correct / float(len(actual)) * 100.0
    return round(a,3)



# Loading the Dataset- iris
dataset = datasets.load_iris()
target_iris_names =  list(dataset.target_names)
X = dataset.data  # input features
y = dataset.target # target features

# Stratified split of training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, random_state = 41, stratify = y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initializing the count of each of all classes of flowers - Versicolor,Setosa,Virginica
# let the total number of classes be 'C'
C = len(np.unique(np.array(y_test)))

# set the total number of features/ dimensions
M = X.shape[1]

print("Number of Classes = ",C)
print("Number of Features = ",M)

# K-NEAREST NEIGHBOURS ALGORITHM
# set the value of k
k = 3
# predicting the class labels for test data
y_pred = knn_predict(X_train, y_train, X_test, k)
my_accuracy = accuracy(y_test, y_pred)

# Running Knn sklearn library function for comparison
sklearn_ypred = sklearn_knn_algo(X_train, y_train, X_test, k)
sklearn_accuracy = accuracy(y_test, sklearn_ypred)

# Comparing Results
print("Predicted Class Labels by My KNN = \n" + str(y_pred))
print()
print("Predicted Class Labels by Sklearn KNN = \n" + str(sklearn_ypred))
print()
print("My KNN Accuracy  = ",my_accuracy, "%")
print("Sklearn library KNN Accuracy = ", sklearn_accuracy, "%")
print("\n")

"""
for i in range(len(y_pred)):
    print("Test data = ", X_test[i])
    print("Class label = ", y_pred[i])
    print(i," Predicted Value = iris ",target_iris_names[int(y_pred[i])])
"""

# plotting the accuracy vs k values
k_values = [i for i in range(2,15)]
y_preds = []
accuracies = []
for i in range(2,15):
    cur_pred = knn_predict(X_train, y_train, X_test, i)
    y_preds.append(cur_pred)
    a = accuracy(y_test, cur_pred)
    accuracies.append(a)
print(accuracies)
plt.plot(k_values, accuracies)
plt.xlabel("k-value")
plt.ylabel("Accuracy")
plt.show()