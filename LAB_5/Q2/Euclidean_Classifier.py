"""
EUCLIDEAN DISTANCE CLASSIFIER

The optimal Bayesian classifier is significantly simplified
under the following assumptions:

 The classes are equiprobable.
 The data in all classes follow Gaussian distributions.
 The covariance matrix is the same for all classes.
 The covariance matrix is diagonal and all elements
across the diagonal are equal.
That is, C = σ2I, I is the identity matrix.

"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

print("EUCLIDEAN CLASSIFIER FOR IRIS DATASET BY YUKTI KHURANA\n")
# using covarience formula
def getCovarience(x, mean_point, M):
    N = len(x)
    # initialise the matrix with zeroes
    cov = np.zeros(shape = (M, M))
    for i in range(N):
        xim = np.matrix(x[i] - mean_point)
        cov += np.matmul(xim.T, xim)
    cov /= (N-1)
    return cov

# here no of features = 4, so dimensions M = 4
# Gaussian Theory Formula is used
def get_prob(sample, mean):
    var = (sample-mean)
    dist = (np.matmul(var, var.T))**0.5
    return dist

def accuracy(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# loading the iris dataset
iris = datasets.load_iris()
target_iris_names =  list(iris.target_names)

# defining training data
X = iris.data   # input features
y = iris.target  # target features


# STRATIFIED SPLIT TO TRAIN AND TEST DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, random_state = 41, stratify = y)

# Initializing the count of each of all classes of flowers - Versicolor,Setosa,Virginica
# let the total number of classes be 'C'
#C = len(target_iris_names)
C = len(np.unique(np.array(y_test)))

# set the total number of features/ dimensions
M = X.shape[1]

print("Number of Classes = ",C)
print("Number of Features = ",M)

# count of each class
class_cnts = [0]*C

for i in range(C):
    # if y_train is 0 means its class 1
    class_cnts[i] = np.count_nonzero(y_train == i)

# to get a matrix with class count number of rows and 4 number of columns because there are 4 number of features
xmatrices = []
for  i in range(C):
    x = np.zeros(shape = (class_cnts[i], M), dtype = float)
    xmatrices.append(x)

# index of three classes
class_indices = [0]*C

for i in range(len(X_train)):
    # do for all classes
    for c in range(C):
        if (y_train[i] == c):
            xmatrices[c][class_indices[c]] = X_train[i].tolist()
            class_indices[c]+=1



# finding the mean of the all classes
class_means = []
for i in range(C):
    class_means.append(np.mean(xmatrices[i], axis = 0))


# getting the covarience matrix of each class
class_covs = []
for i in range(C):
    class_covs.append( getCovarience(xmatrices[i], class_means[i], M) )

# checking if the covariance is same for all classes or not
if (np.min(class_covs) == np.max(class_covs)):
    print("The covarience matrix is same for all three classes!")
#else:
    #print("The covarience matrix is NOT same for classes!\n\n")

#  calculating the class probabilities
class_probs = []
for i in range(C):
    class_probs.append( len(xmatrices[i]) / len(X_train) )


# declare the y_prediction np array of the length of test
y_pred = np.zeros(len(X_test))


# Testing
data_classes = [i for i in range(C)]


for z in range(len(X_test)):
    class_post_probs = []
    i = X_test[z]
    # calculate the probabilities for each class
    for c in range(C):
        cur_prob = get_prob(i, class_means[c])
        class_post_probs.append(cur_prob)
    # finding the min distance for mahalonobis
    max_prob = min(class_post_probs)
    for c in range(C):
        if (max_prob == class_post_probs[c]):
            y_pred[z] = data_classes[c]
            break

print("Predicted Class Labels for given Dataset = \n" + str(y_pred))
print()

for i in range(len(y_pred)):
    print("Test data = ", X_test[i])
    print(i," Predicted Value = iris ",target_iris_names[int(y_pred[i])])


print("\nAccuracy of model = ",round(accuracy(y_test, y_pred),3), "%")
