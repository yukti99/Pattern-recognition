import csv
import numpy as np
import math

# Dataset filenames
TrainFile = "train-1.csv"  # or "train-2.csv"
TestFile = "test-1.csv"    # or "test-2.csv"

# ----------------------------------HELPER FUNCTIONS--------------------------------------------------------------------
# Load data from csv file
def load_csv(filename):
    lines = csv.reader(open(filename, "r", encoding='utf-8-sig'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    dataset = np.asarray(dataset, dtype=np.float32)
    return dataset


# Seperate data by class
def class_sorted_data(dataset):
    classes = np.unique(dataset[:, np.size(dataset, 1) - 1])
    sortedclassdata = []
    for i in range(len(classes)):
        item = classes[i]
        itemindex = np.where(dataset[:, np.size(dataset, 1) - 1] == item)   # index  of rows with label class[i]
        singleclassdataset = dataset[itemindex, 0:np.size(dataset, 1) - 1]  # array  of data for class[i]
        sortedclassdata.append(np.matrix(singleclassdataset))               # matrix of data for class[i]
    return sortedclassdata, classes


# posterior prob = likelihood * prior probability
# function to calculate prior probability
def prior_prob(dataset, sortedclassdata):
    priorprob = []
    for i in range(len(sortedclassdata)):
        priorprob.append(len(sortedclassdata[i])/len(dataset))
    return priorprob


# function to find mean and covariance, so that likelihood can be calculated
def find_mean(sortedclassdata):
    classmeans = []
    for i in range(len(sortedclassdata)):
        classmeans.append(sortedclassdata[i].mean(0))
    return classmeans


def find_covariance(sortedclassdata, classmeans):
    cov = []
    # total number of data points (rows) per class
    ndpc = len(sortedclassdata[0])
    for i in range(len(classmeans)):
        xn = np.transpose(sortedclassdata[i])
        mean_class = np.transpose(classmeans[i])
        tempvariance = sum([(xn[:, x] - mean_class) * np.transpose(xn[:, x] - mean_class) for x in range(int(ndpc))])
        tempvariance = tempvariance / (ndpc - 1)
        cov.append(tempvariance)
    return cov


# find likelihood, given a gaussian distribution
# and knowing the mean and variance, or in this case, the covariance
def find_n_class_probability(dataset, classmeans, covariance, priorProb, classes):
    expo = []
    nclassprob = []
    probabilityofclass = []
    datasetDimensions = len(covariance[0])
    testdatasetMatrix = np.matrix(dataset)
    datasetTranspose = np.transpose(testdatasetMatrix[:,0:len(dataset[0])-1])
    for i in range(len(dataset)):
        x = datasetTranspose[:, i]
        for j in range(len(classmeans)):
            determinate = np.linalg.det(covariance[j])
            if determinate == 0:
                addValue = 0.006*np.identity(datasetDimensions)
                covariance[j] = addValue + covariance[j]
                determinate = np.linalg.det(covariance[j])
            exponent = (-0.5)*np.transpose(x-np.transpose(classmeans[j]))*np.linalg.inv(covariance[j])*(x-np.transpose(classmeans[j]))
            expo.append(exponent)
            nprobabilityofclass = priorProb[j]*(1/((2*math.pi)**(datasetDimensions/2)))*(1/(determinate**0.5))*math.exp(expo[j])
            probabilityofclass.append(nprobabilityofclass)
        arrayprob = np.array(probabilityofclass)
        nclassprob.append(classes[np.argmax(arrayprob)])
        probabilityofclass = []
        expo = []
    return nclassprob


def Accuracy(nclassprob, dataset):
    Classes = np.transpose([np.asarray(nclassprob, dtype=np.float32)])
    Truth = np.transpose([np.asarray(dataset[:, dataset.shape[1]-1])])
    validate = np.equal(Classes, Truth)
    accuracy = 100 * (np.sum(validate) / dataset.shape[0])
    return accuracy

def convert_covariance_to_naive(matrix):
    numofclasses = len(matrix)
    numoffeatures = len(matrix[0])
    for i in range(numofclasses):
        for j in range(numoffeatures):
            for k in range(numoffeatures):
                if j != k:
                    matrix[i][j, k] = 0
    print("Converted covariance to Naive Bayes")
    return matrix
#-----------------------------------------------------------------------------------------------------------------------
print("\n***************************************************BAYESIAN CLASSIFIER***************************************")
# loading Training data
trainingData = load_csv(TrainFile)
# loading Testing data
testingData = load_csv(TestFile)
testingData = testingData[0:1000]

no_classes = np.transpose([np.asarray(testingData[:, testingData.shape[1]-1])])
no_classes = np.unique(no_classes)
print("\nClass Labels for this dataset - ")
for i in no_classes:
    print(i,end=" ")
print()

# getting sorted classes
sortclassdata, classes = class_sorted_data(trainingData)
# caculating Prior Probability of all classes
priorProb = prior_prob(trainingData, sortclassdata)
# calculating mean
#print("\n Mean by class of dataset - ")
meansbyclass = find_mean(sortclassdata)
#print(meansbyclass)
#print("\n Covariance - ")
# finding covariance
covariance = find_covariance(sortclassdata, meansbyclass)
#print(covariance)

print("\nBayes Classifer on Training Data\n")
nclassprob_train = find_n_class_probability(trainingData, meansbyclass, covariance, priorProb, classes)
accuracy_train = Accuracy(nclassprob_train, trainingData)
print("Probable Classes assigned to Train Data samples by bayes classifier: ")
print("Train data size = ",len(nclassprob_train))
print(nclassprob_train)
print("Accuracy of Model on Training data  = ",round(accuracy_train,3), "%")

print("\nBayes Classifer on Testing Data\n")
nclassprob_test = find_n_class_probability(testingData, meansbyclass, covariance, priorProb, classes)
accuracy_test = Accuracy(nclassprob_test, testingData)
print("Test data size = ",len(nclassprob_test))
print("Probable Classes assigned to Test Data samples by bayes classifier: ")
print(nclassprob_test)
print("Accuracy of Model on Tsting data = ", round(accuracy_test,3), "%")