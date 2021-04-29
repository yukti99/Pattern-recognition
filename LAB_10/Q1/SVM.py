import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np

print("SUPPORT VECTOR MACHINE CLASSIFICATION BY YUKTI KHURANA")

df = pd.read_csv("Iris.csv")
df = df.drop(['Id'], axis=1)
df

# taking the first 100 samples out of 150 samples
# converting this into a binary classification problem by considering only two classes
# two classes - setosa and versicolor
df = df[0:100]
df

# visualising the two classes
x = df['SepalLengthCm']
y = df['PetalLengthCm']
setosa_x = x[:50]
setosa_y = y[:50]
versicolor_x = x[50:]
versicolor_y = y[50:]
plt.figure(figsize=(8, 6))
plt.scatter(setosa_x, setosa_y, label = 'setosa',marker='+', color='purple')
plt.scatter(versicolor_x, versicolor_y,label='versicolor', marker='_', color='blue')
plt.xlabel("SepalLengthCm")
plt.ylabel("PetalLengthCm")
plt.title("Visualizing the two classes of Iris dataset")
plt.legend()
plt.show()

# Drop unnecessary features
df = df.drop(['SepalWidthCm', 'PetalWidthCm'], axis=1)
# constructing the class labels
y = []
target = df['Species']
for val in target:
    if (val == 'Iris-setosa'):
        y.append(-1)
    else:
        y.append(1)

df = df.drop(['Species'], axis=1)
X = df.values.tolist()
split_size = 0.80
# Splitting the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_size, random_state=41, stratify=y)
print(len(X_train), len(y_train), len(X_test), len(y_test))

X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train = y_train.reshape(X_train.shape[0], 1)
y_test = y_test.reshape(X_test.shape[0], 1)
print(y_train.shape, y_test.shape)

# extracting the training features
# feature 1
train_f1 = X_train[:, 0]
# feature 2
train_f2 = X_train[:, 1]


# extracting the test data features
test_f1 = X_test[:, 0]
test_f2 = X_test[:, 1]

test_f1 = test_f1.reshape(X_test.shape[0], 1)
test_f2 = test_f2.reshape(X_test.shape[0], 1)

train_f1 = train_f1.reshape(X_train.shape[0], 1)
train_f2 = train_f2.reshape(X_train.shape[0], 1)
print(train_f1.shape, train_f2.shape)

w1 = np.zeros((X_train.shape[0], 1))
w2 = np.zeros((X_train.shape[0], 1))

# no of epochs
epochs = 1
# learning rate
alpha = 0.0001

# the regularization parameter λ is set to 1/epochs.
# Therefore, the regularizing value reduces the number of epochs increases.
print("\nModel Training begins.....\n")
while (epochs < 10000):
    y = w1 * train_f1 + w2 * train_f2
    prod = y * y_train
    count = 0
    print("Epoch - ", epochs)
    for val in prod:
        if (val >= 1):
            # no misclassification
            # cost will be zero
            # w = w - alpha * (2*lambda*w)
            cost = 0
            w1 = w1 - alpha * (2 * 1 / epochs * w1)
            w2 = w2 - alpha * (2 * 1 / epochs * w2)
        # When there is a misclassification, i.e our model make a mistake on the prediction of the class of our data point, we include the loss along with the regularization parameter to perform gradient update.
        else:
            # misclassification occurs
            # so gradient updation will involve cost
            # w  = w+ alpha * (yi * xi - (2*lambda*w))
            cost = 1 - val
            w1 = w1 + alpha * (train_f1[count] * y_train[count] - 2 * 1 / epochs * w1)
            w2 = w2 + alpha * (train_f2[count] * y_train[count] - 2 * 1 / epochs * w2)
        count += 1
    epochs += 1
print("\nModel Training ends....\n")

def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Clipping the weights
index = list(range(X_test.shape[0], X_train.shape[0]))
w1 = np.delete(w1, index)
w2 = np.delete(w2, index)

# reshaping the weights
w1 = w1.reshape(X_test.shape[0], 1)
w2 = w2.reshape(X_test.shape[0], 1)

# Prediction
y_pred = w1 * test_f1 + w2 * test_f2
predictions = []
for val in y_pred:
    if (val > 1):
        predictions.append(1)
    else:
        predictions.append(-1)

pred_flower = []
for i in predictions:
    if i == -1:
        pred_flower.append("iris-setosa")
    else:
        pred_flower.append("iris-versicolor")

print("Classification according to my SVM - ")
print(predictions)
print(pred_flower)
print("Accuracy of my SVM classification = ", accuracy(y_test, predictions), "%")
print("\n")


print("SVM Classification using sklearn library for comparison\n")
svc_clf = SVC(kernel='linear')
svc_clf.fit(X_train,np.ravel(y_train,order='C'))
sklearn_ypred = svc_clf.predict(X_test)
print("Classification according to Sklearn built-in SVM -")
print(sklearn_ypred.tolist())
print("The accuracy of built-in sklearn SVM classification = ",accuracy(y_test,sklearn_ypred), "%")