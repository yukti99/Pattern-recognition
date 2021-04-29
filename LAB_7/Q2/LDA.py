import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("\nLinear Discriminant Analysis by Yukti Khurana")
# loading the Ionosphere dataset
df = pd.read_csv("Ionosphere.csv",header=None)
df = df[1:]
# grouping based on the class of the classification :
df_grouped = df.groupby(0)

df_group0 = df_grouped.get_group("0")
df_group1 = df_grouped.get_group("1")

X_group0 = df_group0.iloc[:, 1:].values
y_group0 = df_group0.iloc[:, 0].values

X_group1 = df_group1.iloc[:, 1:].values
y_group1 = df_group1.iloc[:, 0].values

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_group1, y_group1, train_size = 0.10, random_state = 41, stratify = y_group1)
X_train0, X_test0, y_train0, y_test0 = train_test_split(X_group0, y_group0, train_size = 0.84, random_state = 41, stratify = y_group0)

sc = StandardScaler()
X_train_std1 = sc.fit_transform(X_train1)
X_test_std1 = sc.transform(X_test1)

X_train_std0 = sc.fit_transform(X_train0)
X_test_std0 = sc.transform(X_test0)

mean1 = np.mean(X_train_std1, axis=0)
mean0 = np.mean(X_train_std0, axis=0)

def Mean(samples):
    s = np.array(samples)
    m = np.mean(s, axis=0)
    return np.array(m)

def getZ(samples, mean):
    return np.array(samples) - np.array(mean)


print("Before Dimension Reduction of Class-0 ")
print(np.size(X_train_std0,0), np.size(X_train_std0,1))

print("\nBefore Dimension Reduction of Class-1 ")
print(np.size(X_train_std1,0), np.size(X_train_std1,1))
print()
u0 = getZ(X_train_std0, mean0)
u1 = getZ(X_train_std1, mean1)

print("Class-0 Data : ")
print(u0)
print("\nClass-1 Data : ")
print(u1)
print()

def getScatterMat(Z):
    return np.dot(np.array(Z).T, np.array(Z))
m0 = getScatterMat(u0)
m1 = getScatterMat(u1)

print("SCATTER MATRIX FOR CLASS 0 : ")
print(m0)
print("\nSCATTER MATRIX FOR CLASS 1 : ")
print(m1)
print()

#Sw is the sum of the scatter matrix for both the classes
sw = np.add(np.array(m0), np.array(m1))
t1 = np.subtract(np.array(mean0), np.array(mean1))
t2 = t1[np.newaxis]
print(np.size(t2,1))
t3 = t2.T
print(np.size(t3,0))

#Sb is the between class scatter calculated using the multiplation of difference transpose and difference of the mean matrices
sb = np.matmul(t3, t2)
print(sb)

#Sb-1Sw for eigen vector calculation
req = np.matmul((sw), sb)

#Calculation of eigen values and vectors
w, v = np.linalg.eig(req)
eigen_pairs = [(np.abs(w[i]), v[:, i]) for i in range(len(w))]

# sorting the eigen vectors based on eigen values
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# taking reduced dimensionality to 4
matw = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis], eigen_pairs[2][1][:, np.newaxis], eigen_pairs[3][1][:, np.newaxis]))

data0 = np.matmul(matw.T, u0.T)
data1 = np.matmul(np.array(matw).T, u1.T)

print("After Dimension Reduction of Class-0 ")
print(data0)
print("\nAfter Dimension Reduction of Class-1 ")
print(data1)
print("\nDimensions After LDA of Ionosphere dataset = ", np.size(data1,0))
