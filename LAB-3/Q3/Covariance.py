"""
Given a set of d-dimensional samples. Write a program to find out covariance matrix.
"""

# Calculating Covariance
import random
import numpy as np

def calculateMean(samples):
    s = np.array(samples)
    m = np.mean(s, axis=0)
    return np.array(m)

def Covarience(samples, mean, N, d):
    cov = np.zeros((d,d))
    for i in range(N):
        # for each sample
        x = samples[i]
        # difference with the mean of data
        t1 = x-mean
        # to multiply transpose of sample point with sample point
        q1 = t1.reshape(-1,1)
        q2 = t1.reshape(1,-1)
        q = np.matmul(q1,q2)
        # add to covarience matrix
        cov += q
    # divide by number of samples minus one
    c = 1.0/(N-1)
    cov = np.multiply(cov,c)
    return cov

# function to create n samples points
def d_dim_samples(start, end, N, d):
    if (start>=end):
        return
    # list of d-dimensional samples
    samples = []
    for i in range(N):
        t = []
        for j in range(d):
            t.append(random.uniform(start,end))
        samples.append(np.array(t))
    return samples


def main():
    # variables that can be changed according to the user
    N = 10
    # d = no of dimensions or features of x
    d = 3
    start = -10
    end = 10
    samples = d_dim_samples(start, end, N, d)
    print("\nNumber of Data Points = ", N)
    print("Number of Dimensions/features = ",d)
    print("\nData points : ")
    cnt=0
    for i in samples:
        cnt+=1
        print(cnt," ",i)
    print()
    mean = calculateMean(samples)
    print("Mean:\n ",mean)
    print()
    Cov = Covarience(samples, mean, N, d)
    print("Covariance:\n ",Cov)
    print()


main()
# Yukti Khurana

