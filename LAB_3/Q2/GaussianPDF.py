"""
Generate N = 500 2-D random data points and plot its corresponding Gaussian PDF.
"""
import matplotlib.pyplot as plt
import random
import numpy as np
import math


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

def GaussianPx(x, mean, Cov, d):
    # determinant of Covariance matrix
    cov_det = np.linalg.det(Cov)
    cov_inv = np.linalg.inv(Cov)

    #print("inv = ",cov_inv)

    # difference of x (data point) and mean of distribution, and its transpose
    x1 = np.array(x - mean).reshape(1,-1)
    x2 = np.array(x - mean).reshape(-1, 1)

    #print("x1 = ",x1)
    t1 = 1.0/((2*math.pi)**(d/2.0))
    t2 = 1.0/(cov_det**0.5)
    t3 = np.matmul(x1,cov_inv)
    t4 = np.matmul(t3,x2)
    px = t1 * t2 * pow(math.e,-0.5*t4)

    return px

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
    start = -10
    end = 10
    # variables that can be changed according to the user
    N = 500
    # d = no of dimensions or features of x
    d = 2
    samples = d_dim_samples(start, end, N, d)
    print(samples)
    print("\nNumber of Data Points = ", N)
    print("Number of Dimensions/features = ",d)
    print("\nData points : ")
    cnt=0
    for i in samples:
        cnt+=1
        print(cnt," ",i)
    print()
    print("\nMean: ")
    mean = calculateMean(samples)
    print(mean)

    print("\nCovariance: ")
    Cov = Covarience(samples, mean, N, d)
    print(Cov)
    print()
    distr = []
    for i in samples:
        px = GaussianPx(i, mean, Cov, d)
        distr.append(px[0][0])

    # 3D Graph to visualize the Gaussian Distribution
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    zvals = np.array(distr)
    xvals = []
    yvals = []
    for i in samples:
        xvals.append(i[0])
        yvals.append(i[1])
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    ax.scatter3D(xvals, yvals, zvals, c=zvals, cmap='Oranges')
    plt.title("Gaussian PDF by Yukti Khurana")
    plt.xlabel('Feature-1')
    plt.ylabel('Feature-2')
    ax.set_zlabel('Px')
    plt.show()

main()

# Yukti Khurana

