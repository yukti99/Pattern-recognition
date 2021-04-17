# NORMAL DISTRIBUTION

import random
import math
import matplotlib.pyplot as plt

def data_mean(samples):
    return sum(samples)/(1.0*len(samples))

def data_varience(samples, data_mean):
    summ = 0
    for x in samples:
        t1 = abs(x-data_mean)
        t2 = pow(t1,2)
        summ+=t2
    return summ*1.0/len(samples)

# function to apply normal distribution
def Normal_Distribution(x, mean,var):
    t1 = 2 * math.pi * var
    t2 = 1.0 / pow(t1, 0.5)
    t3 = ((-1.0) * pow(x - mean, 2)) / 2 * var
    t4 = pow(math.e, t3)
    return (t2*t4)

# function to create n samples points
def generate_random_samples(start, end, n):
    if (start>=end):
        return 
    samples = []
    for i in range(n):
        samples.append(random.uniform(start,end))
    return samples

def main():
    start = -4
    end = 4
    n = 100
    samples = generate_random_samples(start,end,n)
    samples.sort()
    mean = data_mean(samples)
    var = data_varience(samples,mean)
    print("Mean = ",mean)
    print("Varience = ",var)
    distr = []
    for i in samples:
        px = Normal_Distribution(i,mean,var)
        distr.append(px)
    plt.plot(samples,distr)
    plt.show()

main()

