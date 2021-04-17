# Exponential DISTRIBUTION

import random
import math
import matplotlib.pyplot as plt

# function to apply exponential distribution
def Exponential_Distribution(x, lambd):
    if (x>=0):
        t1 = (-1.0)*lambd*x
        t2 = pow(math.e, t1)
        return lambd*t2
    return 0


# function to create n samples points
def generate_random_samples(start, end, n):
    if (start >= end):
        return
    samples = []
    for i in range(n):
        samples.append(random.uniform(start, end))
    return samples


def main():
    start = -1
    end = 5
    n = 100
    samples = generate_random_samples(start, end, n)
    samples.sort()
    lambd = 2
    distr = []
    for i in samples:
        px = Exponential_Distribution(i, lambd)
        distr.append(px)
    plt.plot(samples, distr)
    plt.show()


main()

