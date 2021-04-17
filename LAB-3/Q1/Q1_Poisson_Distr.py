# POISSON DISTRIBUTION

import random
import math
import matplotlib.pyplot as plt

# function to apply poisson distribution
def Poisson_Distribution(x, lambd):
    t1 = pow(lambd,x)
    t2 = pow(math.e,-1.0*lambd)
    t3 = math.factorial(x)
    return (1.0*t1*t2)/t3

# function to create n samples points
def generate_random_samples(start, end, n):
    if (start >= end):
        return
    samples = []
    for i in range(n):
        samples.append(random.randint(start, end))
    return samples


def main():
    start = 0
    end = 50
    n = 300
    samples = generate_random_samples(start, end, n)
    samples.sort()
    lambd = 15
    distr = []
    for i in samples:
        px = Poisson_Distribution(i, lambd)
        distr.append(px)
    plt.plot(samples, distr)
    plt.show()


main()

