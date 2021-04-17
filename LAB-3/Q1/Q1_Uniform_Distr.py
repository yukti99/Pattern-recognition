# UNIFORM DISTRIBUTION
import random
import matplotlib.pyplot as plt

# function to apply uniform distribution
def Uniform_Distribution(x, a, b):
    if (a<x<b):
        return 1.0/(b-a)
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
    start = 0
    end = 10
    n = 100
    samples = generate_random_samples(start, end, n)
    samples.sort()
    a = 4
    b = 6
    distr = []
    for i in samples:
        px = Uniform_Distribution(i, a, b)
        distr.append(px)
    plt.plot(samples, distr)
    plt.show()

main()

