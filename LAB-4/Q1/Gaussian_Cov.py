import numpy as np
import matplotlib.pyplot as plt

# Mean of the Gaussian distribution
# by Yukti Khurana

Mean = [0, 0]
print("\nMean: ")
print(Mean)
print()

# Covariance matrices
Cov1 = [[1, 0], [0, 1]]
Cov2 = [[0.2, 0], [0, 0.2]]
Cov3 = [[2, 0], [0, 2]]
Cov4 = [[0.2, 0], [0, 2]]
Cov5 = [[2, 0], [0, 0.2]]
Cov6 = [[1, 0.5], [0.5, 1]]
Cov7 = [[0.3, 0.5], [0.5, 2]]
Cov8 = [[0.3, -0.5], [-0.5, 2]]

Cov_matrices = [Cov1, Cov2, Cov3, Cov4, Cov5, Cov6, Cov7, Cov8]

for i in range(8):
    print("Covariance Matrix-",i+1,":")
    print(Cov_matrices[i])
    x, y = np.random.multivariate_normal(Mean, Cov_matrices[i], 5000).T
    plt.plot(x, y, 'x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    s = "Gaussian Distribution of Covariance Matrix "+str(i+1)
    plt.title(s)
    plt.show()
    print()

