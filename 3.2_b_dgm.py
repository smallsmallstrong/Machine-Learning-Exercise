"""
Linear Discriminant Analysis
Date:27.06.2019
Author:Yulian Sun
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

file =np.loadtxt('../ldaData.txt')
c1 = file[0:50]
c2 = file[50:93]
c3 = file[93:137]

class LDA:
    def __init__(self,file,c1,c2,c3):
        # mean value for file,C1,C2,C3
        self.mu_file = np.mean(file,axis=0)
        self.mu1 = np.mean(c1,axis=0)
        self.mu2 = np.mean(c2,axis=0)
        self.mu3 = np.mean(c3,axis=0)

        self.sigma = 1.0/(len(file)-3.0)* (np.cov(c1.T)+np.cov(c2.T)+np.cov(c3.T))

        self.p1_prior = 50/137
        self.p2_prior = 43/137
        self.p3_prior = 44/137

    def bayesian_classifier(self):
        C1 = [];C2 = [];C3 = []
        for i in range(len(file)):
            y1 = file[i]@la.solve(self.sigma,self.mu1) - 1 / 2 \
                 * self.mu1.T@la.solve(self.sigma,self.mu1)+np.log(self.p1_prior)
            y2 = file[i] @ la.solve(self.sigma, self.mu2) - 1 / 2 \
                 * self.mu2.T @ la.solve(self.sigma, self.mu2) + np.log(self.p2_prior)
            y3 = file[i] @ la.solve(self.sigma, self.mu3) - 1 / 2 \
                 * self.mu3.T @ la.solve(self.sigma, self.mu3) + np.log(self.p3_prior)
            if (y1 > y2 and y1 > y3): C1.append(file[i])
            elif (y2 > y1 and y2 > y3): C2.append(file[i])
            elif (y3 > y1 and y3 > y2): C3.append(file[i])
        return C1,C2,C3

lda = LDA(file,c1,c2,c3)
N = len(file)
C1,C2,C3 = lda.bayesian_classifier()
C1 = np.array(C1)
C2 = np.array(C2)
C3 = np.array(C3)

plt.subplot(211) # show the original points
plt.scatter(c1[:, 0], c1[:, 1], label = 'Class C1')
plt.scatter(c2[:, 0], c2[:, 1], label = 'Class C2')
plt.scatter(c3[:, 0], c3[:, 1], label = 'Class C3')
plt.legend()

plt.subplot(212) # show the classifier
plt.scatter(C1[:, 0], C1[:, 1], label='Class C1')
plt.scatter(C2[:, 0], C2[:, 1], label='Class C2')
plt.scatter(C3[:, 0], C3[:, 1], label='Class C3')
plt.legend()

plt.show()
