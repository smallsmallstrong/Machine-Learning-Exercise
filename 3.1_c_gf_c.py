"""
Gaussian Features,Continued
Date:25.06.2019
Author:Yulian Sun
"""
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

file = np.loadtxt('../linRegData.txt', dtype=float)
trainingSet = file[0:20]
testingSet = file[20:150,:]
class LR:
    def __init__(self,sigma_sq,trainingSet,testingSet):
        self.sigma_sq = sigma_sq
        self.trainingSet = file[0:20]
        self.testingSet = file[20:150,:]

    def gaussian(self,x, mu):
        equ1 = 1 / np.sqrt(2 * np.pi * self.sigma_sq)
        gaus = equ1 * np.exp(-(x - mu) ** 2 / (2 * self.sigma_sq))
        return gaus

    def get_phix(self,dataset,mean,number):
        input =dataset[:,0]
        n = len(dataset)
        phi = np.empty((n, number))
        equ = np.empty((n, number))
        sum = np.empty((n, 1))
        for i in range(n):
            for k in range(len(mean)):
                equ[i][k] = self.gaussian(input[i],mean[k])
                sum[i] = np.sum(equ[i,:],axis=0)

            phi[i,:] = equ[i,:]/sum[i]
        Phi = np.array(phi).reshape(n,number)
        return Phi

    def gaussian_features_fit(self,dataset,mean,number):
        PhiX = self.get_phix(self.trainingSet,mean,number)
        Y = self.trainingSet[:, 1]
        beta = la.solve(PhiX.T @ PhiX + 0.000001 * np.eye(number, number), PhiX.T @ Y)
        phix = self.get_phix(dataset,mean,number)
        y = phix@beta
        return y

    def evaluvation_RMSE(self,dataset,y):# calculate RMSE
        n = len(dataset)
        output = dataset[:, 1]
        sum = 0.0
        for i in range(n):
            sum = sum + (output[i] - y[i]) ** 2
        error = np.sqrt(1 / n * sum)
        return error

sigma_sq = 0.02
LR = LR(sigma_sq,trainingSet,testingSet)

least_testing_err = 10000;k = 0
y_error = np.empty((26,1))

for i in range(15,41):
    mean = np.linspace(0, 2, i)
    y = LR.gaussian_features_fit(testingSet,mean,i)
    error = LR.evaluvation_RMSE(testingSet, y)
    y_error[i-15] = error
    if (error < least_testing_err):
        least_testing_err = error
        k = i

print(k,"-------",least_testing_err)
x = np.linspace(15,40,26)
plt.plot(x,y_error)
plt.xlabel('number of basis functions')
plt.ylabel('RMSE')
plt.show()

