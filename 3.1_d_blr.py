"""
Bayesian Linear Regression
Date:26.06.2019
Author:Yulian Sun
"""
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

file = np.loadtxt('../linRegData.txt', dtype=float)
trainingSet = file[0:20]
testingSet = file[20:150] #for x*
# polynomial model
def get_phix(dataSet,degree):
    N = len(dataSet)
    PhiX = np.zeros((N, degree))
    for i in range(N):
        for k in range(degree):
            PhiX[i, k] = dataSet[i, 0] ** k
    return PhiX

# calculate mean
def predictive_distribution(testingSet,caseSet):
    degree = 13
    noise = 0.0025
    beta = 1 / noise
    lamb = 0.000001
    alpha = lamb * beta
    Y = caseSet[:, 1]
    phiX = get_phix(testingSet, degree)
    PhiX = get_phix(caseSet, degree)
    #use testing set and each training set to calculate means
    equ = la.solve(PhiX.T @ PhiX + lamb * np.eye(degree,degree),PhiX.T @ Y)
    mu = phiX @ equ
    return mu

# calculate standard derivation
def cal_sigma(dataset,caseSet):
    n = len(dataset)
    degree = 13
    noise = 0.0025
    beta = 1 / noise
    lamb = 0.000001
    alpha = lamb * beta
    PhiX = get_phix(caseSet,degree)
    phiX = get_phix(dataset,degree)
    equ_sigma = np.empty((n,1))
    for i in range(n):# iterate phiX matrix by lines
        temp = 1.0 / beta + phiX[i, :] @ \
               la.solve(alpha * np.eye(degree,degree)
                        + beta * (PhiX.T @ PhiX),phiX[i, :].T)
        equ_sigma[i] = temp
    sta = equ_sigma**0.5
    return sta

pos1 = np.linspace(0,2,1000).T
pos2 = np.linspace(0,2,1000).T
pos = np.c_[pos1,pos2]

data_num = np.array([10,12,16,20,50,150])
mu = np.empty((1,1000))
sta = np.empty((1,1000))
data1 = np.empty((1,1000))
data2 = np.empty((1,1000))
for i in range(6):
    num = data_num[i]
    case = file[0:num]
    mu = predictive_distribution(pos,case).flatten()
    sta = cal_sigma(pos, case).flatten()
    data1 = mu+sta #parameter 1 for fill_between function
    data2 = mu-sta #parameter 2 for fill_between function
    plt.subplot(2, 3, i + 1)
    plt.title('the first {} data points'.format(data_num[i]))
    plt.plot(pos[:,0], mu, '-', linewidth='0.2')
    plt.plot(case[:,0], case[:,1], '.')
    plt.fill_between(pos[:,0], data1, data2, facecolor='yellow')
plt.show()












