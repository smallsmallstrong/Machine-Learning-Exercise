"""
Polynomial Features
Date:23.06.2019
Author:Yulian Sun
"""
import numpy as np
import matplotlib.pyplot as plt

file = np.loadtxt('../linRegData.txt', dtype=float)
trainingSet = file[0:20]
testingSet = file[20:150]
#degree = 21
def get_phix(dataset,degree):
    n = len(dataset)
    input = dataset[:, 0]
    output = dataset[:, 1]
    X = np.vander(input, degree, increasing=True)
    equ = np.linalg.inv(np.dot(X.T, X) + 0.000001 * np.eye(degree, degree))
    be = np.dot(equ, X.T)
    beta = np.dot(be, output)
    return beta

def poly_equation(dataset,degree):# calculate polynomial equation
    n = len(dataset)
    input = dataset[:,0]
    output = dataset[:,1]
    x = np.vander(input, degree, increasing=True)
    beta = get_phix(trainingSet,degree)
    y = np.dot(x,beta)
    return y

def evaluvation_RMSE(dataset,y):# calculate RMSE
    n = len(dataset)
    output = dataset[:,1]
    sum = 0.0
    for i in range(n):
        sum = sum+(output[i] - y[i]) ** 2
    error = np.sqrt(1/n * sum)
    return error

def main():
    D = 22
    Y_trainig = np.empty((D,1))
    Y_testing = np.empty((D,1))
    least_training_err=10000;j=0
    least_testing_err=10000;k=0
    X_axis = np.arange(22).reshape(22,1)
    for i in range(D):
        y_training = poly_equation(trainingSet,i)
        error_training = evaluvation_RMSE(trainingSet,y_training)
        Y_trainig[i,:]=error_training
        if (error_training < least_training_err):
            j = i
            least_training_err = error_training

        y_testing = poly_equation(testingSet,i)
        error_testing = evaluvation_RMSE(testingSet,y_testing)
        Y_testing[i, :] = error_testing
        if (error_testing <  least_testing_err):
            k = i
            least_testing_err = error_testing

    plt.plot(X_axis, Y_trainig,label='from training data')
    plt.plot(X_axis, Y_testing,label='from testing data')
    plt.legend()
    plt.show()
    best_model= poly_equation(file, k)
    print('best testing error',k,least_testing_err)
    print('best training error',j,least_training_err)
    plt.plot(file[:, 0], file[:, 1],'o', label='from origin data')
    plt.plot(file[:, 0], best_model, 'o', label='from best model')
    plt.xlabel('Complexity')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()


    # evaluvation_RMSE(testingSet, y)

if __name__ == '__main__':
        main()