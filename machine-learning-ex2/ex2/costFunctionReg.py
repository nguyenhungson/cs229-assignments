import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #


    # ===========================================================
    hypothesis = sigmoid(X.dot(theta))
    cost = np.sum(-y*np.log(hypothesis) - (1-y)*np.log(1-hypothesis)) / m
    cost += lmd / (2*m) * np.sum(theta[1:] ** 2)

    grad[0] = np.sum((hypothesis - y) * X[:, 0]) / m
    for j in range(1, theta.size):
        grad[j] = np.sum((hypothesis - y) * X[:, j]) / m + lmd*theta[j]/m
    return cost, grad
