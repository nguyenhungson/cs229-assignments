import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    hypothesis = sigmoid(X.dot(theta))
    cost = (np.sum(-y*np.log(hypothesis) - (1-y)*np.log(1-hypothesis)))/m
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #


    # ===========================================================

    for j in range(theta.size):
        grad[j] = np.sum((hypothesis - y) * X[:, j]) / m
    return cost, grad
