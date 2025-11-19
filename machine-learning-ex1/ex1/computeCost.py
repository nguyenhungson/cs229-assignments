import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.
    
    # Cost function formula: J(theta) = 1/(2m) * sum((h(xi) - yi)^2) for i=1 to m
    # Here the index i goes from 0 to m-1
    for i in range(m):
        # theta.T * X[i] is equivalent to X[i] * theta cause in NumPy, 1D array is treated the same as its transpose 
        # theta and X[i] shape is (2, ), unoriented 1D arrays
        hypothesis = np.dot(X[i], theta)
        cost += (hypothesis - y[i]) ** 2
    cost = cost / (2 * m)

    # ==========================================================

    return cost
