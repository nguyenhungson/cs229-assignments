import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()

    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    #
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', c='b', label='Admitted')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', c='y', label='Not admitted')
    plt.draw()
    plt.show()