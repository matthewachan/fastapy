"""
This script shows how to use FASTA to solve regularized least-square problem:
        min  .5||Ax-b||^2 + mu*|x|
Where A is an MxN matrix, b is an Mx1 vector of measurements, and x is the Nx1 vector of unknowns.
The parameter 'mu' controls the strength of the regularizer.

@author: Proloy DAS

"""

import numpy as np
from scipy import linalg
from fastapy import Fasta
import matplotlib.pyplot as plt
import time


def shrink(x, mu):
    """
    Soft theresholding function
    mu = threshold
    """
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - mu, 0))


def setup_rls():
    np.random.seed(0)
    # Define problem parameters
    M = 200  # number of measurements
    N = 1000  # dimension of sparse signal
    K = 10  # signal sparsity
    mu = 0.02  # regularization parameter
    sigma = 0.01  # The noise level in 'b'

    print("Testing sparse least-squares with N={:}, M={:}".format(N, M))

    # Create sparse signal
    x = np.zeros((N, 1))
    perm = np.random.permutation(N)
    x[perm[0:K]] = 1

    # define random Gaussian matrix
    A = np.random.randn(M, N)
    A = A / linalg.norm(
        A, 2
    )  # Normalize the matrix so that our value of 'mu' is fairly invariant to N

    # Define observation vector
    b = np.dot(A, x)
    b = b + sigma * np.random.randn(*b.shape)  # add noise

    #  The initial iterate:  a guess at the solution
    x0 = np.zeros((N, 1))

    # Create function handles
    def f(x):
        return 0.5 * linalg.norm(np.dot(A, x) - b, 2) ** 2  # .5||Ax-b||^2

    def gradf(x):
        return np.dot(A.T, np.dot(A, x) - b)  # gradient of f(x)

    def g(x):
        return mu * np.abs(x).sum()  # mu*|x|

    def proxg(x, t):
        return shrink(x, mu * t)  # proximal operator for g(x)

    return f, gradf, g, proxg, x0, x


def test_rls(debugging=False):
    f, gradf, g, proxg, x0, x = setup_rls()
    # Set up Fasta solver
    lsq = Fasta(f, g, gradf, proxg)
    # Call Solver
    lsq.learn(x0, verbose=True)

    assert lsq.residuals[-1] / lsq.residuals[0] < 1e-4

    if debugging:
        plt.figure("sparse least-square")
        plt.subplot(2, 1, 1)
        plt.stem(x, markerfmt="go", linefmt="g:", label="Ground truth")
        plt.stem(lsq.coefs_, markerfmt="bo", label="Fasta solution")
        plt.xlabel("Index")
        plt.ylabel("Signal Value")

        plt.subplot(2, 1, 2)
        plt.semilogy(lsq.residuals)

        plt.show()


def test_fix_stepsize(debugging=False):
    f, gradf, g, proxg, x0, x = setup_rls()
    # Set up Fasta solver
    lsq = Fasta(f, g, gradf, proxg)
    # custom stepsize
    def fixed_step(*args):
        return 2

    # Call Solver
    lsq.learn(x0, verbose=True, linesearch=False, next_stepsize=fixed_step)

    assert lsq.residuals[-1] / lsq.residuals[0] < 1e-4

    if debugging:
        plt.figure("sparse least-square")
        plt.subplot(2, 1, 1)
        plt.stem(x, markerfmt="go", linefmt="g:", label="Ground truth")
        plt.stem(lsq.coefs_, markerfmt="bo", label="Fasta solution")
        plt.xlabel("Index")
        plt.ylabel("Signal Value")

        plt.subplot(2, 1, 2)
        plt.semilogy(lsq.residuals)

        plt.show()
