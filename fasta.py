"""
python implementation of FASTA:

An fast implementation of Proximal Gradient Descent solver
Created on Sun Mar 11 16:05:29 2018

@author: Proloy Das
"""

import numpy as np
import tqdm
import time
from scipy import linalg


def fastapy(f, g, gradf, proxg, x0, beta=0.5, max_Iter=1000, tol=1e-8):
    """

    :param f: function handle of smooth differentiable function
    :param g: function handle of non-smooth convex function
    :param gradf: function handle for gradient of smooth differentiable function
    :param proxg: function handle for proximal operator of non-smooth convex function
    :param x0: initial guess
    :param beta: parameter
    :param max_Iter: maximum number of iteration
    :param tol: tolerance

    :return: solution, function values, residual values
    """

    # Save f(x) values for back-tracking
    fx = np.empty((0))
    fx = np.append (fx, f(x))
    
    # Save objective values for returing
    fval = np.empty ((0))
    fval = np.append (fval, fx[-1] + g(x))
    
    # Save Residuals for returning
    residual = np.empty((0))
    
    # estimate Lipschitz constant ans initialize tau
    x = x0 + 0.01*np.random.randn(x0.shape[0],x0.shape[1])
    L = np.square(np.linalg.norm(gradf(x0)-gradf(x),'fro'))/np.square(np.linalg.norm(x0-x, 'fro'))
    tau = 1/L
    tol = 1e-8
    x = np.copy(x0)
    gradfx = gradf(x)

    for i in range(max_Iter):
        # time.sleep(0.001)
        z = proxg(x - tau*gradfx, tau)

        "backtracking"
        fk = np.max(fx)
        # fk = fx[-1]
        while f(z) > fk + np.sum(gradfx*(z-x)) + np.square(np.linalg.norm(z-x, 'fro'))/(2*tau):
            tau = beta*tau
            z = proxg(x - tau*gradfx, tau)

        # Check for convergence and if reached break
        gradfz = gradf(z)
        residual = np.append(residual, np.linalg.norm(gradfz + (x - tau*gradfx - z) / tau, 'fro') ** 2)
        if residual[-1]/residual[0] < tol:
            break
        
        # choose next step size using adaptive BB method
        deltax = z - x
        deltaF = gradfz - gradfx
        n_deltax = np.linalg.norm(deltax, 'fro') ** 2
        n_deltaF = np.linalg.norm(deltaF, 'fro') ** 2
        innerproduct_xF = np.sum(deltax * deltaF)
        if n_deltax == 0:
            break
        elif (n_deltaF == 0) | (innerproduct_xF == 0):
            tau = 1/L
        else:
            tau_s = n_deltax/innerproduct_xF  # steepest descent
            tau_m = innerproduct_xF/n_deltaF  # minimum residual
            # adaptive BB method
            if 2*tau_m > tau_s:
                tau_k = tau_m
            else:
                tau_k = tau_s - 0.5*tau_m
            if tau_k > 0:
                tau = tau_k

        x = np.copy(z)
        gradfx = np.copy(gradfz)
        fx = np.append(fx, f(x))
        fval = np.append(fval, fx[-1] + g(x))

    return x, fval, residual
