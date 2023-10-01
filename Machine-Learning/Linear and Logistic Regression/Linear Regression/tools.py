# -*- coding: utf-8 -*-
"""
@author: LEFTERIS POLYCHRONAKIS
ID: math6090
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

def vanilla_gradient_descent(X, y, theta, max_it, learning_rate, 
                             epsilon, delta, cost_func):
    """
    Implemantation of Vanilla Gradient Descent algorithm for linear regression
    
    Parameters
    ----------
    X : TYPE numpy.ndarray
        DESCRIPTION. Train matrix
    y : TYPE numpy.ndarray
        DESCRIPTION. Train vector
    theta : TYPE numpy.ndarray
        DESCRIPTION. Starting vector of theta
    max_it : TYPE int
        DESCRIPTION. Maximum number of iterations to exexute
    learning_rate : TYPE int
        DESCRIPTION. Learning rate 
    epsilon : TYPE float
        DESCRIPTION. Stopping criterion for the 2-norm of 2 consecutive thetas
    delta : TYPE float
        DESCRIPTION. Stopping criterion for the absolute value of cost.
    cost_func : TYPE function
        DESCRIPTION. Used to compute cost at each iteration.

    Returns
    -------
    theta : TYPE
        DESCRIPTION. Hyperplane parameters
    thetas : TYPE
        DESCRIPTION. history of theta
    costs : TYPE
        DESCRIPTION. history of cost
    it_ctr : TYPE int
        DESCRIPTION. number of iterations executed

    """
        
    N = len(y)
    thetas = [theta]
    costs =[cost_func(theta, X, y)]
    
    it_ctr = 0
    
    while True:
        
        error = y - X @ theta
        grad = (X.T) @ error
        theta = theta + 1/N * learning_rate * grad
        thetas.append(theta)
        it_ctr += 1
        
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                cost = abs( cost_func(theta, X, y) )
                costs.append(cost)
            except RuntimeWarning as warn:
                print("An error occurred: {} , try changing the learning rate.".format(warn))
                cost = np.nan
                costs = np.inf * np.ones(len(costs))
                theta = np.inf * np.ones(len(theta))
                return theta, thetas, costs, it_ctr
                
        
        if np.linalg.norm(thetas[-1] - thetas[-2]) < epsilon:
            break 
        
        elif cost < delta:
            break        
        
        elif it_ctr >= max_it - 1:
            break
         
    return theta, thetas, costs, it_ctr


def cost_plotter(cost, lr):
    """
    
    Parameters
    ----------
    cost : numpy.ndarray
        DESCRIPTION. Cost contains history of costs as returned 
                     by Gradient Descent implementation above.
    lr : TYPE float
        DESCRIPTION. Learning rate used

    Returns Plots cost after each iteration
    -------
    None.

    """
    
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = [16, 9]
    
    plt.plot(range(1,len(cost)+1), cost, color = "maroon", linewidth = 3)
    plt.title(r'Cost after each iteration for $\lambda = {}$'.format(lr))
    plt.xlabel(r'$k$')
    plt.ylabel(r'$J( \theta ^k)$')
    
    plt.show()
    
    return None