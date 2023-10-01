# -*- coding: utf-8 -*-
"""
@author: lpoly
"""

import numpy as np
import matplotlib.pyplot as plt

def softmax(Z):

    return np.exp(Z) / np.sum(np.exp(Z))


def data_cleaner(matrix):
    """

    Parameters
    ----------
    matrix : TYPE numpy.ndarray
        DESCRIPTION. array of handwritten digit photos  

    Returns
    -------
    out : TYPE numpy.ndarray
        DESCRIPTION. replaced strings with integers and normalized

    """
    n = np.shape(matrix)[0]
    
    out = np.zeros((n, 784), dtype = float)
    
    for i in range(n):
        
        photo = np.fromstring(matrix[i], dtype = float,
                            count = -1, sep = ',') / 255
        
        out[i] = photo
        
    
    return out

def one_hot_representator(vector):
    """

    Parameters
    ----------
    vector : TYPE numpy.ndarray
        DESCRIPTION. vector containing integer labels in range 0-9.

    Returns
    -------
    out : TYPE numpy.ndarray
        DESCRIPTION. matrix representing each label with a vector of length 10.
                    vector has 0 in everywhere except the position of the
                    integer given.

    """
    n = len(vector)
    out = np.zeros((len(vector), 10))
    
    for i in range(n):
        out[i][int(vector[i])] = 1
    
    return out

def max_index(vector):
    """

    Parameters
    ----------
    vector : TYPE numpy.ndarray
        DESCRIPTION. vector of length 10 which contains probabilities

    Returns
    -------
    out : TYPE numpy.ndarray
        DESCRIPTION.  vector of length 10 with 0 everywhere except the position
                      of the greater probabilty which has value 1.

    """
    pos = 0
    for i in range(len(vector)):
        if vector[i] > vector[pos]:
            pos = i
    
    out = np.zeros(len(vector))
    out[pos] = 1
    
    return out

def cost_plotter(cost, lr):
    
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = [16, 9]
    
    x = np.array(range(1,len(cost)+1))
    
    plt.plot(x, cost, color = "maroon", linewidth = 2)
    plt.title(r'Cost after each iteration for $\lambda = {}$'.format(lr))
    plt.xlabel(r'$iteration$')
    plt.ylabel(r'$J(W^{(1)}, W^{(2)}, b^{(1)}, b^{(2)})$')
    
    pass