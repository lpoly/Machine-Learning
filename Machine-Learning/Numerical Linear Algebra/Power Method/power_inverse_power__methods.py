# -*- coding: utf-8 -*-
"""
@author: lpoly
"""
import numpy as np

def Power_Method(matrix, initial_guess, error, max_iterations):
    """
    
    Parameters
    ----------
    matrix : Matrix. type: numpy.ndarray

    initial_guess : initial guess of the dominant eigenvalue. type: numpy.ndarray

    error : tolerable error of two succesive approximations. type: float

    max_iterations : tolerable iterations in order not to have infinite loop. type: int


    Returns
    -------
    type: tupple
        Dominant eigenvalue aproximation using Von Mises iteration, number of iterations executed

    """
    x = initial_guess
    lambda_old = -1
    condition = True
    iteration_counter = 0
    
    while condition:
        
        
        y = matrix @ x
        lambda_new = np.linalg.norm(y)
        x = y/lambda_new
        iteration_counter +=1
        
        if iteration_counter > max_iterations:
            print('Power Method not convergent for {} iterations.'.format(max_iterations))
            condition = False
            
        elif error > abs(lambda_new - lambda_old):
            condition = False

            
        lambda_old = lambda_new

    return lambda_new, iteration_counter

def Inverse_Power_Method(matrix, initial_guess, error, max_iterations, sigma):
    """
    Parameters
    ----------
    matrix : Matrix. type: numpy.ndarray

    initial_guess : initial guess of the dominant eigenvalue. type: numpy.ndarray

    error : tolerable error of two succesive approximations. type: float

    max_iterations : tolerable iterations in order not to have infinite loop. type: int
    
    sigma: sigma approximation type: float


    Returns
    -------
    type: tupple
        Closest eigenvalue aproximation using Inverse iteration, number of iterations executed


    """
    x = initial_guess
    lambda_old = 1.0
    condition = True
    iteration_counter = 0
    
    while condition:
        
        y = np.linalg.solve(matrix-sigma*np.identity(len(matrix[0])),x)
        lambda_new = np.linalg.norm(y)
        x = y/lambda_new
        iteration_counter += 1
        
        if iteration_counter > max_iterations: 
             print('Inverse Power Method not convergent for {} iterations.'.format(max_iterations))   
             condition = False
             
        elif error > abs(lambda_new - lambda_old):
            condition = False

        lambda_old = lambda_new
    
    return ((1/lambda_new) + sigma), iteration_counter