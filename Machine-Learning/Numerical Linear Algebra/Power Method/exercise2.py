# -*- coding: utf-8 -*-
"""
@author:  ELEFTHERIOS POLYCHRONAKIS
ID:       MATH6090
"""

import numpy as np
import power_inverse_power__methods as tools

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


def AddToOne(A):
    ans = np.zeros(len(A))
    Sum = 0
    
    for element in A:
        Sum += element
    
    for i in range(len(A)):
        ans[i] = A[i] / Sum 
        
    return ans

A = np.array([ [0,0,0.33],[0.3,0,0],[0,0.71,0.94] ])

beautiful_eigenvector = []

# I have to use the Power Method in order to compute the dominant eigenvalue.
# Then using np.linalg.eig I am going to find the eigenvector corresponding to
# the dominant eigenvalue I already computed.

Dom_eigenvalueBest = Power_Method(A, np.ones(len(A[0])),1e-15,100)
Dom_eigenvalue0001 = Power_Method(A, np.ones(len(A[0])),1e-3,100)

# Note: A has 2 Complex and 1 Real eigenvalues.
Eigenvectors = (np.linalg.eig(A))

# Hence the only Real eigenvector is:
Dom_eigenvector = Eigenvectors[1][:,2]
print("Eigenvalue calculated by Power Method: {}\nDominant eigenvalue using NumPy:       {} ".format(Dom_eigenvalueBest[0],float(max(np.linalg.eigvals(A)))))

#in order to get rid of the imaginary part
for i in Dom_eigenvector:
    beautiful_eigenvector.append(float(i))
    
print("\nRescaled eigenvector corresponding to the dominant eigenvalue = {}\n".format(AddToOne(beautiful_eigenvector)))

print("For tolerance = 0.001, are needed {} iterations.\n".format(Dom_eigenvalue0001[1]))


