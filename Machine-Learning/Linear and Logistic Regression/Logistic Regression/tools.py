# -*- coding: utf-8 -*-
"""
@author: lpoly
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def logistic(z):
    return 1.0 / (1 + np.exp(-z))


def h_theta(x, theta): 
    #print(logistic(x @ theta)[3,])
    return logistic(x @ theta)


def grad(x, y, theta):
    h = h_theta(x, theta)   
    
    return (y - h) @ x  


def hessian(x, theta):                                                           
    h = h_theta(x, theta) 
    
    p = np.array(h)                                        
    W = np.diag((p * (1 - p)))   
                                   
    return x.T @ W @ x
    

def Newton_Raphson(f, df, H, x, y, theta, epsilon):
    """

    Parameters
    ----------
    f : TYPE numpy.ndarray
        DESCRIPTION. Function of which to compute the root
    df : TYPE numpy.ndarray
        DESCRIPTION. Derivative of the function
    H : TYPE np.array
        DESCRIPTION. Hessian of the function
    x : TYPE nunpy.ndarray
        DESCRIPTION.
    y : TYPE numpy.ndarray
        DESCRIPTION.
    theta : TYPE numpy.ndarray
        DESCRIPTION. Initial guess for root
    epsilon : TYPE float
        DESCRIPTION. Stopping criterion for the 2-norm of 2 consecutive root approx

    Returns
    -------
    thetanew : TYPE numpy.ndarray
        DESCRIPTION. root approximation using Newton-Raphson method

    """
    condition =  True
    thetaold = theta
    ctr = 0
    
    while condition :
        
        thetanew = thetaold + np.linalg.inv(H(x, thetaold)) @ df(x,y,thetaold)
        ctr += 1
        
        if np.linalg.norm(thetanew - thetaold) < epsilon:
            condition = False
              
        thetaold = thetanew

    return thetanew
    

def point_plotter(x1, x2, booleans, theta, k, sci = None):
    """

    Parameters
    ----------
    x1 : TYPE numpy.ndarray
        DESCRIPTION. x_1 componenst of points
    x2 : TYPE numpy.ndarray
        DESCRIPTION. x_1 components of points
    booleans : TYPE numpy.ndarray
        DESCRIPTION. Array of 0,1 corresponding to point color
    theta : TYPE numpy.ndarray
        DESCRIPTION. Parameters of line to plot
    k : TYPE int
        DESCRIPTION. For plot title

    Returns Plots scattered points and line
    -------
    None.

    """
    
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.style.use('ggplot')

    if sci is not None:
        xs = np.linspace(min(x1) -1 , max(x1) + 1, num = 50)
        ys = lambda x:  (sci[0] + sci[1] * x) / -sci[2] 
        
        plt.plot(xs, ys(xs), color = 'violet', 
                 label = 'Decision Boundary Sci-kit')
        
    x = np.linspace(min(x1) -1 , max(x1) + 1, num = 50)
    y = lambda x:  (theta[0] + theta[1] * x) / -theta[2] 
    
    plt.plot(x, y(x), color = 'maroon',
             label = "Decision Boundary Newton's method")
    

    
    bluex1 = []
    bluex2 = []
    
    greenx1 = []
    greenx2 = []
    
    for x in range(len(x1)):
        if booleans[x] == 1.0 :
            bluex1.append(x1[x])
            bluex2.append(x2[x])
        else:
            greenx1.append(x1[x])
            greenx2.append(x2[x])
            
    plt.scatter(bluex1, bluex2, color = "royalblue",  label = r'$y^{(i)} = 1$')
    plt.scatter(greenx1, greenx2, color = "darkgreen", marker ='x', label = r'$y^{(i)} = 0$')
    plt.title('Test Set {}'.format(k))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend()
    plt.show()
    
    return None

def confusion_matrix(predictions, true_values, k):
    """
    
    Builds the confusion matrix and plots the heatmap of correct predictions
    
    Parameters
    ----------
    predictions : TYPE numpy.ndarray
        DESCRIPTION. Predictions 
    true_values : TYPE
        DESCRIPTION. Actual Values
    k : TYPE int
        DESCRIPTION. For the plot title

    Returns Return heatmap 
    -------
    None.

    """
    
    is_true_pred_true, is_true_pred_false, is_false_pred_false, is_false_pred_true = np.zeros(4,)
    
    
    for i in range(len(predictions)):
        
        if predictions[i] == 1.0 and true_values[i] == 1.0:
            is_true_pred_true += 1
        elif predictions[i] == 0.0 and true_values[i] == 1.0:
            is_true_pred_false += 1
        elif predictions[i] == 1.0 and true_values[i] == 0.0:
            is_false_pred_true += 1
        else:
            is_false_pred_false += 1
        
        confusion_matrix = np.array([ [ is_false_pred_false, is_false_pred_true],
                                     [is_true_pred_false, is_true_pred_true] ])
        
        
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)

    sns.heatmap(confusion_matrix, annot=True, cmap = 'coolwarm', vmax=50)
    plt.title('Heatmap for test set {}'.format(k))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return None