# -*- coding: utf-8 -*-
"""
@author: lpoly
"""
import numpy as np
from tools import Newton_Raphson, h_theta, grad, hessian, point_plotter, confusion_matrix
from sklearn.linear_model import LogisticRegression


np.set_printoptions(precision = 8, linewidth = 75, suppress = True)
np.random.seed(6090)  

# %%
# Get train data

data1_train = np.loadtxt(r'./set1_train.txt')
data2_train = np.loadtxt(r'./set2_train.txt')

X1_train_ = data1_train[:,:2]
y1_train = data1_train[:,-1]

X2_train_ = data2_train[:,:2]
y2_train = data2_train[:,-1]

# Add column for theta_0  
X1_train = np.column_stack((np.ones(len(X1_train_[:,:0])),X1_train_))
X2_train = np.column_stack((np.ones(len(X2_train_[:,:0])),X2_train_))

# %% 
# Get test data

data1_test = np.loadtxt(r'./set1_test.txt')
data2_test = np.loadtxt(r'./set2_test.txt')

X1_test_ = data1_test[:,:2]
y1_test = data1_test[:,-1]

X2_test_ = data2_test[:,:2]
y2_test = data2_test[:,-1]

# To use for heatmap later
X1_test = np.column_stack((np.ones(len(X1_test_[:,:0])), X1_test_))
X2_test = np.column_stack((np.ones(len(X2_test_[:,:0])), X2_test_))

# %%
# Compute theta using build-in sickit learn method and Newton-Raphson 
# implementaition to minimize cost function

theta = np.zeros((3,))

ans1 = Newton_Raphson(f = h_theta, df = grad, H = hessian, x = X1_train,
                     y = y1_train, theta = theta, epsilon=1e-4)

ans2 = Newton_Raphson(f = h_theta, df = grad, H = hessian, x = X2_train,
                     y = y2_train, theta = theta, epsilon=1e-4)

logreg1 = LogisticRegression(random_state=6090)
logreg1.fit(X1_train_, y1_train)

logreg2 = LogisticRegression(random_state=6090)
logreg2.fit(X2_train_, y2_train)

# %%
#get scikit coefficients in same format

coeffs1 = np.insert(logreg1.coef_, 0, logreg1.intercept_ )

coeffs2 = np.insert(logreg2.coef_, 0, logreg2.intercept_ )

# %%
# Plot points and decision boundary

point_plotter(data1_test[:,0], data1_test[:,1],
              data1_test[:,2], ans1, 1)#, sci = coeffs1)

point_plotter(data2_test[:,0], data2_test[:,1], 
              data2_test[:,2], ans2, 2)#, sci = coeffs2)

#%%
# make predictions calculate probabilities and print confusion matrix

predX1 = []
predX2 = []
prob1 = []
prob2 = []

for i in range(len(X1_test)):
    p1 = h_theta(X1_test[i,:], ans1)
    prob1.append("P{0:3d}: {1:0.3f}".format(i+1, p1))
    if p1 > 0.5:
        predX1.append(1.0)
    else:
        predX1.append(0.0)

for i in range(len(X2_test)):
    p2 = h_theta(X2_test[i,:], ans2)
    prob2.append("P{0:3d}: {1:0.3f}".format(i+1, p2))
    if p2 > 0.5:
        predX2.append(1.0)
    else:
        predX2.append(0.0)

confusion_matrix(predX1, data1_test[:,2], 1)
confusion_matrix(predX2, data2_test[:,2], 2)  


prob1 = np.array(prob1).reshape(20,5)
prob2 = np.array(prob2).reshape(20,5)

# %%
# print output
print("\n")
print(33 * '=' + ' Dataset 1 ' + 33 * '=')
print("\nUsing Sickit Learn function: {}".format(coeffs1))
print("Using Newton-Raphson method for LR: {} ".format(ans1))
print("2-norm of the difference: {0:0.4f}\n"
      .format(np.linalg.norm(coeffs1 - ans1)))
print("Probability for each point to be blue(P is abriviation for point): \n")
print(prob1)
print(75 * "=")
print(2*"\n")
print(33 * '=' + ' Dataset 2 ' + 33 * '=')
print("\nUsing Sickit Learn function: {}".format(coeffs2))
print("Using Newton-Raphson method: {} ".format(ans2))
print("2-norm of the difference: {0:0.4f}\n"
      .format(np.linalg.norm(coeffs2 - ans2)))
print("Probability for each point to be blue(P is abriviation for point): \n")
print(prob2)
print(75 * "=")