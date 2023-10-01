# -*- coding: utf-8 -*-
"""
@author: LEFTERIS POLYCHRONAKIS
ID: math6090
"""
# %%
# import libraries,
# initialize pseudorandom , set printing options

import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from tools import vanilla_gradient_descent, cost_plotter

np.set_printoptions(precision = 8, linewidth = 75, suppress = True)
np.random.seed(20) # 20 is close, 

#%%
# Get and rescale train data using max col element

data_train = np.loadtxt(r'./car_train.txt')

n = len(data_train[:,:0])

X = data_train[:,:4]
for col in range(len(X[0])):
    X[:,col] = X[:,col]/max(X[:,col])
    
X = np.column_stack((np.ones(n),X))
y = data_train[:,4]/max(data_train[:,4])

# %%
# Train data using Normal Distribution

scaler = preprocessing.StandardScaler().fit(np.column_stack( (np.ones(n),data_train) ))
data = scaler.transform(np.column_stack( (np.ones(n)/np.sqrt(n),data_train) ) )

XN = data[:,:5]
yN = data[:,5]

# %%
#define theta, learning rate

theta  = np.random.randn(5,)
lr = .876
lrN = 0.08

ans = vanilla_gradient_descent(X = X, y = y,  theta = theta, max_it = 1000,
                               learning_rate = lr, epsilon = 1e-5, delta = 1e-3,
                               cost_func = lambda theta, X, y:  ( 1/(2*len(y)) ) * np.sum(np.square((X @ theta) - y)))

ansN = vanilla_gradient_descent(X = XN, y = yN,  theta = theta, max_it = 1000,
                               learning_rate = lrN, epsilon = 1e-5, delta = 1e-3,
                               cost_func = lambda theta, X, y:  ( 1/(2*len(y)) ) * np.sum(np.square((X @ theta) - y)))

# %%
# get data from answer and plot cost 

theta = ans[0]      
thetas = ans[1] # theta history                           
costs = ans[2] # cost history
iterations = ans[3] # iter

thetaN = ansN[0]      
thetasN = ansN[1] # theta history                           
costsN = ansN[2] # cost history
iterationsN = ansN[3] # iter

cost_plotter(costs, lr) # plot cost function
cost_plotter(costsN, lrN) # plot cost function

# %%
# Get and normalize test data

data_test = np.loadtxt(r'./car_test.txt')

m = len(data_test[:,:0])

Xtest = np.column_stack((np.ones(m),data_test[:,:4]))
for col in range(len(X[0])):
    Xtest[:,col] = Xtest[:,col]/max(Xtest[:,col])

ytest = data_test[:,4]/max(data_test[:,4])

scaler = preprocessing.StandardScaler().fit(np.column_stack( (np.ones(m),data_test) ))
dataN = scaler.transform(np.column_stack( (np.ones(m)/np.sqrt(m),data_test) ) )

XtestN = dataN[:,:5]
ytestN = dataN[:,5]

# %%
# Check build-in method

reg = linear_model.LinearRegression()
reg.fit(X,y)

# %%
# print results

ModelQ = np.linalg.norm( (Xtest @ theta) - ytest)
ModelQN = np.linalg.norm( (XtestN @ thetaN) - ytestN)

print()

print(28*'=' + ' Using Max Element '+ 28*'=')
print('\n Theta =  {0} \n'.format(theta) ) 
print(' 2-norm of the difference between sickit solution and GD solution: {0:0.3f}\n'.format(np.linalg.norm(theta - reg.coef_)))
print(' Learning Rate =  {0:0.6f} \n Iterations =  {1:11d} \n Cost =  {2:17f}\n Model Quality =  {3:0.6f}'
      .format(lr, iterations, costs[-1], ModelQ))
print(75 * '=')
print(2*'\n')

print(24*'=' + ' Using Normal Distribution ' + 24*'=')
print('\n Theta =  {0} \n'.format(thetaN) ) 
print(' 2-norm of the difference between sickit solution and GD solution: {0:0.2f}\n'.format(np.linalg.norm(thetaN - reg.coef_)))
print(' Learning Rate =  {0:0.6f} \n Iterations =  {1:11d} \n Cost =  {2:17f}\n Model Quality =  {3:0.6f}'
      .format(lrN, iterationsN, costsN[-1], ModelQN))
print(75 * '=')