# -*- coding: utf-8 -*-
"""
@author: lpoly
"""

import numpy as np
import tools 
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

warnings.filterwarnings('ignore')

# %%
# load data

X_train = np.load(r'./X_train.npy', allow_pickle = True)
y_train = np.load(r'./y_train.npy', allow_pickle = True)

X_test = np.load(r'./X_test.npy', allow_pickle = True)
y_test = np.load(r'./y_test.npy', allow_pickle = True)

# %%
# build model

lr = 0.09; alpha = 0

neural_classifier = MLPClassifier(activation =  'logistic', solver='sgd',
                                  hidden_layer_sizes = (100,), tol = 1e-4,
                                  max_iter = 20, random_state = 6090,
                                  learning_rate = 'constant', alpha = 0,
                                  learning_rate_init = lr, verbose = False)

neural_classifier.fit(X_train, y_train)

# %%
# get output

predictedProb = neural_classifier.predict_proba(X_test)

cost = neural_classifier.loss_curve_

predicted_softmax = tools.softmax(predictedProb)

predicted = np.zeros((len(predictedProb), 10))

for i in range(len(predicted)):
    predicted[i] = tools.max_index(predicted_softmax[i])

score = accuracy_score(y_test, predicted)
error =  mean_squared_error(y_test, predicted)

print(75 * '=')
print(33*  ' ' + 'alpha  = 0' + 33 * ' ')
print(75 * '-')
print("Accuracy = {0:0.3f}".format(score))
print("Mean Squared Error = {0:0.3f}".format(error))

# %%
# build model with l2-regularization

alpha = 1e-4

neural_classifierA = MLPClassifier(activation =  'logistic', solver='sgd',
                                  hidden_layer_sizes = (100,), tol = 1e-4,
                                  max_iter = 20, random_state = 6090,
                                  learning_rate = 'constant',
                                  learning_rate_init = lr, verbose = False)

# %%
# get output

neural_classifierA.fit(X_train, y_train)

predictedProbA = neural_classifierA.predict_proba(X_test)

costA = neural_classifier.loss_curve_

predicted_softmaxA = tools.softmax(predictedProbA)

predictedA = np.zeros((len(predictedProbA), 10))

for i in range(len(predictedA)):
    predictedA[i] = tools.max_index(predicted_softmaxA[i])

scoreA = accuracy_score(y_test, predictedA)
errorA =  mean_squared_error(y_test, predictedA)

print('\n' + 75 * '=')
print(31*  ' ' + 'alpha  = .0001' + 31 * ' ' )
print(75 * '-')
print("Accuracy = {0:0.3f}".format(scoreA))
print("Mean Squared Error = {0:0.3f}\n".format(errorA))

# %%
# plot cost per iteration

tools.cost_plotter(costA, lr)