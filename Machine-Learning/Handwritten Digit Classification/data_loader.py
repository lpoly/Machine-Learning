# -*- coding: utf-8 -*-
"""
@author: lpoly
"""

import numpy as np
import tools

train_photos = np.loadtxt(r'./images_train.csv', dtype = 'object') 
train_labels = np.loadtxt(r'./labels_train.csv', dtype = 'int')

test_photos = np.loadtxt(r'./images_test.csv', dtype = 'object') 
test_labels = np.loadtxt(r'./labels_test.csv', dtype = 'int')

X_train = tools.data_cleaner(train_photos)
X_test = tools.data_cleaner(test_photos)
y_train = tools.one_hot_representator(train_labels)
y_test = tools.one_hot_representator(test_labels)

np.save(r'./X_train.npy', X_train, allow_pickle=True)
np.save(r'./y_train.npy', y_train, allow_pickle=True)
np.save(r'./X_test.npy', X_test, allow_pickle=True)
np.save(r'./y_test.npy', y_test, allow_pickle=True)