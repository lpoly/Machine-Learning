# -*- coding: utf-8 -*-
"""
@author: lpoly
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def data_cleaner(matrix):
    """
    
    Parameters
    ----------
    matrix : TYPE numpy.ndarray
        DESCRIPTION. input array containing string objects on both columns

    Returns
    -------
    Xy : TYPE numpy.ndarray
        DESCRIPTION. when first row is value is ham replace with 0 else with 1

    """
    
    Xy = np.zeros((np.shape(matrix)[0], 2), dtype = 'object')
    
    for i in range(len(matrix)):
        
            t = tuple(matrix[i][0].split(sep = '\t')) # get flag and text
            
            Xy[i][0] = 1 if t[0] == 'spam' else 0
            Xy[i][1] = t[1] 
    
    return Xy

def spamicity(words, vectorizer, model):
    """
    
    Parameters
    ----------
    words : TYPE  list
        DESCRIPTION. vectorizer  features
    vectorizer : TYPE list
        DESCRIPTION. vectorizer values
    model : TYPE scikit model
        DESCRIPTION. model with which to predict probabilities

    Returns
    -------
    spamicity_sorted : TYPE list
        DESCRIPTION. returns a sorted list with tupples each value of the  
                    features and the probability of an email containing only
                    this feature to be spam. The array is sorted in relation
                    to the score given by: log(P(feature | email is spam)) 
                                        - log(P(feature | email is not spam))

    """
    log_probs = []
    spamicity = {}  
    
    for word in words:
        
        mat = np.zeros(1, dtype = 'object')
        mat[0] = word
        
        word_count = vectorizer.transform(mat)
        log_probs.append(model.predict_log_proba(word_count))

    scores = list(ar[0][1] - ar[0][0] for ar in log_probs)

    
    for i in range(len(words)):
        spamicity[words[i]] = scores[i]
    
    spamicity_sorted = sorted(spamicity.items(),key= lambda item: item[1],
                   reverse = True)

    return spamicity_sorted

def heatmap(confusion_mat, title):
    
    # plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)

    sns.heatmap(confusion_mat, annot=True, fmt = '3d',
                cmap = 'coolwarm', vmax=64)

    plt.title('Heatmap using {}'.format(title))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()