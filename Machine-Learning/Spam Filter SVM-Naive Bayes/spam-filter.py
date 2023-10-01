# -*- coding: utf-8 -*-
"""
@author: lpoly
"""

import numpy as np
import tools
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision = 8, linewidth = 75, suppress = True)

# %%
# load test-train data

with open("./spam_train.txt") as file:
    train = [[x.rstrip('\n')] for x in file]

with open("./spam_test.txt") as file:
    test = [[x.rstrip('\n')] for x in file]
    
test = np.array(test, dtype = 'str')
train = np.array(train, dtype = 'str')

# %%
# store data properly in arrays

Xy_train =  tools.data_cleaner(train)
Xy_test =  tools.data_cleaner(test)

# %% 
# build dictionary

vectorizer = CountVectorizer(min_df = 5,
                              stop_words =(["i","you", "he", "she", "it",
                                            "we", "they", "to", "on", "in",
                                            "at", "and", "that", "so", "for",
                                            "is", "are", "him", "her",
                                            "", "as", "u", "the", "do",
                                            "does", "not", "with", "from"]) )

# %% 
# get train data

X_train = Xy_train[:,1]
y_train = np.array(Xy_train[:,0], dtype = 'int')
X_train_count = vectorizer.fit_transform(X_train)

# %%
# build Naive Bayes model

nb_model = MultinomialNB() # alpha = 1 is Default
nb_model.fit(X_train_count, y_train)

# %%
# build Support Vector Classifier model

svm_model = SVC(probability = True) # alpha = 1.0 is Default
svm_model.fit(X_train_count, y_train)

# %% 
# test model 

X_test = Xy_test[:,1]
y_test = np.array(Xy_test[:,0], dtype = 'int')
X_test_count = vectorizer.transform(X_test)

# %%
# compute spamicity

spamicity = tools.spamicity(vectorizer.get_feature_names_out(),
                            vectorizer, nb_model)

print(27 * '=' + ' Top Five Spam Words ' + 27 * '=' + "\n")   
for i in range(5):
    print(23*' ' + "Word: {0:8s}  Score = {1:0.3f}"
          .format(spamicity[i][0], spamicity[i][1]))

# %%
# compute error for Naive Bayes

nb_probs = nb_model.predict_proba(X_test_count) # get probabilities
nb_error = np.zeros(len(nb_probs)) # initialize matrix to compute norm of

for i in range(len(y_test)):

    nb_error[i] = nb_probs[i][0] if y_test[i] == 1 else nb_probs[i][1] 

print("\n" + 75 * '=')
print("\nError using Naive Bayes = {0:0.3f}".format(np.linalg.norm(nb_error)))

# %% 
# compute error for Support Vector Classifier

svm_probs = svm_model.predict_proba(X_test_count) # get probabilities
svm_error = np.zeros(len(svm_probs)) # initialize matrix to compute norm of

for i in range(len(y_test)):

    svm_error[i] = svm_probs[i][0] if y_test[i] == 1 else svm_probs[i][1] 

print("Error using Support Vector Classifier = {0:0.3f}\n\n"
      .format(np.linalg.norm(svm_error)))

# %%
# compute accuracy 

print("Model Accuracy using Naive Bayes = {0:0.3f}"
      .format(nb_model.score(X_test_count, y_test)))

print("Model Accuracy using Support Vector Classifier = {0:0.3f}"
      .format(svm_model.score(X_test_count, y_test)))
print("\n" + 75 * '=')

# %%
# plot confusion matrices

svm_predicted = svm_model.predict(X_test_count)
svm_confusion_matrix = confusion_matrix(y_test, svm_predicted)
nb_predicted = nb_model.predict(X_test_count)
nb_confusion_matrix = confusion_matrix(y_test, nb_predicted)


tools.heatmap(nb_confusion_matrix, "Naive Bayes")
tools.heatmap(svm_confusion_matrix, "Support Vector Classifier")