# ML in Python, exercise 1.1
# date: 24/09/2018
# name: Stefano Mozart
# description:
#   Using an esemble of ML models in place of Naive Bayes on Martine De
#   Cock's YouTubeNB.py for gender recognition of YouTube bloggers

import random
from typing import Dict, Union, List, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Reading the data into a dataframe and selecting the columns we need
df = pd.read_table("YouTube-User-Text-gender-personality.tsv")
data_YouTube = df.loc[:,['transcripts', 'gender']]

# Splitting the data into 300 training instances and 104 test instances
n = 104
all_Ids = np.arange(len(data_YouTube)) 
#random.shuffle(all_Ids)
test_Ids = all_Ids[0:n]
train_Ids = all_Ids[n:]
data_test = data_YouTube.loc[test_Ids, :]
data_train = data_YouTube.loc[train_Ids, :]

# Pre-processing and partitioning the dataset (train/test subsets)
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['transcripts'])
def factor(gender):
    return 1 if gender == 'Female' else 0
y_train: List[int] = [factor(g) for g in data_train['gender']]
X_test = count_vect.transform(data_test["transcripts"])
y_test: List[int] = [factor(g) for g in data_test['gender']]

# Training and testing the models individually
clf = {
  'lr': LogisticRegression(),
  'svm': svm.SVC(),
  'rf': RandomForestClassifier(),
  'gb': GradientBoostingClassifier(),
  'sgd': SGDClassifier(max_iter=400, loss='log')
}

y_predicted = {}
for c in clf:
    clf[c].fit(X_train, y_train)
    y_predicted[c] = clf[c].predict(X_test)

# Creating the ensemble result
y_predicted['en'] = []
for i, x in enumerate(X_test):
    y_commity = 0
    for c in clf:
        y_commity += y_predicted[c][i]
    y_predicted['en'].append(1 if y_commity > 2 else 0)

# Reporting on classification performance
classes = ['Male', 'Female']
clf['en'] = ''
for c in clf:
    print("Accuracy for %s: %.2f" % (c, accuracy_score(y_test, y_predicted[c])))
    cnf_matrix = confusion_matrix(y_test, y_predicted[c])
    print("Confusion matrix:")
    print(cnf_matrix)

