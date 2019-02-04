#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def precision_and_recall(real, pred):
    ''' Calculate precision and recall comparing two lists '''
    print('Starting precision and recall func')
    total_n, total_p, TP, TN, FP, FN = 0, 0, 0, 0, 0, 0
    for idx, val in enumerate(real):
        if real[idx] == 0.:
            # negative
            total_n += 1
            if real[idx] == pred[idx]:
                # true negative
                TN += 1
            else:
                # false negative
                FP += 1
        else:
            # positive
            total_p += 1
            if real[idx] == pred[idx]:
                # true positive
                TP += 1
            else:
                # false positive
                FN += 1
    recall = TP / (TP + float(FN))
    precision = TP / (TP + float(FP))

    print('total postive: ', total_p)
    print('total negative: ', total_n)
    print('tp: ', TP)
    print('tn: ', TN)
    print('fp: ', FP)
    print('fn: ', FN)
    print('precision: ', precision)
    print('recall: ', recall)

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )
# data_df = pd.DataFrame.from_dict(data_dict, orient="index")
# data_df.reset_index(inplace=True)
# data_df.rename(columns={"index":"names"}, inplace=True)
### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)
# Create DT
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
# If all predictions were 0:
# pred = [0. if i == 0. else 0. for i in pred]
acc = accuracy_score(labels_test, pred)
print(acc)
print(pred)
# How many POIs are in the test set for your POI indetifier?
no_poi = sum(pred)
print("Number of POIs predicted: ", no_poi)
# How many people total are in your test set?
no_people = len(labels_test)
print('Number of people in test set: ', no_people)
print(labels_test)

# recall, precision = precision_and_recall(labels_test, pred)
# Precision
precision = precision_score(labels_test, pred)
# Recall
recall = recall_score(labels_test, pred)
print('Recall: ', recall)
print('Precision: ', precision)

new_pred = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
new_real_pred = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
precision_and_recall(new_real_pred, new_pred)