#!/usr/bin/python

import sys
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tester
import pickle


payment_data = ['salary',
                'bonus',
                'long_term_incentive',
                'deferred_income',
                'deferral_payments',
                'loan_advances',
                'other',
                'expenses',                
                'director_fees', 
                'total_payments']

stock_data = ['exercised_stock_options',
              'restricted_stock',
              'restricted_stock_deferred',
              'total_stock_value']

email_data = ['to_messages',
              'from_messages',
              'from_poi_to_this_person',
              'from_this_person_to_poi',
              'shared_receipt_with_poi']

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# Take all features except email
features_list = ['poi'] + payment_data + stock_data + email_data 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Load dict into pandas df, replace NaN string for np.nan values
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)
df = df[features_list]

### Replace missing values in financial data with zeros
df[payment_data] = df[payment_data].fillna(value=0)
df[stock_data] = df[stock_data].fillna(value=0)

### Task 2: Remove outliers
df.drop(axis=0, labels=['TOTAL','THE TRAVEL AGENCY IN THE PARK'], inplace=True)

### Task 3: Create new feature(s)
# Add the new email features to the dataframe
df['to_poi_average'] = df['from_poi_to_this_person'] / df['to_messages']
df['from_poi_average'] = df['from_this_person_to_poi'] / df['from_messages']
df['shared_poi_average'] = df['shared_receipt_with_poi'] / df['to_messages']

# Add the new features to the features list
features_list.append('to_poi_average')
features_list.append('from_poi_average')
features_list.append('shared_poi_average')

### Replace any NaN financial data with a 0
df.fillna(value=0, inplace=True)

### Store to my_dataset for easy export below.
my_dataset = df.to_dict(orient='index')

### Return statistics about the dataset
non_poi, poi = df.poi.value_counts()
lines, columns = df.shape[0], df.shape[1]
print 'number of data points: %s' % (lines * columns)
print 'number of POIs and non-POIs: %s, %s' % (poi, non_poi)
print 'number of features used: %s' % len(features_list)
print 'features with missing values: see plot'
count_nan = []
for label in df.columns:
    miss_val = (df[label] == 0).sum(axis=0)
    if df[label].dtype == 'bool':
        count_nan.append(0)
    else:
        count_nan.append(miss_val)
# Plot missing values distribution by columns
# plt.bar(df.columns, count_nan)
# plt.xlabel('Data columns', fontsize=10)
# plt.ylabel('Proportion of missing data', fontsize=10)
# plt.xticks(df.columns, df.columns, rotation=90, fontsize=8)
# plt.show()

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Create list with percentiles to test
percentile = list(range(1, 100))
min_split = list(range(2, 100))
estimator = list(range(10,100,10))

# Create pipeline to test feature selection and the algorithms
# DecisionTree
# pipeline_dt = Pipeline([('select', SelectPercentile()), ('dt', DecisionTreeClassifier())])
# grids = GridSearchCV(pipeline_dt, {
#     'select__percentile': percentile, 'dt__min_samples_split': min_split}, cv=5, iid=0, 
#     scoring='precision')

# GaussianNB
# pipeline_gnb = Pipeline([('select', SelectPercentile()), ('gnb', GaussianNB())])
# grids = GridSearchCV(pipeline_gnb, {
#     'select__percentile': percentile}, cv=5, iid=0, scoring='precision')

# AdaBoostClassifier
pipeline_ab = Pipeline([('select', SelectPercentile()), ('ab', AdaBoostClassifier())])
grids = GridSearchCV(pipeline_ab, {
    'select__percentile': percentile, 'ab__n_estimators': estimator}, cv=5, iid=0, scoring='precision')

# General code to fit and print results
grids.fit(features, labels)
print 'Best score: ', grids.best_score_
print 'Best parameter no: ', grids.best_params_

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# clf = AdaBoostClassifier(n_estimators=30)
# clf = SVC(kernel='linear', gamma='auto')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Select features with SelectPercentile
select = SelectPercentile(percentile=67)
select.fit(features_train, labels_train)
# Tranform features and labels
features_train_selected = select.transform(features_train)
features_test_selected = select.transform(features_test)

# Create Decision Tree's model
# clf = DecisionTreeClassifier(min_samples_split=3)
# Best score: 0.4647619047619047
# Best parameter no: dt__min_samples_split: 3, select__percentile: 25
# Accuracy: 0.81107	Precision: 0.26652	
# Recall: 0.23800	F1: 0.25145

# Create GaussianNB's model
# clf = GaussianNB()
# Best score:  0.4833333333333333
# Best parameter no:  {'select__percentile': 15}
# Accuracy: 0.73900	Precision: 0.22604	
# Recall: 0.39500	F1: 0.28753

# Create AdaBoostClassifier's model
clf = AdaBoostClassifier(n_estimators=40)
# Best score:  0.6444444444444445
# Best parameter no:  {'select__percentile': 67, 
# 'ab__n_estimators': 40}
# Accuracy: 0.85333	Precision: 0.43606	
# Recall: 0.34100	F1: 0.38272

# Fit and predict with the selected data
clf.fit(features_train_selected, labels_train)
pred = clf.predict(features_test_selected)

#######################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)