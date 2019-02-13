#!/usr/bin/python

import sys
sys.path.append("../tools/")

from tester import dump_classifier_and_data, test_classifier
from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tester
import pickle


def return_grid_result(grids, features, labels):
    ''' General code to fit and print results from GridSearch '''
    grids.fit(features, labels)
    print 'Best score: ', grids.best_score_
    print 'Best parameter no: ', grids.best_params_
    selected_percentile = grids.best_params_['select__percentile']

    return grids, selected_percentile

def fit_and_predict(features_train_selected, labels_train, features_test_selected, clf):
    ''' Fit model and predict test dataset '''
    clf.fit(features_train_selected, labels_train)
    pred = clf.predict(features_test_selected)

    return clf, pred

def select_best_perc(selected_percentile, features_train, labels_train):
    ''' Select features with SelectPercentile '''
    select = SelectPercentile(percentile=selected_percentile)
    # Fit data
    select.fit(features_train, labels_train)
    # Get features score
    feature_scores = np.array(select.scores_)
    mask = select.get_support()
    # Transform features and labels
    features_train_selected = select.transform(features_train)
    features_test_selected = select.transform(features_test)

    return mask, features_train_selected, features_test_selected, feature_scores



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
# features_list = ['poi'] + payment_data + stock_data + email_data 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Load dict into pandas df, replace NaN string for np.nan values
df = pd.DataFrame.from_dict(data_dict, orient='index')
# Create list with columns, remove email and poi from  it
column_list = list(df.columns)
column_list.remove('poi')
column_list.remove('email_address')
# Create feature list with poi at first position
features_list = ['poi'] + column_list

# Replance NaN string with np.nan and filter df
# df = df.replace('NaN', np.nan)
df = df[features_list]

### Replace missing values in financial data with zeros
# df[payment_data] = df[payment_data].fillna(value=0)
# df[stock_data] = df[stock_data].fillna(value=0)
df[email_data] = df[email_data].replace('NaN', 0)

### Task 2: Remove outliers
df.drop(axis=0, labels=['TOTAL','THE TRAVEL AGENCY IN THE PARK'], inplace=True)

### Task 3: Create new feature(s)
# Add the new email features to the dataframe
# df['to_poi_average'] = df['from_poi_to_this_person'].astype(int) / df['to_messages'].astype(int)
# df['from_poi_average'] = df['from_this_person_to_poi'].astype(int) / df['from_messages'].astype(int)
# df['shared_poi_average'] = df['shared_receipt_with_poi'].astype(int) / df['to_messages'].astype(int)

# # Add the new features to the features list
# features_list.append('to_poi_average')
# features_list.append('from_poi_average')
# features_list.append('shared_poi_average')

### Replace any NaN financial data with a 0
df.fillna(value=0, inplace=True)

### Store to my_dataset for easy export below.
my_dataset = df.to_dict(orient='index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Create lists with variable parameters to test
percentile = list(range(1, 100))
min_split = list(range(2, 100))
estimator = list(range(10,100,10))
neighbors = list(range(1,30))
distances = list(range(1, 5))


# Create pipeline to test feature selection and the algorithms
# KNN
print 'Starting pipeline for KNN model'
pipeline_knn = Pipeline([('select', SelectPercentile()), ('knn', KNeighborsClassifier())])
grids_knn = GridSearchCV(pipeline_knn, {'select__percentile': percentile, 
'knn__n_neighbors': neighbors, 'knn__p': distances}, cv=5, iid=0, scoring='f1')
# Variables for KNN
grids_knn, selected_percentile_knn = return_grid_result(grids_knn, features, labels)
selected_neighbors = grids_knn.best_params_['knn__n_neighbors']
selected_distance = grids_knn.best_params_['knn__p']
best_score_knn = grids_knn.best_score_

# # DecisionTree
# print 'Starting pipeline for DecisionTree model'
# pipeline_dt = Pipeline([('select', SelectPercentile()), ('dt', DecisionTreeClassifier())])
# grids_dt = GridSearchCV(pipeline_dt, {
#     'select__percentile': percentile, 'dt__min_samples_split': min_split}, cv=5, iid=0, 
#     scoring='f1')
# # Variables for DecisitonTree
# grids_dt, selected_percentile_dt = return_grid_result(grids_dt, features, labels)
# selected_min_samples_split = grids_dt.best_params_[
#     'dt__min_samples_split']
# best_score_dt = grids_dt.best_score_

# # GaussianNB
# print 'Starting pipeline for GaussianNB model'
# pipeline_gnb = Pipeline([('select', SelectPercentile()), ('gnb', GaussianNB())])
# grids_gnb = GridSearchCV(pipeline_gnb, {
#     'select__percentile': percentile}, cv=5, iid=0, scoring='f1')
# # Variables for GaussianNB
# grids_gnb, selected_percentile_gnb = return_grid_result(grids_gnb, features, labels)
# best_score_gnb = grids_gnb.best_score_

# # AdaBoostClassifier
# print 'Starting pipeline for AdaBoostClassifier model'
# pipeline_ab = Pipeline([('select', SelectPercentile()), ('ab', AdaBoostClassifier())])
# grids_ab = GridSearchCV(pipeline_ab, {
#     'select__percentile': percentile, 'ab__n_estimators': estimator}, cv=5, iid=0,
#     scoring='f1')
# # Variables for AdaBoostClassifier
# grids_ab, selected_percentile_ab = return_grid_result(grids_ab, features, labels)
# selected_n_estimators = grids_ab.best_params_['ab__n_estimators']
# best_score_ab = grids_ab.best_score_

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Select features with SelectPercent
# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Check which had the highest F1-score
# if (best_score_knn > best_score_dt) and (best_score_knn > best_score_gnb) and (
#     best_score_knn > best_score_ab):
#     print 'The best model was KNN'
# Create KNNs's model
clf = KNeighborsClassifier(n_neighbors=selected_neighbors,
p=selected_distance)
# Select the best features percentile
mask, features_train_selected, features_test_selected, feature_scores = select_best_perc(
    selected_percentile_knn, features_train, labels_train)
# Fit and predict with the selected data
clf, pred = fit_and_predict(
    features_train_selected, labels_train, features_test_selected, clf)
# elif (best_score_dt > best_score_gnb) and (best_score_dt > best_score_ab):
#     print 'The best model was DecisionTree'
#     # Create Decision Tree's model
#     clf = DecisionTreeClassifier(
#         min_samples_split=selected_min_samples_split)
#     # Select the best percentile
#     mask, features_train_selected, features_test_selected, feature_scores = select_best_perc(
#         selected_percentile_dt, features_train, labels_train)
#     # Fit and predict with the selected data
#     clf, pred = fit_and_predict(
#         features_train_selected, labels_train, features_test_selected, clf)
# elif best_score_gnb > best_score_ab:
#     print 'The best model was GaussianNB'
#     # Create GaussianNB's model
#     clf = GaussianNB()
#     # Select the best percentile
#     mask, features_train_selected, features_test_selected, feature_scores = select_best_perc(
#         selected_percentile_gnb, features_train, labels_train)
#     # Fit and predict with the selected data
#     clf, pred_nb = fit_and_predict(
#         features_train_selected, labels_train, features_test_selected, clf)
# else:
#     print 'The best model was AdaBoostClassifier'
#     # Create AdaBoostClassifier's model
#     clf = AdaBoostClassifier(n_estimators=selected_n_estimators)
#     # Select the best percentile
#     mask, features_train_selected, features_test_selected, feature_scores = select_best_perc(
#         selected_percentile_ab, features_train, labels_train)
#     # Fit and predict with the selected data
#     clf, pred = fit_and_predict(
#         features_train_selected, labels_train, features_test_selected, clf)

# Features without poi and features selected
new_features_list = np.array(features_list[1:])
features_selected = new_features_list[mask]
features_list = ['poi'] + list(features_selected)

# Select features scores
feature_scores_selected = feature_scores[mask]

### Return statistics about the dataset
non_poi, poi = df.poi.value_counts()
lines, columns = df.shape[0], df.shape[1]
print 'number of data points: %s' % (lines * columns)
print 'number of POIs and non-POIs: %s, %s' % (poi, non_poi)
print 'number of features used: %s' % len(features_list)
print 'features used and there score:\n'
for feature, score in zip(features_selected, feature_scores_selected):
    print 'Feature: %s, Score: %s' %  (feature, score)

#######################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
# test_classifier(clf, my_dataset, features_list)