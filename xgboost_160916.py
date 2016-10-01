# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:05:02 2016
"""

#---------------------------------------------------------------------------
# KAGGLE - RED HAT COMPETITION
#---------------------------------------------------------------------------

from __future__ import print_function, division

import numpy as np 
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV

import os
os.chdir('D:/Kaggle/redhat')

#---------------------------------------------------------------------------
# My functions

# Function to print out structure of data (mimic str() in R)
def strdf(df):
    print(type(df), ':\t', df.shape[0], 'obs. of', df.shape[1], 'variables:')
    if df.shape[0] > 4:
        df = df.head(4) # Take first 4 obs.
        dots = '...' # Print ... for the rest values  
    else:
        dots = ''
    space = len(max(list(df), key=len))
    for c in list(df):
        print(' $', '{:{align}{width}}'.format(c, align='<', width=space),
              ':', df[c].dtype, str(df[c].values)[1:-1], dots)

# Function to print out NAN values and their data types
def nadf(df):
    print(type(df), ':\t', df.shape[0], 'obs. of', df.shape[1], 'variables:')
    df_type = df.dtypes    
    df_NA = df.isnull().sum()
    space = len(max(list(df), key=len))
    space_type = len(max([d.name for d in df.dtypes.values], key=len))
    space_NA = len(str(max(df_NA)))
    for c in list(df):
        print(' $', '{:{align}{width}}'.format(c, align='<', width=space),
              ':', '{:{align}{width}}'.format(df_type[c], align='<', width=space_type),
              '{:{align}{width}}'.format(df_NA[c], align='>', width=space_NA),
              'NAN value(s)')

# Function to convert categorical, boolean, datetime variables to integer
def var_to_int(df):
    X = df.copy(deep=True)
    vars_type = X.dtypes

    # Convert categorical vars
    categorical_list = list(X.columns[vars_type == 'object'].values)
    for c in categorical_list:
        X[c].fillna('NAN', inplace=True)
        X[c] = pd.factorize(X[c])[0]
    
    # Convert boolean vars
    bool_list = list(X.columns[vars_type == 'bool'].values)
    for c in bool_list: X[c] = X[c].astype(np.int8)
        
    # Convert datetime vars
    datetime_list = list(X.columns[vars_type == 'datetime64[ns]'].values)
    for c in datetime_list:
        X[str(c) + '_day'] = X[c].dt.day
        X[str(c) + '_month'] = X[c].dt.month
        X[str(c) + '_year'] = X[c].dt.year
        X[str(c) + '_isweekend'] = (X[c].dt.weekday >= 5).astype(np.int8)
        X = X.drop(c, axis=1)
    
    return X
        
#---------------------------------------------------------------------------
# Import data

# Read data from files
act_train = pd.read_csv('input/act_train.csv', parse_dates=['date'])
act_test = pd.read_csv('input/act_test.csv', parse_dates=['date'])
people = pd.read_csv('input/people.csv', parse_dates=['date'])

# Describe data
strdf(act_train)
strdf(act_test)
strdf(people)

# Check NA values
nadf(act_train)
nadf(act_test)
nadf(people)

#---------------------------------------------------------------------------
# Preprocess data

# Merge data
train = act_train.merge(people, on='people_id', how='left', left_index=True)
test = act_test.merge(people, on='people_id', how='left', left_index=True)

# Encode categorical variables from string to integer
y = train.outcome
test_activity_id = test.activity_id
train = train.drop(['outcome'], axis=1)

X = pd.concat([train, test], ignore_index=True)
X = X.drop(['people_id', 'activity_id'], axis=1)

X_int = var_to_int(X)

# Encode categorical variables from integer to dummy
cat_vars = list(X.columns[X.dtypes == 'object'].values)
non_cat_vars = [c for c in X_int.columns.values if c not in cat_vars]

enc = OneHotEncoder().fit(X_int[cat_vars])

X_train_int = X_int[:len(train)]
X_test_int = X_int[len(train):]

X_train_sparse = enc.transform(X_train_int[cat_vars])
X_test_sparse = enc.transform(X_test_int[cat_vars])

X_train = sparse.hstack((X_train_int[non_cat_vars], X_train_sparse))
X_test = sparse.hstack((X_test_int[non_cat_vars], X_test_sparse))

# Remove unused vars to save space
del X, X_int, train, test, X_train_int, X_test_int
del X_train_sparse, X_test_sparse
del act_train, act_test, people
  
#---------------------------------------------------------------------------
# Train simple xgboost model
  
dtrain = xgb.DMatrix(X_train, label=y)
dtest = xgb.DMatrix(X_test)

# Set xgboost params
param = {'max_depth':10, 'eta':0.02, 'silent':1, 'objective':'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['min_child_weight'] = 0
param['booster'] = "gblinear"

# Run cross-validation to have an idea about the final result
watchlist = [(dtrain, 'train'), (dtest, 'test')]
num_round = 50
num_fold = 5
early_stopping_rounds = 10
xgb_cv = xgb.cv(param, dtrain, num_round, num_fold, watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=True, seed=0)

# Actual train the xgboost model on all training set
watchlist = [(dtrain, 'train')]
num_round = 300
early_stopping_rounds = 10
xgb_model = xgb.train(param, dtrain, num_round, watchlist,
                      early_stopping_rounds=early_stopping_rounds)

# Apply the model on test set and save the output
pred = xgb_model.predict(dtest)
output = pd.DataFrame({'activity_id':test_activity_id, 'outcome':pred})
output.head()
output.to_csv('submission_xgboost_150916_2.csv', index=False)

#---------------------------------------------------------------------------
# Train xgboost + dimensional reduction

# Dimensional reduction using LinearSVC
from sklearn.svm import LinearSVC
svc = LinearSVC(penalty="l1", dual=False).fit(X_train, y)

X_train_svc = svc.transform(X_train)
X_test_svc = svc.transform(X_test)

X_train_svc.shape
X_test_svc.shape

dtrain = xgb.DMatrix(X_train_svc, label=y)
dtest = xgb.DMatrix(X_test_svc)

# Set xgboost params
param = {'max_depth':10, 'eta':0.02, 'silent':1, 'objective':'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['min_child_weight'] = 0
param['booster'] = "gblinear"

# Run cross-validation to have an idea about the final result
watchlist = [(dtrain, 'train'), (dtest, 'test')]
num_round = 50
num_fold = 5
early_stopping_rounds = 10
xgb_cv = xgb.cv(param, dtrain, num_round, num_fold, watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=True, seed=0)

# Actual train the xgboost model on all training set
watchlist = [(dtrain, 'train')]
num_round = 300
early_stopping_rounds = 10
xgb_model = xgb.train(param, dtrain, num_round, watchlist,
                      early_stopping_rounds=early_stopping_rounds)

# Apply the model on test set and save the output
pred = xgb_model.predict(dtest)
output = pd.DataFrame({'activity_id':test_activity_id, 'outcome':pred})
output.head()
output.to_csv('submission_xgboost_svc_150916_1.csv', index=False)

#---------------------------------------------------------------------------
# Train xgboost + RandomizedSearchCV

from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

# Split data sets, prepare to train model
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.3,
                                                    random_state=123, stratify=y)

# Train the model with xgboost + GridSearchCV
scoring_type = 'roc_auc'
xgb_model = xgb.XGBClassifier({'nthread':4,
                               'silent':0,
                               'learning_rate':0.1,
                               'subsample':0.5,
                               'colsample_bytree':0.8,
                               'objective':'binary:logistic',
                               'min_child_weight':0,
                               'seed':123})
tuned_parameters =  {'max_depth':[12,15,20],
                     'n_estimators':[300,500,1000]}
clf_xgb = GridSearchCV(xgb_model, tuned_parameters, cv=2,
                       scoring=scoring_type, verbose=10)
clf_xgb.fit(X_train, y_train)

print(clf_xgb.best_score_)
print(clf_xgb.best_params_)

# Evaluation and plot ROC curve
from sklearn.metrics import auc, roc_curve
target_pred_xgb_proba = clf_xgb.predict_proba(X_test)[:,1]
#target_pred_xgb_proba = clf_xgb.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, target_pred_xgb_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.title('ROC curve (area = {0:0.2f})'.format(roc_auc))
plt.plot(fpr, tpr, linewidth=2)
