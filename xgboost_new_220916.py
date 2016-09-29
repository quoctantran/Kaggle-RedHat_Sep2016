# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:05:02 2016
@author: quoctanvn
"""

#---------------------------------------------------------------------------
# KAGGLE - RED HAT COMPETITION
#---------------------------------------------------------------------------

from __future__ import print_function, division

import numpy as np 
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import os
os.chdir('E:/Kaggle/RedHat')

#---------------------------------------------------------------------------
# My data explore functions, mimic str() function of R

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

# Function to print out unique variables and its percentages in data
def proportion_print(pd_series):
    df = pd.DataFrame({pd_series.name:pd_series,
                       'count':1,
                       'percent':1/len(pd_series)})
    print(df.groupby(pd_series.name).sum())

#---------------------------------------------------------------------------
# Explore data (e.g. type, NA, etc.)

# Train data
act_train = pd.read_csv('input/act_train.csv')
strdf(act_train)
nadf(act_train)

people = pd.read_csv('input/people.csv')
strdf(people)
nadf(people)

# Check balanced class [55%:44%]
proportion_print(act_train.outcome)

#---------------------------------------------------------------------------
# Import data

# Read data from files
act_train = pd.read_csv('input/act_train.csv', parse_dates=['date'])
act_test = pd.read_csv('input/act_test.csv', parse_dates=['date'])
people = pd.read_csv('input/people.csv', parse_dates=['date'])

# Fill NA values
nadf(act_train)
nadf(act_test)
nadf(people)

act_train.fillna('Missing', inplace=True)
act_test.fillna('Missing', inplace=True)

# Merge data
train = act_train.merge(people, on='people_id', how='left', left_index=True)
test = act_test.merge(people, on='people_id', how='left', left_index=True)

# Drop used columns
y = train.outcome
test_id = test.activity_id
train.drop(['outcome', 'people_id', 'activity_id'], axis=1, inplace=True)
test.drop(['people_id', 'activity_id'], axis=1, inplace=True)

#---------------------------------------------------------------------------
# Preprocessing data

# Extract datetime variables
datetime_vars = ['date_x', 'date_y']
for v in datetime_vars:
    # Convert for train data
    train[v + '_year'] = train[v].dt.year
    train[v + '_month'] = train[v].dt.month
    train[v + '_day'] = train[v].dt.day
    train[v + '_isweekend'] = (train[v].dt.weekday >= 5).astype(np.int8)
    train.drop(v, axis=1, inplace=True)
    
    # Convert for test data
    test[v + '_year'] = test[v].dt.year
    test[v + '_month'] = test[v].dt.month
    test[v + '_day'] = test[v].dt.day
    test[v + '_isweekend'] = (test[v].dt.weekday >= 5).astype(np.int8)
    test.drop(v, axis=1, inplace=True)

# Convert bool variables to int8
bool_vars = list(train.columns[train.dtypes == 'bool'].values)
for v in bool_vars:
    train[v] = train[v].astype(np.int8)
    test[v] = test[v].astype(np.int8)

# Convert categorical variables in text to integer
categorical_vars = list(train.columns[train.dtypes == 'object'].values)
for v in categorical_vars:
    enc = LabelEncoder().fit(pd.concat((train[v], test[v])))
    train[v] = enc.transform(train[v])
    test[v] = enc.transform(test[v])

# Convert categorical variables to dummy, very high dimension
enc = OneHotEncoder().fit(pd.concat((train[categorical_vars], test[categorical_vars])))
train_cat_dummy = enc.transform(train[categorical_vars])
test_cat_dummy = enc.transform(test[categorical_vars])

# Merge all variables to sparse matrix
non_categorical_vars = [v for v in list(train.columns.values)
                        if v not in categorical_vars]
X_train = sparse.hstack((train[non_categorical_vars], train_cat_dummy))
X_test = sparse.hstack((test[non_categorical_vars], test_cat_dummy))

#---------------------------------------------------------------------------
# Dimensional reduction

from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC
svc = LinearSVC(penalty='l1', dual=False).fit(X_train, y)
model = SelectFromModel(svc, prefit=True)
X_train_new = model.transform(X_train)
X_test_new = model.transform(X_test)

#from sklearn.decomposition import TruncatedSVD
#svd = TruncatedSVD(n_components=100, random_state=123).fit(X_train) 
#svd.explained_variance_ratio_.sum()

from sklearn.linear_model import LassoCV
clf = LassoCV().fit(X_train, y)
model = SelectFromModel(clf, threshold=0.25)
#n_features = model.transform(X_train).shape[1]

#---------------------------------------------------------------------------
# Train ưith xgboost model
  
dtrain = xgb.DMatrix(X_train_new, label=y)
dtest = xgb.DMatrix(X_test_new)

# Set xgboost params
param = {'max_depth':10,
         'eta':0.02,
         'silent':1,
         'objective':'binary:logistic',
         'eval_metric':'auc',
         'subsample':0.7,
         'colsample_bytree':0.7,
         'min_child_weight':0,
         'booster':"gblinear"}

# Run cross-validation to have an idea about the final result
watchlist = [(dtrain, 'train')]
num_round = 50
num_fold = 3
early_stopping_rounds = 10
xgb_cv = xgb.cv(param, dtrain, num_round, num_fold, watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=True, seed=123)

# Actual train the xgboost model on all training set
watchlist = [(dtrain, 'train')]
num_round = 300
early_stopping_rounds = 10
xgb_model = xgb.train(param, dtrain, num_round, watchlist,
                      early_stopping_rounds=early_stopping_rounds)

# Apply the model on test set and save the output
y_pred = xgb_model.predict(dtest)
output = pd.DataFrame({'activity_id':test_id, 'outcome':y_pred})
output.head()
output.to_csv('submission_xgboost_180916_1.csv', index=False)
