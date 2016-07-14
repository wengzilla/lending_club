from __future__ import division
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import util
import seaborn as sns
import clean_data as cd
import pdb
TARGET = 'loan_status'

'''
def make_models(model_func, params, groups):
    groups = {}
    for g, _ in groups:
        model = model_func(**params)
        groups[] = model
    return groups
'''

# Make a model for each group and return it as a dict
def fit(X, y, group_feat, model_func, params):
    models = {}

    for gname, X_group in X.groupby(group_feat):
        X_group = X_group.drop(group_feat, axis=1)
        y_group = y.loc[X_group.index]
        model = model_func(**params)
        model.fit(X_group, y_group)
        models[gname] = model
    return models

# Return dicts of scores and confusion matrices keyed on the groupby feature
def grouped_loss(X, y, group_feat, group_models):
    grouped = combined_df.groupby(group_feat)
    scores = {}
    cms = {}

    for gname, X_group in grouped.groups.keys():
        X_group = X_group.drop(group_feat, axis=1)
        y_group = y.loc[X_group.index]
        predictions = group_models[gname].predict(X_group)
        scores[gname] = group_models[gname].score(X_group, y_group)
        cms[gname] = confusion_matrix(y_group, predictions, labels=[True, False])

    return scores, cms

# X, are all indexed by id
def predicted_indices(X, group_feat, group_models):
    indices = []
    grouped = X.groupby(group_feat)
     
    for gname, X_group in grouped:
        X_group = X_group.drop(group_feat, axis=1)
        predicted = group_models[gname].predict(X_group)

        good_indices = [X_group.index[i] for i in range(len(X_group)) if predicted[i] == 1]
        indices.extend(good_indices)
        print("Selected %d things from model on group %s" %(len(good_indices), gname))

    return indices
