from __future__ import division
import pandas as pd
import numpy as np
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
        print "Making group for group %s" %gname
        X_group = X_group.drop(group_feat, axis=1)
        y_group = y.ix[X_group.index]
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

def group_conf_matrices(X, y, group_feat, group_models):
    pass

def group_results(X, y, payouts, group_feat, group_models):
    grouped = X.groupby(group_feat)
    all_indices = []
    true_pos = 0
    false_pos = 0
 
    for gname, group in grouped:
        X_group = group.drop('grade', axis=1)
        y_group = y.ix[X_group.index]

        predicted = group_models[gname].predict(X_group)
        cm = confusion_matrix(y_group, predicted, labels=[True, False])
        precision = cm[0,0] / np.sum(cm[:, 0])
        indices = [X_group.index[i] for i in range(len(X_group)) if predicted[i] == 1]
        all_indices.extend(indices)
        #if cm.shape[0] >= 2:
        true_pos += cm[0,0]
        false_pos += cm[1,0]
        print "Precision for group %s: %.2f" %(gname, precision)
        print "Benchmark for group %s: %.2f" %(gname, np.mean(payouts.ix[indices]))
        print cm

    print "Overall precision: %d %d" %(true_pos, false_pos)
    print "Overall benchmark: %.2f" %np.mean(payouts.ix[all_indices])
