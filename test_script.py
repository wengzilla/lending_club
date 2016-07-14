from __future__ import division
import pdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import util
import seaborn as sns
import clean_data as cd
import benchmark
import group_models


pd2007 = cd.import_data(2007)
pd2012 = cd.import_data(2012)

all_data = pd.concat([pd2007, pd2012])
prepped_data = cd.prep_for_classification(all_data)
X, y = cd.feature_target_split(prepped_data)

# grades are still here
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
train_payouts = all_data.ix[X_train.index]['payout_prop']
test_payouts = all_data.ix[X_test.index]['payout_prop']

print "Benchmark for all data on A/B/C: %.2f" %benchmark.total_return(all_data, filter_col='grade', filter_vals=['A', 'B', 'C'])
print "Benchmar for all data on all loans: %.2f" %benchmark.total_return(all_data)
print "Benchmark for all data is : %.2f" %np.mean(all_data['payout_prop'])
print "Benchmark for 2012 data is: %.2f" %np.mean(pd2012['payout_prop'])
print "Benchmark for 2007 data is: %.2f" %np.mean(pd2007['payout_prop'])

lr_params = {
    'C': 100,
    'class_weight': {False: 12},
    'penalty': 'l2',    
}

models = group_models.fit(X_train, y_train, 'grade', LogisticRegression, lr_params)
group_models.group_results(X_test, y_test, test_payouts, 'grade', models)
