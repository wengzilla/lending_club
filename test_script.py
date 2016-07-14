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

util.df_binarize_column(all_data, 'home_ownership', drop_col=True)
util.df_binarize_column(all_data, 'purpose', drop_col=True)
util.df_binarize_column(all_data, 'verification_status', drop_col=True)
all_data['loan_status'] = (all_data['loan_status'] == 'Fully Paid').astype(int)


drop_feats = cd.drop_features()
all_data_dropped_cols = all_data.drop(drop_feats, axis=1)
X, y = cd.get_X_y(all_data_dropped_cols)

lr_params = {
    'C': 1,
    'class_weight': {False: 10},
    'penalty': 'l2',    
}

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=0)
models = group_models.fit(X_test, y_test, 'grade', LogisticRegression, lr_params)

print benchmark.total_return(all_data, filter_col='grade', filter_vals=['A', 'B', 'C', 'D', 'E'])
# X = all_data but then prepped for model
# create a new dataframe of only the things we chose

indices = group_models.predicted_indices(X, 'grade', models)
predicted_df = all_data.ix[indices]
print benchmark.total_return(predicted_df)
