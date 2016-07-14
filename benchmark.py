from __future__ import division
import pandas as pd
import numpy as np

def compute_returns(df):
    return (df['total_pymnt'] - df['collection_recovery_fee']) / df['loan_amnt']
    
def total_return(df, filter_col=None, filter_vals=None):
    if filter_col == None:
        return np.average(compute_returns(df))
    else:
        filtered_df = df.loc[df[filter_col].isin(filter_vals)] 
        return np.average(compute_returns(filtered_df))

def total_return_model(X, y, model):
    pass
      
