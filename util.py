from __future__ import division
import pandas as pd
import numpy as np
import bisect
import pdb

# Takes a continuous feature and turns it into a categorical feature
# by bucketing values based on quantile or range.
def gen_categorical(df, col, max_categories=10, quantile=False):
    max_val = df[col].max()
    min_val = df[col].min()

    if not quantile:
        bucket_size = (max_val - min_val) / max_categories
        buckets = [min_val + i*bucket_size for i in range(1, max_categories)]
    else:
        percentiles = np.arange(0, 1, 1.0/max_categories)
        buckets = np.percentile(df[col], percentiles)
    return df[col].apply(lambda x: bisect.bisect_left(buckets, x))

# Adds a binary feature for each category of a categorical feature
def df_binarize_column(df, col, drop_col=False):
    grouped = df.groupby(col)
    groups = grouped.groups.keys()

    for g in groups:
        cat_name = col + '_' + g
        df[cat_name] = (df[col] == g).astype(int)

    if drop_col:
       df.drop(col, axis=1, inplace=True)

def df_normalize_column(df, col, inplace=False):
    mean = np.nanmean(df[col])
    col_max = np.max(df[col])
    col_min = np.min(df[col])
    normed = df[col].apply(lambda x: x if pd.isnull(x) else (x - mean)/(col_max-col_min))

    if inplace:
        df[category] = normed
    else:
        return normed 

###############################################################################
# Functions to clean a data frame
###############################################################################
def df_clean_percent(df, cols, inplace=False):
    if inplace:
        df[cols] = df[cols].replace(to_replace='%', value='', inplace=False, regex=True)
        df[cols] = df[cols].astype(float)
    else:
        return df[col].replace(to_replace='%', value='', inplace=False).astype(float)

def df_clean_month(df, cols, inplace=False):
    if inplace:
        df[cols] = df[cols].replace(to_replace=' months', value='', inplace=False, regex=True)
        df[cols] = df[cols].astype(int)
    else:
        return df[col].replace(to_replace=' months', value='', inplace=False, regex=True).astype(float)

def df_clean_emp_length(df, cols, inplace=False):
    if inplace:
        df[cols] = df[cols].replace(to_replace='< 1 year', value='0', inplace=False, regex=True)
        df[cols] = df[cols].replace(to_replace='[^0-9]+', value='', inplace=False, regex=True)
        df[cols].fillna(0, axis=0, inplace=True)
        df[cols] = df[cols].astype(int)
    else:
        new_col = df[cols].replace(to_replace='[^0-9]+', value='', inplace=False, regex=True)
        new_col = new_col.fillna(0, axis=0, inplace=True).astype(int)
        return new_col

def df_clean_date(df, col, inplace=False):
    if inplace:
        df[col] = pd.to_datetime(df[col])
    else:
        return pd.to_datetime(df[col])

