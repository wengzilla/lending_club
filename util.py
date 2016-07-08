from __future__ import division
import pandas as pd
import numpy as np
import bisect

# Takes a continuous feature and turns it into a categorical feature
# by bucketing values based on quantile or range.
def gen_categorical(category, df, max_categories=10, quantile=False):
    max_val = df[category].max()
    min_val = df[category].min()

    if not quantile:
        bucket_size = (max_val - min_val) / max_categories
        buckets = [min_val + i*bucket_size for i in range(1, max_categories)]
    else:
        percentiles = np.arange(0, 1, 1.0/max_categories)
        buckets = np.percentile(df[category], percentiles)
    return df[category].apply(lambda x: bisect.bisect_left(buckets, x))

# Adds a binary feature for each category of a categorical feature
def binarize_category(category, df, drop_category=False):
    grouped = df.groupby(category)
    groups = grouped.groups.keys()

    for g in groups:
        cat_name = category + '_' + g
        df[cat_name] = (df[category] == g).astype(int)

    if drop_category:
       df.drop(category, axis=1, inplace=True)

def normalize_column(category, df, inplace=False):
    mean = np.nanmean(df[category])
    col_max = np.max(df[category])
    col_min = np.min(df[category])
    normed = df[category].apply(lambda x: x if pd.isnull(x) else (x - mean)/(col_max-col_min))

    if inplace:
        df[category] = normed
    else:
        return normed 

def sq_loss(true_y, predicted_y):
    return np.sum(np.square(true_y - predicted_y))

def clean_percent(val):
    if pd.isnull(val):
        return val
    return float(val[:-1])

