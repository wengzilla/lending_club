from __future__ import division
import pandas as pd
import numpy as np
import bisect

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

def normalize_column(category, df, inplace=False):
    mean = np.nanmean(df[category])
    col_max = np.max(df[category])
    col_min = np.min(df[category])
    normed = df[category].apply(lambda x: x if pd.isnan(x) else (x - mean)/(col_max-col_min), axis=1)

    if inplace:
        df[category] = normed
    else:
        return normed 

def sq_loss(true_y, predicted_y):
    return np.sum(np.square(true_y - predicted_y))

def clean_percent(val):
    return float(val[:-1])

